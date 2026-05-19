"""SQLite + NumPy storage backend for MemPalace.

Replaces ChromaDB to eliminate HNSW index corruption under concurrent
multi-process access. Uses SQLite WAL mode for crash-safe multi-reader
+ serialized-writer concurrency, and NumPy brute-force for vector
similarity search (~30ms at 479K vectors on Apple Silicon).

No external daemons, no extension loading, works on system Python 3.9.6.

Architecture decisions (plan 005):
- WAL mode + BEGIN IMMEDIATE — no flock, no external locking
- NumPy brute-force — no sqlite-vec (blocked on macOS Python 3.9)
- FTS5 for $contains text search
- Separate VectorIndex per collection
- Event bus with junction table for cross-session orchestration
"""

import atexit
import json
import logging
import os
import sqlite3
import struct
import time
from typing import ClassVar, Optional

import numpy as np

from .base import (
    BaseBackend,
    BaseCollection,
    BackendClosedError,
    DimensionMismatchError,
    GetResult,
    HealthStatus,
    PalaceRef,
    QueryResult,
    UnsupportedFilterError,
    _IncludeSpec,
)

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 384
DB_FILENAME = "mempalace.db"
SCHEMA_VERSION = 1

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    collection TEXT NOT NULL,
    wing TEXT,
    room TEXT,
    document TEXT NOT NULL,
    metadata JSON NOT NULL DEFAULT '{}',
    embedding BLOB NOT NULL,
    created_at REAL NOT NULL DEFAULT (unixepoch('subsec')),
    updated_at REAL NOT NULL DEFAULT (unixepoch('subsec'))
);

CREATE INDEX IF NOT EXISTS idx_memories_collection ON memories(collection);
CREATE INDEX IF NOT EXISTS idx_memories_wing_room ON memories(wing, room);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    document,
    content=memories,
    content_rowid=rowid,
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, document) VALUES (new.rowid, new.document);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, document)
        VALUES ('delete', old.rowid, old.document);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, document)
        VALUES ('delete', old.rowid, old.document);
    INSERT INTO memories_fts(rowid, document) VALUES (new.rowid, new.document);
END;

CREATE TABLE IF NOT EXISTS event_bus (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload JSON,
    created_at REAL NOT NULL DEFAULT (unixepoch('subsec'))
);

CREATE TABLE IF NOT EXISTS event_consumption (
    event_id INTEGER NOT NULL REFERENCES event_bus(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL,
    consumed_at REAL NOT NULL DEFAULT (unixepoch('subsec')),
    PRIMARY KEY (event_id, session_id)
);

CREATE INDEX IF NOT EXISTS idx_event_bus_type ON event_bus(event_type, created_at);
CREATE INDEX IF NOT EXISTS idx_event_bus_created ON event_bus(created_at);

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at REAL NOT NULL DEFAULT (unixepoch('subsec'))
);
INSERT OR IGNORE INTO schema_version(version) VALUES (1);
"""


# ---------------------------------------------------------------------------
# Filter translator: Chroma DSL → SQL WHERE
# ---------------------------------------------------------------------------

_OP_SQL = {
    "$eq": "=",
    "$ne": "!=",
    "$gt": ">",
    "$gte": ">=",
    "$lt": "<",
    "$lte": "<=",
}


def _build_where_clause(where: Optional[dict]) -> tuple[str, list]:
    """Translate a Chroma-style where filter to a SQL WHERE fragment.

    Supports: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $and, $or.
    Bare values are treated as implicit $eq.

    Args:
        where: Chroma-style filter dict, or None.

    Returns:
        Tuple of (sql_fragment, params). sql_fragment is empty string if
        no filter was provided.

    Raises:
        UnsupportedFilterError: If an unknown operator is encountered.
    """
    if not where:
        return "", []

    clauses = []
    params = []

    for key, condition in where.items():
        if key == "$and":
            parts = [_build_where_clause(sub) for sub in condition]
            combined = " AND ".join(p[0] for p in parts if p[0])
            if combined:
                clauses.append(f"({combined})")
            for p in parts:
                params.extend(p[1])
        elif key == "$or":
            parts = [_build_where_clause(sub) for sub in condition]
            combined = " OR ".join(p[0] for p in parts if p[0])
            if combined:
                clauses.append(f"({combined})")
            for p in parts:
                params.extend(p[1])
        elif isinstance(condition, dict):
            for op, value in condition.items():
                if op == "$in":
                    placeholders = ",".join("?" * len(value))
                    clauses.append(
                        f"json_extract(metadata, '$.{key}') IN ({placeholders})"
                    )
                    params.extend(value)
                elif op == "$nin":
                    placeholders = ",".join("?" * len(value))
                    clauses.append(
                        f"json_extract(metadata, '$.{key}') NOT IN ({placeholders})"
                    )
                    params.extend(value)
                elif op in _OP_SQL:
                    clauses.append(
                        f"json_extract(metadata, '$.{key}') {_OP_SQL[op]} ?"
                    )
                    params.append(value)
                else:
                    raise UnsupportedFilterError(
                        f"operator {op!r} not supported by sqlite backend"
                    )
        else:
            # Bare value = implicit $eq
            clauses.append(f"json_extract(metadata, '$.{key}') = ?")
            params.append(condition)

    return " AND ".join(clauses), params


def _build_where_document_clause(where_document: Optional[dict]) -> tuple[str, list]:
    """Translate a where_document filter to an FTS5 MATCH clause.

    Args:
        where_document: Dict with "$contains" key, or None.

    Returns:
        Tuple of (sql_fragment, params).

    Raises:
        UnsupportedFilterError: If an unknown operator is used.
    """
    if not where_document:
        return "", []

    if "$contains" in where_document:
        term = where_document["$contains"]
        # Quote the search term to handle multi-word phrases
        safe_term = '"' + term.replace('"', '""') + '"'
        return (
            "rowid IN (SELECT rowid FROM memories_fts WHERE memories_fts MATCH ?)",
            [safe_term],
        )

    raise UnsupportedFilterError(
        f"where_document operators {list(where_document.keys())} not supported"
    )


# ---------------------------------------------------------------------------
# VectorIndex: in-memory NumPy cosine search
# ---------------------------------------------------------------------------


class VectorIndex:
    """In-memory numpy index for fast cosine similarity search.

    Loads all embeddings for a collection into a pre-normalized matrix.
    Queries compute cosine similarity via a single matrix-vector dot product.

    Args:
        collection: Collection name to filter on.
        conn: SQLite connection to read embeddings from.
        dim: Embedding dimension (default 384 for all-MiniLM-L6-v2).
    """

    def __init__(self, collection: str, conn: sqlite3.Connection, dim: int = EMBEDDING_DIM):
        self._collection = collection
        self._conn = conn
        self._dim = dim
        self._ids: list[str] = []
        self._matrix: Optional[np.ndarray] = None
        self._dirty = True

    def _load(self):
        """Load all embeddings for this collection into memory."""
        rows = self._conn.execute(
            "SELECT id, embedding FROM memories WHERE collection = ? ORDER BY rowid",
            (self._collection,),
        ).fetchall()

        if not rows:
            self._ids = []
            self._matrix = np.empty((0, self._dim), dtype=np.float32)
        else:
            self._ids = [r[0] for r in rows]
            # Unpack embeddings from BLOBs into a contiguous array
            n = len(rows)
            self._matrix = np.empty((n, self._dim), dtype=np.float32)
            for i, row in enumerate(rows):
                self._matrix[i] = np.frombuffer(row[1], dtype=np.float32)

            # Pre-normalize for cosine similarity
            norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._matrix = self._matrix / norms

        self._dirty = False

    def invalidate(self):
        """Mark index as stale — will reload on next query."""
        self._dirty = True

    def search(
        self,
        query_embedding: list[float],
        n: int = 10,
        allowed_ids: Optional[set[str]] = None,
    ) -> list[tuple[str, float]]:
        """Return top-n (id, cosine_distance) pairs.

        Args:
            query_embedding: Query vector as a list of floats.
            n: Number of results to return.
            allowed_ids: If provided, only return results with IDs in this set.

        Returns:
            List of (id, distance) tuples sorted by ascending distance.
            Distance is cosine distance: 1.0 - cosine_similarity.
        """
        if self._dirty:
            self._load()

        if self._matrix is None or len(self._ids) == 0:
            return []

        q = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q = q / q_norm

        scores = self._matrix @ q

        if allowed_ids is not None:
            mask = np.array(
                [i for i, id_ in enumerate(self._ids) if id_ not in allowed_ids]
            )
            if len(mask) > 0:
                scores[mask] = -np.inf

        valid = np.where(np.isfinite(scores))[0]
        if len(valid) == 0:
            return []

        k = min(n, len(valid))
        top_valid = valid[np.argpartition(scores[valid], -k)[-k:]]
        top_valid = top_valid[np.argsort(scores[top_valid])[::-1]]

        return [(self._ids[i], float(1.0 - scores[i])) for i in top_valid]


# ---------------------------------------------------------------------------
# SqliteCollection
# ---------------------------------------------------------------------------


def _pack_embedding(embedding: list[float]) -> bytes:
    """Pack a float list into a bytes BLOB for SQLite storage."""
    return struct.pack(f"{len(embedding)}f", *embedding)


def _unpack_embedding(blob: bytes, dim: int = EMBEDDING_DIM) -> list[float]:
    """Unpack an embedding BLOB back into a float list."""
    return list(struct.unpack(f"{dim}f", blob))


class SqliteCollection(BaseCollection):
    """Per-collection read/write surface backed by SQLite + NumPy.

    Implements the full BaseCollection ABC including add, upsert, query,
    get, delete, count, update, and health.

    Args:
        conn: SQLite connection (WAL mode, shared across the process).
        collection_name: Name of the collection (e.g. 'mempalace_drawers').
        embed_fn: Callable that takes a list of strings and returns embeddings.
            Used to embed query_texts when query_embeddings is not provided.
        dim: Embedding dimension.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        collection_name: str,
        embed_fn=None,
        dim: int = EMBEDDING_DIM,
    ):
        self._conn = conn
        self._collection = collection_name
        self._embed_fn = embed_fn
        self._dim = dim
        self._index = VectorIndex(collection_name, conn, dim)
        self._write_count = 0

    def _checkpoint_if_needed(self):
        """Run a passive WAL checkpoint every 100 writes."""
        self._write_count += 1
        if self._write_count % 100 == 0:
            try:
                self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            except sqlite3.Error:
                pass

    def add(
        self,
        *,
        documents: list[str],
        ids: list[str],
        metadatas: Optional[list[dict]] = None,
        embeddings: Optional[list[list[float]]] = None,
    ) -> None:
        """Add new items to the collection.

        Args:
            documents: List of document texts.
            ids: List of unique IDs.
            metadatas: Optional list of metadata dicts.
            embeddings: Optional pre-computed embeddings. If None and embed_fn
                is set, embeddings are computed from documents.

        Raises:
            DimensionMismatchError: If embedding dimension doesn't match.
        """
        if embeddings is None and self._embed_fn is not None:
            embeddings = self._embed_fn(documents)
        if embeddings is None:
            raise ValueError("embeddings required (no embed_fn configured)")

        if metadatas is None:
            metadatas = [{} for _ in ids]

        rows = []
        for i, id_ in enumerate(ids):
            emb = embeddings[i]
            if len(emb) != self._dim:
                raise DimensionMismatchError(
                    f"Expected {self._dim} dims, got {len(emb)}"
                )
            meta = metadatas[i] if metadatas[i] is not None else {}
            rows.append((
                id_,
                self._collection,
                meta.get("wing"),
                meta.get("room"),
                documents[i],
                json.dumps(meta),
                _pack_embedding(emb),
            ))

        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._conn.executemany(
                "INSERT INTO memories(id, collection, wing, room, document, metadata, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

        self._index.invalidate()
        self._checkpoint_if_needed()

    def upsert(
        self,
        *,
        documents: list[str],
        ids: list[str],
        metadatas: Optional[list[dict]] = None,
        embeddings: Optional[list[list[float]]] = None,
    ) -> None:
        """Insert or replace items in the collection.

        Args:
            documents: List of document texts.
            ids: List of unique IDs.
            metadatas: Optional list of metadata dicts.
            embeddings: Optional pre-computed embeddings.

        Raises:
            DimensionMismatchError: If embedding dimension doesn't match.
        """
        if embeddings is None and self._embed_fn is not None:
            embeddings = self._embed_fn(documents)
        if embeddings is None:
            raise ValueError("embeddings required (no embed_fn configured)")

        if metadatas is None:
            metadatas = [{} for _ in ids]

        rows = []
        for i, id_ in enumerate(ids):
            emb = embeddings[i]
            if len(emb) != self._dim:
                raise DimensionMismatchError(
                    f"Expected {self._dim} dims, got {len(emb)}"
                )
            meta = metadatas[i] if metadatas[i] is not None else {}
            rows.append((
                id_,
                self._collection,
                meta.get("wing"),
                meta.get("room"),
                documents[i],
                json.dumps(meta),
                _pack_embedding(emb),
                time.time(),
            ))

        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._conn.executemany(
                "INSERT OR REPLACE INTO memories"
                "(id, collection, wing, room, document, metadata, embedding, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

        self._index.invalidate()
        self._checkpoint_if_needed()

    def query(
        self,
        *,
        query_texts: Optional[list[str]] = None,
        query_embeddings: Optional[list[list[float]]] = None,
        n_results: int = 10,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
        include: Optional[list[str]] = None,
    ) -> QueryResult:
        """Semantic similarity search.

        Args:
            query_texts: Text queries (embedded via embed_fn).
            query_embeddings: Pre-computed query embeddings.
            n_results: Number of results per query.
            where: Metadata filter in Chroma DSL.
            where_document: Document content filter ($contains).
            include: Fields to include in results.

        Returns:
            QueryResult with ids, documents, metadatas, distances.
        """
        if query_embeddings is None and query_texts is not None:
            if self._embed_fn is None:
                raise ValueError("query_texts requires embed_fn")
            query_embeddings = self._embed_fn(query_texts)

        if query_embeddings is None:
            return QueryResult.empty()

        spec = _IncludeSpec.resolve(include)

        # Pre-filter by metadata/document if filters are specified
        allowed_ids = None
        if where or where_document:
            allowed_ids = self._filter_ids(where, where_document)
            if not allowed_ids:
                return QueryResult.empty(
                    num_queries=len(query_embeddings),
                    embeddings_requested=spec.embeddings,
                )

        all_ids = []
        all_docs = []
        all_metas = []
        all_dists = []
        all_embs = [] if spec.embeddings else None

        for q_emb in query_embeddings:
            hits = self._index.search(q_emb, n=n_results, allowed_ids=allowed_ids)

            if not hits:
                all_ids.append([])
                all_docs.append([])
                all_metas.append([])
                all_dists.append([])
                if all_embs is not None:
                    all_embs.append([])
                continue

            hit_ids = [h[0] for h in hits]
            hit_dists = [h[1] for h in hits]

            # Fetch full records for the hits
            placeholders = ",".join("?" * len(hit_ids))
            inc_cols = "id, document, metadata"
            if spec.embeddings:
                inc_cols += ", embedding"

            # IDs come from VectorIndex which is already collection-scoped.
            # Omit "AND collection = ?" so SQLite uses the PK index.
            rows = self._conn.execute(
                f"SELECT {inc_cols} FROM memories "
                f"WHERE id IN ({placeholders})",
                hit_ids,
            ).fetchall()

            # Build lookup by id to preserve distance ordering
            row_map = {r[0]: r for r in rows}

            q_ids = []
            q_docs = []
            q_metas = []
            q_dists = []
            q_embs = [] if spec.embeddings else None

            for hit_id, dist in zip(hit_ids, hit_dists):
                row = row_map.get(hit_id)
                if row is None:
                    continue
                q_ids.append(row[0])
                q_docs.append(row[1] if spec.documents else "")
                q_metas.append(json.loads(row[2]) if spec.metadatas else {})
                q_dists.append(dist)
                if q_embs is not None:
                    emb_idx = 3
                    q_embs.append(_unpack_embedding(row[emb_idx], self._dim))

            all_ids.append(q_ids)
            all_docs.append(q_docs)
            all_metas.append(q_metas)
            all_dists.append(q_dists)
            if all_embs is not None:
                all_embs.append(q_embs)

        return QueryResult(
            ids=all_ids,
            documents=all_docs,
            metadatas=all_metas,
            distances=all_dists,
            embeddings=all_embs,
        )

    def _filter_ids(
        self,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
    ) -> set[str]:
        """Return IDs matching the given filters.

        Args:
            where: Metadata filter.
            where_document: Document content filter.

        Returns:
            Set of matching IDs.
        """
        clauses = ["collection = ?"]
        params = [self._collection]

        where_sql, where_params = _build_where_clause(where)
        if where_sql:
            clauses.append(where_sql)
            params.extend(where_params)

        doc_sql, doc_params = _build_where_document_clause(where_document)
        if doc_sql:
            clauses.append(doc_sql)
            params.extend(doc_params)

        sql = f"SELECT id FROM memories WHERE {' AND '.join(clauses)}"
        rows = self._conn.execute(sql, params).fetchall()
        return {r[0] for r in rows}

    def get(
        self,
        *,
        ids: Optional[list[str]] = None,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        include: Optional[list[str]] = None,
    ) -> GetResult:
        """Retrieve items by ID or filter.

        Args:
            ids: Specific IDs to fetch.
            where: Metadata filter.
            where_document: Document content filter.
            limit: Max items to return.
            offset: Number of items to skip.
            include: Fields to include.

        Returns:
            GetResult with ids, documents, metadatas, and optionally embeddings.
        """
        spec = _IncludeSpec.resolve(include, default_distances=False)

        clauses: list[str] = []
        params: list = []

        if ids is not None and not where and not where_document:
            # Pure PK lookup — skip collection filter so SQLite uses PK index
            placeholders = ",".join("?" * len(ids))
            clauses.append(f"id IN ({placeholders})")
            params.extend(ids)
        else:
            clauses.append("collection = ?")
            params.append(self._collection)
            if ids is not None:
                placeholders = ",".join("?" * len(ids))
                clauses.append(f"id IN ({placeholders})")
                params.extend(ids)

        where_sql, where_params = _build_where_clause(where)
        if where_sql:
            clauses.append(where_sql)
            params.extend(where_params)

        doc_sql, doc_params = _build_where_document_clause(where_document)
        if doc_sql:
            clauses.append(doc_sql)
            params.extend(doc_params)

        inc_cols = "id, document, metadata"
        if spec.embeddings:
            inc_cols += ", embedding"

        sql = f"SELECT {inc_cols} FROM memories WHERE {' AND '.join(clauses)} ORDER BY rowid"

        if limit is not None:
            sql += f" LIMIT {int(limit)}"
            if offset is not None:
                sql += f" OFFSET {int(offset)}"

        rows = self._conn.execute(sql, params).fetchall()

        result_ids = []
        result_docs = []
        result_metas = []
        result_embs = [] if spec.embeddings else None

        for row in rows:
            result_ids.append(row[0])
            result_docs.append(row[1] if spec.documents else "")
            result_metas.append(json.loads(row[2]) if spec.metadatas else {})
            if result_embs is not None:
                result_embs.append(_unpack_embedding(row[3], self._dim))

        return GetResult(
            ids=result_ids,
            documents=result_docs,
            metadatas=result_metas,
            embeddings=result_embs,
        )

    def delete(
        self,
        *,
        ids: Optional[list[str]] = None,
        where: Optional[dict] = None,
    ) -> None:
        """Delete items by ID or filter.

        Args:
            ids: Specific IDs to delete.
            where: Metadata filter for deletion.
        """
        clauses = ["collection = ?"]
        params: list = [self._collection]

        if ids is not None:
            placeholders = ",".join("?" * len(ids))
            clauses.append(f"id IN ({placeholders})")
            params.extend(ids)

        where_sql, where_params = _build_where_clause(where)
        if where_sql:
            clauses.append(where_sql)
            params.extend(where_params)

        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._conn.execute(
                f"DELETE FROM memories WHERE {' AND '.join(clauses)}", params
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

        self._index.invalidate()
        self._checkpoint_if_needed()

    def count(self) -> int:
        """Return the number of items in this collection."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM memories WHERE collection = ?",
            (self._collection,),
        ).fetchone()
        return row[0] if row else 0

    def health(self) -> HealthStatus:
        """Check backend health via integrity_check."""
        try:
            result = self._conn.execute("PRAGMA integrity_check").fetchone()
            if result and result[0] == "ok":
                return HealthStatus.healthy(f"sqlite: {self.count()} items")
            return HealthStatus.unhealthy(f"integrity_check: {result}")
        except sqlite3.Error as e:
            return HealthStatus.unhealthy(str(e))


# ---------------------------------------------------------------------------
# SqliteBackend
# ---------------------------------------------------------------------------


class SqliteBackend(BaseBackend):
    """SQLite + NumPy storage backend.

    Long-lived factory that manages one connection per palace. Connections
    use WAL mode for safe multi-process concurrency.

    Registered as 'sqlite' in the backend registry.
    """

    name: ClassVar[str] = "sqlite"
    spec_version: ClassVar[str] = "1.0"
    capabilities: ClassVar[frozenset[str]] = frozenset({"fts", "event_bus"})

    def __init__(self):
        self._connections: dict[str, sqlite3.Connection] = {}
        self._collections: dict[tuple[str, str], SqliteCollection] = {}
        self._closed = False
        atexit.register(self._atexit_cleanup)

    def _db_path(self, palace: PalaceRef) -> str:
        """Return the path to the SQLite database file for a palace."""
        if palace.local_path:
            return os.path.join(palace.local_path, DB_FILENAME)
        raise ValueError("SqliteBackend requires a local_path in PalaceRef")

    def _get_connection(self, palace: PalaceRef) -> sqlite3.Connection:
        """Get or create a WAL-mode connection for the given palace.

        Args:
            palace: Palace reference with local_path set.

        Returns:
            Configured SQLite connection.

        Raises:
            BackendClosedError: If the backend has been closed.
        """
        if self._closed:
            raise BackendClosedError("SqliteBackend is closed")

        key = palace.id
        conn = self._connections.get(key)
        if conn is not None:
            return conn

        db_path = self._db_path(palace)
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

        conn = sqlite3.connect(db_path, timeout=10.0, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-256000")
        conn.execute("PRAGMA mmap_size=4294967296")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA foreign_keys=ON")

        # Create schema
        conn.executescript(_SCHEMA_SQL)

        self._connections[key] = conn
        return conn

    def get_collection(
        self,
        *,
        palace: PalaceRef,
        collection_name: str,
        create: bool = False,
        options: Optional[dict] = None,
    ) -> SqliteCollection:
        """Get a collection handle for the given palace.

        Collections are cached per (palace_id, collection_name) so the
        in-memory VectorIndex survives across calls.

        Args:
            palace: Palace reference.
            collection_name: Name of the collection.
            create: Ignored for SQLite (tables are always created on connect).
            options: Optional dict with 'embed_fn' key.

        Returns:
            SqliteCollection instance.
        """
        cache_key = (palace.id, collection_name)
        cached = self._collections.get(cache_key)
        if cached is not None:
            if options and options.get("embed_fn") and cached._embed_fn is None:
                cached._embed_fn = options["embed_fn"]
            return cached

        conn = self._get_connection(palace)
        embed_fn = (options or {}).get("embed_fn")
        col = SqliteCollection(conn, collection_name, embed_fn=embed_fn)
        self._collections[cache_key] = col
        return col

    def close_palace(self, palace: PalaceRef) -> None:
        """Close the connection for a specific palace."""
        key = palace.id
        for ck in [k for k in self._collections if k[0] == key]:
            self._collections.pop(ck, None)
        conn = self._connections.pop(key, None)
        if conn is not None:
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error:
                pass
            conn.close()

    def close(self) -> None:
        """Shut down all connections."""
        self._closed = True
        self._collections.clear()
        for key in list(self._connections.keys()):
            conn = self._connections.pop(key)
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error:
                pass
            conn.close()

    def _atexit_cleanup(self):
        """Checkpoint and close on process exit."""
        try:
            self.close()
        except Exception:
            pass

    def health(self, palace: Optional[PalaceRef] = None) -> HealthStatus:
        """Check backend health.

        Args:
            palace: If provided, check the specific palace DB integrity.
        """
        if palace is not None:
            try:
                conn = self._get_connection(palace)
                result = conn.execute("PRAGMA integrity_check").fetchone()
                if result and result[0] == "ok":
                    return HealthStatus.healthy("sqlite WAL mode")
                return HealthStatus.unhealthy(f"integrity: {result}")
            except Exception as e:
                return HealthStatus.unhealthy(str(e))
        return HealthStatus.healthy("sqlite backend")

    @classmethod
    def detect(cls, path: str) -> bool:
        """Detect if a palace directory contains a SQLite mempalace DB."""
        return os.path.isfile(os.path.join(path, DB_FILENAME))

    # ------------------------------------------------------------------
    # Event bus methods
    # ------------------------------------------------------------------

    def publish_event(
        self,
        *,
        palace: PalaceRef,
        event_type: str,
        payload: dict,
        session_id: Optional[str] = None,
    ) -> int:
        """Publish an event to the event bus.

        Args:
            palace: Palace reference.
            event_type: Type of event (e.g. 'memory_added', 'decision_made').
            payload: Event data as a dict.
            session_id: Publisher's session ID. Defaults to PID.

        Returns:
            The event ID.
        """
        if session_id is None:
            session_id = str(os.getpid())

        conn = self._get_connection(palace)
        conn.execute("BEGIN IMMEDIATE")
        try:
            cursor = conn.execute(
                "INSERT INTO event_bus(session_id, event_type, payload) VALUES (?, ?, ?)",
                (session_id, event_type, json.dumps(payload)),
            )
            event_id = cursor.lastrowid
            conn.commit()
        except Exception:
            conn.rollback()
            raise

        return event_id

    def poll_events(
        self,
        *,
        palace: PalaceRef,
        session_id: str,
        since_id: int = 0,
        event_types: Optional[list[str]] = None,
    ) -> list[dict]:
        """Poll for new events not yet consumed by this session.

        Args:
            palace: Palace reference.
            session_id: The polling session's ID.
            since_id: Only return events with ID > since_id.
            event_types: Optional filter on event types.

        Returns:
            List of event dicts with id, session_id, event_type, payload, created_at.
        """
        conn = self._get_connection(palace)

        sql = (
            "SELECT id, session_id, event_type, payload, created_at "
            "FROM event_bus "
            "WHERE id > ? AND session_id != ? "
            "AND id NOT IN (SELECT event_id FROM event_consumption WHERE session_id = ?) "
        )
        params: list = [since_id, session_id, session_id]

        if event_types:
            placeholders = ",".join("?" * len(event_types))
            sql += f"AND event_type IN ({placeholders}) "
            params.extend(event_types)

        sql += "ORDER BY id"
        rows = conn.execute(sql, params).fetchall()

        if rows:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.executemany(
                    "INSERT OR IGNORE INTO event_consumption(event_id, session_id) VALUES (?, ?)",
                    [(r[0], session_id) for r in rows],
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        return [
            {
                "id": r[0],
                "session_id": r[1],
                "event_type": r[2],
                "payload": json.loads(r[3]) if r[3] else {},
                "created_at": r[4],
            }
            for r in rows
        ]

    def gc_events(self, *, palace: PalaceRef, max_age_seconds: float = 86400) -> int:
        """Garbage-collect events older than max_age_seconds.

        Args:
            palace: Palace reference.
            max_age_seconds: Max age in seconds (default 24h).

        Returns:
            Number of events deleted.
        """
        conn = self._get_connection(palace)
        cutoff = time.time() - max_age_seconds

        conn.execute("BEGIN IMMEDIATE")
        try:
            cursor = conn.execute(
                "DELETE FROM event_bus WHERE created_at < ?", (cutoff,)
            )
            deleted = cursor.rowcount
            conn.commit()
        except Exception:
            conn.rollback()
            raise

        return deleted
