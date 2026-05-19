#!/usr/bin/env python3
"""Migrate MemPalace data from ChromaDB backup to SQLite + NumPy backend.

Reads from gzipped JSON checkpoint files (ids, documents, metadatas) and
writes to the new SQLite database, embedding documents on the fly.

Features:
- Idempotent: INSERT OR IGNORE, safe to re-run
- Resumable: tracks progress in migration_state table
- Paginated: processes in configurable batch sizes
- Verified: count check + random sample embedding comparison
- Non-destructive: never modifies source files

Usage:
    python3 -m mempalace.migrate_to_sqlite [--backup-dir DIR] [--target-dir DIR] [--batch-size N]

Args:
    --backup-dir: Path to backup with *.json.gz files (default: ~/.mempalace/palace.backup-20260510-102907)
    --target-dir: Path to write mempalace.db (default: ~/.mempalace)
    --batch-size: Items per batch for embedding + insert (default: 256)
"""

import argparse
import gzip
import json
import logging
import os
import sqlite3
import struct
import sys
import time

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger("migrate")

DEFAULT_BACKUP = os.path.expanduser(
    "~/.mempalace/palace.backup-20260510-102907"
)
DEFAULT_TARGET = None  # resolved from MempalaceConfig at runtime
EMBEDDING_DIM = 384
DB_FILENAME = "mempalace.db"


def _load_checkpoint(backup_dir: str, collection: str) -> dict:
    """Load a gzipped JSON checkpoint file.

    Args:
        backup_dir: Directory containing the .json.gz files.
        collection: Collection name (e.g. 'mempalace_drawers').

    Returns:
        Dict with 'ids', 'documents', 'metadatas' lists.
    """
    path = os.path.join(backup_dir, f"{collection}.json.gz")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    logger.info("Loading %s ...", path)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(
        "  %s: %d items loaded", collection, len(data["ids"])
    )
    return data


def _init_target_db(target_dir: str) -> sqlite3.Connection:
    """Initialize the target SQLite database with schema.

    Args:
        target_dir: Directory to create mempalace.db in.

    Returns:
        Configured SQLite connection.
    """
    from mempalace.backends.sqlite_backend import _SCHEMA_SQL

    db_path = os.path.join(target_dir, DB_FILENAME)
    os.makedirs(target_dir, exist_ok=True)

    conn = sqlite3.connect(db_path, timeout=30.0, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")
    conn.execute("PRAGMA mmap_size=268435456")
    conn.execute("PRAGMA temp_store=MEMORY")

    conn.executescript(_SCHEMA_SQL)

    # Migration state tracking table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS migration_state (
            collection TEXT PRIMARY KEY,
            offset INTEGER NOT NULL DEFAULT 0,
            total INTEGER NOT NULL DEFAULT 0,
            completed_at REAL
        )
    """)

    return conn


def _get_resume_offset(conn: sqlite3.Connection, collection: str) -> int:
    """Get the offset to resume migration from.

    Args:
        conn: SQLite connection.
        collection: Collection name.

    Returns:
        Offset to resume from (0 if starting fresh).
    """
    row = conn.execute(
        "SELECT offset FROM migration_state WHERE collection = ? AND completed_at IS NULL",
        (collection,),
    ).fetchone()
    return row[0] if row else 0


def _update_progress(conn: sqlite3.Connection, collection: str, offset: int, total: int):
    """Update migration progress."""
    conn.execute(
        "INSERT OR REPLACE INTO migration_state(collection, offset, total) VALUES (?, ?, ?)",
        (collection, offset, total),
    )


def _mark_complete(conn: sqlite3.Connection, collection: str):
    """Mark migration as complete."""
    conn.execute(
        "UPDATE migration_state SET completed_at = ? WHERE collection = ?",
        (time.time(), collection),
    )


def _migrate_collection(
    conn: sqlite3.Connection,
    collection: str,
    data: dict,
    embed_fn,
    batch_size: int = 256,
):
    """Migrate a single collection from checkpoint data to SQLite.

    Args:
        conn: Target SQLite connection.
        collection: Collection name.
        data: Dict with 'ids', 'documents', 'metadatas' lists.
        embed_fn: Embedding function.
        batch_size: Items per batch.
    """
    ids = data["ids"]
    docs = data["documents"]
    metas = data["metadatas"]
    total = len(ids)

    resume_offset = _get_resume_offset(conn, collection)
    if resume_offset > 0:
        logger.info("  Resuming %s from offset %d / %d", collection, resume_offset, total)

    t0 = time.time()
    processed = resume_offset

    for batch_start in range(resume_offset, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        b_ids = ids[batch_start:batch_end]
        b_docs = docs[batch_start:batch_end]
        b_metas = metas[batch_start:batch_end]

        # Embed the batch
        t_embed = time.time()
        b_embeddings = embed_fn(b_docs)
        embed_time = time.time() - t_embed

        # Prepare rows
        rows = []
        for i, id_ in enumerate(b_ids):
            meta = b_metas[i] if b_metas[i] is not None else {}
            # Handle both "wing" and "hall" metadata keys
            wing = meta.get("wing") or meta.get("hall")
            room = meta.get("room")
            emb_bytes = struct.pack(f"{len(b_embeddings[i])}f", *b_embeddings[i])
            rows.append((
                id_,
                collection,
                wing,
                room,
                b_docs[i],
                json.dumps(meta),
                emb_bytes,
            ))

        # Insert batch
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.executemany(
                "INSERT OR IGNORE INTO memories"
                "(id, collection, wing, room, document, metadata, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

        processed = batch_end
        _update_progress(conn, collection, processed, total)

        # Progress reporting
        elapsed = time.time() - t0
        rate = (processed - resume_offset) / max(elapsed, 0.01)
        remaining = total - processed
        eta = remaining / max(rate, 0.01) if rate > 0 else 0

        if processed % (batch_size * 10) < batch_size or processed == total:
            logger.info(
                "  %s: %d/%d (%.0f/s, embed %.1fs/batch, ETA %.0fm)",
                collection,
                processed,
                total,
                rate,
                embed_time,
                eta / 60,
            )

    _mark_complete(conn, collection)
    total_time = time.time() - t0
    logger.info(
        "  %s: DONE — %d items in %.0fs (%.1f min)",
        collection,
        total,
        total_time,
        total_time / 60,
    )


def _verify(conn: sqlite3.Connection, data_by_collection: dict, embed_fn):
    """Verify migration fidelity.

    Args:
        conn: SQLite connection.
        data_by_collection: Dict mapping collection name to checkpoint data.
        embed_fn: Embedding function for sample verification.

    Raises:
        AssertionError: If verification fails.
    """
    logger.info("\nVerification:")

    for collection, data in data_by_collection.items():
        expected = len(data["ids"])
        actual = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE collection = ?",
            (collection,),
        ).fetchone()[0]

        logger.info("  %s: expected=%d, actual=%d", collection, expected, actual)
        assert actual >= expected, (
            f"{collection}: count {actual} < expected {expected}"
        )

        # Sample document fidelity
        rng = np.random.RandomState(42)
        sample_indices = rng.choice(expected, size=min(100, expected), replace=False)

        mismatches = 0
        for idx in sample_indices:
            expected_id = data["ids"][idx]
            expected_doc = data["documents"][idx]
            expected_meta = data["metadatas"][idx]

            row = conn.execute(
                "SELECT document, metadata FROM memories WHERE id = ?",
                (expected_id,),
            ).fetchone()

            if row is None:
                mismatches += 1
                continue

            if row[0] != expected_doc:
                mismatches += 1
                logger.warning("  Doc mismatch for %s", expected_id)
                continue

            stored_meta = json.loads(row[1])
            if stored_meta != expected_meta:
                mismatches += 1
                logger.warning("  Meta mismatch for %s", expected_id)

        logger.info(
            "  %s: %d/%d samples verified (%.1f%% match)",
            collection,
            len(sample_indices) - mismatches,
            len(sample_indices),
            (len(sample_indices) - mismatches) / len(sample_indices) * 100,
        )
        assert mismatches == 0, (
            f"{collection}: {mismatches} sample mismatches"
        )

    # Integrity check
    integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
    assert integrity == "ok", f"Integrity check failed: {integrity}"
    logger.info("  Integrity check: %s", integrity)
    logger.info("  Verification PASSED")


def main():
    parser = argparse.ArgumentParser(description="Migrate MemPalace to SQLite")
    parser.add_argument(
        "--backup-dir",
        default=DEFAULT_BACKUP,
        help="Path to backup directory with .json.gz files",
    )
    parser.add_argument(
        "--target-dir",
        default=None,
        help="Path to write mempalace.db (default: palace_path from config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Items per embedding batch (default 256)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only run verification, skip migration",
    )
    args = parser.parse_args()

    if args.target_dir is None:
        from mempalace.config import MempalaceConfig
        args.target_dir = MempalaceConfig().palace_path

    logger.info("=" * 60)
    logger.info("MemPalace Migration: ChromaDB → SQLite")
    logger.info("=" * 60)
    logger.info("Backup: %s", args.backup_dir)
    logger.info("Target: %s/%s", args.target_dir, DB_FILENAME)
    logger.info("Batch:  %d", args.batch_size)

    # Load checkpoints
    data_by_collection = {}
    for name in ["mempalace_drawers", "mempalace_closets"]:
        data_by_collection[name] = _load_checkpoint(args.backup_dir, name)

    # Init embedding function
    logger.info("\nInitializing embedding function...")
    from mempalace.embedding import get_embedding_function

    ef = get_embedding_function()
    logger.info("  Ready (embedding dim: %d)", EMBEDDING_DIM)

    # Init target DB
    conn = _init_target_db(args.target_dir)

    if not args.verify_only:
        # Migrate each collection
        for name, data in data_by_collection.items():
            logger.info("\nMigrating %s (%d items)...", name, len(data["ids"]))
            _migrate_collection(conn, name, data, ef, batch_size=args.batch_size)

    # Verify
    _verify(conn, data_by_collection, ef)

    # Final stats
    db_path = os.path.join(args.target_dir, DB_FILENAME)
    db_size = os.path.getsize(db_path)
    logger.info("\n" + "=" * 60)
    logger.info("MIGRATION COMPLETE")
    logger.info("Database: %s (%.1f MB)", db_path, db_size / 1024 / 1024)
    logger.info("=" * 60)

    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        logger.error("VERIFICATION FAILED: %s", e)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\nInterrupted — progress saved. Re-run to resume.")
        sys.exit(130)
