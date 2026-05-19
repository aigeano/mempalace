"""Tier 1 unit tests for the SQLite + NumPy storage backend.

Tests are written before the implementation (TDD). All 25 tests must pass
before Phase 1 is accepted. Each test targets a single behavior documented
in plan 005-mempalace-sqlite-rebuild.md.
"""

import json
import os
import struct
import tempfile

import numpy as np
import pytest

from mempalace.backends.base import (
    BaseBackend,
    BaseCollection,
    GetResult,
    PalaceRef,
    QueryResult,
    UnsupportedFilterError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_embedding(seed: int = 0, dim: int = 384) -> list[float]:
    """Generate a deterministic embedding vector from a seed."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    return vec.tolist()


@pytest.fixture()
def backend():
    """Create a fresh SqliteBackend pointed at a temp directory."""
    from mempalace.backends.sqlite_backend import SqliteBackend

    with tempfile.TemporaryDirectory(prefix="mp_sqlite_test_") as tmp:
        be = SqliteBackend()
        palace = PalaceRef(id="test", local_path=tmp)
        yield be, palace, tmp
        be.close()


@pytest.fixture()
def collection(backend):
    """Return a drawers collection ready for use."""
    be, palace, tmp = backend
    return be.get_collection(
        palace=palace, collection_name="mempalace_drawers", create=True
    )


@pytest.fixture()
def populated_collection(collection):
    """Collection pre-loaded with 100 items across wings/rooms."""
    docs = []
    ids = []
    metas = []
    embeddings = []
    wings = ["technical", "personal", "creative", "memory"]
    rooms = ["room_a", "room_b", "room_c"]
    for i in range(100):
        ids.append(f"item_{i:03d}")
        docs.append(f"Document number {i} about {wings[i % len(wings)]} topic in {rooms[i % len(rooms)]}")
        metas.append({
            "wing": wings[i % len(wings)],
            "room": rooms[i % len(rooms)],
            "index": i,
            "score": float(i) / 100.0,
            "active": i % 2 == 0,
        })
        embeddings.append(_make_embedding(seed=i))

    collection.add(documents=docs, ids=ids, metadatas=metas, embeddings=embeddings)
    return collection


# ---------------------------------------------------------------------------
# Test 1-2: Basic CRUD
# ---------------------------------------------------------------------------


class TestBasicCRUD:
    def test_add_and_get(self, collection):
        """Round-trip: add items, get by ID, verify document + metadata identical."""
        docs = [f"doc_{i}" for i in range(100)]
        ids = [f"id_{i}" for i in range(100)]
        metas = [{"wing": "tech", "index": i} for i in range(100)]
        embs = [_make_embedding(i) for i in range(100)]

        collection.add(documents=docs, ids=ids, metadatas=metas, embeddings=embs)

        result = collection.get(ids=["id_0", "id_50", "id_99"])
        assert isinstance(result, GetResult)
        assert set(result.ids) == {"id_0", "id_50", "id_99"}
        assert len(result.documents) == 3
        assert len(result.metadatas) == 3

        # Verify content round-trips exactly
        idx = result.ids.index("id_50")
        assert result.documents[idx] == "doc_50"
        assert result.metadatas[idx]["wing"] == "tech"
        assert result.metadatas[idx]["index"] == 50

    def test_upsert_overwrites(self, collection):
        """Insert item, upsert with new doc, get returns new doc."""
        collection.add(
            documents=["original"],
            ids=["u1"],
            metadatas=[{"wing": "tech"}],
            embeddings=[_make_embedding(0)],
        )
        assert collection.get(ids=["u1"]).documents[0] == "original"

        collection.upsert(
            documents=["updated"],
            ids=["u1"],
            metadatas=[{"wing": "personal"}],
            embeddings=[_make_embedding(1)],
        )
        result = collection.get(ids=["u1"])
        assert result.documents[0] == "updated"
        assert result.metadatas[0]["wing"] == "personal"


# ---------------------------------------------------------------------------
# Test 3-4: Deletion
# ---------------------------------------------------------------------------


class TestDeletion:
    def test_delete_by_id(self, collection):
        """Add 10, delete 3, count = 7, get returns empty for deleted IDs."""
        ids = [f"d_{i}" for i in range(10)]
        collection.add(
            documents=[f"doc_{i}" for i in range(10)],
            ids=ids,
            metadatas=[{"wing": "tech"} for _ in range(10)],
            embeddings=[_make_embedding(i) for i in range(10)],
        )
        assert collection.count() == 10

        collection.delete(ids=["d_2", "d_5", "d_8"])
        assert collection.count() == 7

        result = collection.get(ids=["d_2", "d_5", "d_8"])
        assert result.ids == []

    def test_delete_by_where(self, populated_collection):
        """Delete by filter, verify only matching items removed."""
        col = populated_collection
        count_before = col.count()

        # Delete all items in wing "technical" (25 out of 100)
        col.delete(where={"wing": {"$eq": "technical"}})

        count_after = col.count()
        assert count_after == count_before - 25

        # Verify no technical items remain
        result = col.get(where={"wing": {"$eq": "technical"}})
        assert result.ids == []


# ---------------------------------------------------------------------------
# Test 5-11: Filter operators
# ---------------------------------------------------------------------------


class TestFilterOperators:
    def test_filter_eq(self, populated_collection):
        """$eq returns only matching items."""
        result = populated_collection.get(where={"wing": {"$eq": "technical"}})
        assert len(result.ids) == 25
        for meta in result.metadatas:
            assert meta["wing"] == "technical"

    def test_filter_ne(self, populated_collection):
        """$ne returns everything EXCEPT the matched value."""
        result = populated_collection.get(where={"wing": {"$ne": "technical"}})
        assert len(result.ids) == 75
        for meta in result.metadatas:
            assert meta["wing"] != "technical"

    def test_filter_in(self, populated_collection):
        """$in returns items matching any value in the list."""
        result = populated_collection.get(
            where={"wing": {"$in": ["technical", "creative"]}}
        )
        assert len(result.ids) == 50
        for meta in result.metadatas:
            assert meta["wing"] in ("technical", "creative")

    def test_filter_nin(self, populated_collection):
        """$nin excludes items matching any value in the list."""
        result = populated_collection.get(
            where={"wing": {"$nin": ["technical", "creative"]}}
        )
        assert len(result.ids) == 50
        for meta in result.metadatas:
            assert meta["wing"] not in ("technical", "creative")

    def test_filter_and(self, populated_collection):
        """$and with two conditions — both must match."""
        result = populated_collection.get(
            where={
                "$and": [
                    {"wing": {"$eq": "technical"}},
                    {"room": {"$eq": "room_a"}},
                ]
            }
        )
        for meta in result.metadatas:
            assert meta["wing"] == "technical"
            assert meta["room"] == "room_a"
        assert len(result.ids) > 0

    def test_filter_or(self, populated_collection):
        """$or — either condition matches."""
        result = populated_collection.get(
            where={
                "$or": [
                    {"wing": {"$eq": "technical"}},
                    {"wing": {"$eq": "creative"}},
                ]
            }
        )
        assert len(result.ids) == 50
        for meta in result.metadatas:
            assert meta["wing"] in ("technical", "creative")

    def test_filter_gt_lt(self, populated_collection):
        """Range operators on numeric metadata fields."""
        result = populated_collection.get(
            where={
                "$and": [
                    {"index": {"$gte": 10}},
                    {"index": {"$lt": 20}},
                ]
            }
        )
        assert len(result.ids) == 10
        for meta in result.metadatas:
            assert 10 <= meta["index"] < 20


# ---------------------------------------------------------------------------
# Test 12: where_document ($contains via FTS)
# ---------------------------------------------------------------------------


class TestWhereDocument:
    def test_where_document_contains(self, populated_collection):
        """FTS $contains finds documents with specific text."""
        result = populated_collection.get(
            where_document={"$contains": "technical topic"}
        )
        assert len(result.ids) > 0
        for doc in result.documents:
            assert "technical" in doc.lower()


# ---------------------------------------------------------------------------
# Test 13-16: Vector search
# ---------------------------------------------------------------------------


class TestVectorSearch:
    def test_vector_search_correctness(self, collection):
        """Verify top-10 matches manual numpy cosine computation exactly."""
        n = 1000
        dim = 384
        rng = np.random.RandomState(42)

        # Insert 1000 known vectors
        embeddings = [rng.randn(dim).astype(np.float32).tolist() for _ in range(n)]
        collection.add(
            documents=[f"doc_{i}" for i in range(n)],
            ids=[f"v_{i}" for i in range(n)],
            metadatas=[{"wing": "test"} for _ in range(n)],
            embeddings=embeddings,
        )

        # Query with a known vector
        query_vec = rng.randn(dim).astype(np.float32).tolist()
        result = collection.query(
            query_embeddings=[query_vec], n_results=10
        )

        # Compute expected results manually with numpy
        db = np.array(embeddings, dtype=np.float32)
        q = np.array(query_vec, dtype=np.float32)
        db_norm = db / np.linalg.norm(db, axis=1, keepdims=True)
        q_norm = q / np.linalg.norm(q)
        similarities = db_norm @ q_norm
        expected_indices = np.argsort(similarities)[::-1][:10]
        expected_ids = [f"v_{i}" for i in expected_indices]

        assert result.ids[0] == expected_ids

    def test_vector_search_distance_range(self, populated_collection):
        """All returned distances must be in [0.0, 2.0]."""
        result = populated_collection.query(
            query_embeddings=[_make_embedding(seed=999)], n_results=10
        )
        assert len(result.distances[0]) > 0
        for d in result.distances[0]:
            assert 0.0 <= d <= 2.0, f"Distance {d} out of range"

    def test_vector_search_ordering(self, populated_collection):
        """Distances must be monotonically non-decreasing."""
        result = populated_collection.query(
            query_embeddings=[_make_embedding(seed=999)], n_results=20
        )
        distances = result.distances[0]
        for i in range(len(distances) - 1):
            assert distances[i] <= distances[i + 1] + 1e-6, (
                f"Distance not sorted: {distances[i]} > {distances[i + 1]}"
            )

    def test_empty_collection_query(self, collection):
        """Query on empty collection returns empty QueryResult, no crash."""
        result = collection.query(
            query_embeddings=[_make_embedding(0)], n_results=10
        )
        assert isinstance(result, QueryResult)
        assert result.ids == [[]]
        assert result.distances == [[]]


# ---------------------------------------------------------------------------
# Test 17-18: Include spec and pagination
# ---------------------------------------------------------------------------


class TestIncludeAndPagination:
    def test_include_spec(self, populated_collection):
        """include parameter controls which fields are returned."""
        # Only documents
        result = populated_collection.get(
            ids=["item_000"], include=["documents"]
        )
        assert len(result.documents) == 1
        assert result.documents[0] != ""

        # With embeddings
        result = populated_collection.get(
            ids=["item_000"], include=["documents", "embeddings"]
        )
        assert result.embeddings is not None
        assert len(result.embeddings) == 1
        assert len(result.embeddings[0]) == 384

    def test_get_pagination(self, populated_collection):
        """limit/offset returns correct slice."""
        all_items = populated_collection.get(limit=100)
        assert len(all_items.ids) == 100

        page = populated_collection.get(limit=10, offset=5)
        assert len(page.ids) == 10


# ---------------------------------------------------------------------------
# Test 19-20: Metadata and FTS sync
# ---------------------------------------------------------------------------


class TestMetadataAndFTS:
    def test_metadata_json_types(self, collection):
        """Metadata with str, int, float, bool, null all survive round-trip."""
        meta = {
            "name": "test",
            "count": 42,
            "ratio": 3.14,
            "active": True,
            "inactive": False,
            "empty": None,
        }
        collection.add(
            documents=["test doc"],
            ids=["json_types"],
            metadatas=[meta],
            embeddings=[_make_embedding(0)],
        )
        result = collection.get(ids=["json_types"])
        got = result.metadatas[0]
        assert got["name"] == "test"
        assert got["count"] == 42
        assert abs(got["ratio"] - 3.14) < 0.01
        assert got["active"] is True
        assert got["inactive"] is False
        assert got["empty"] is None

    def test_fts_trigger_sync(self, collection):
        """FTS index stays in sync across add, upsert (update), and delete."""
        # Add
        collection.add(
            documents=["unique_banana_phrase in document"],
            ids=["fts_1"],
            metadatas=[{"wing": "test"}],
            embeddings=[_make_embedding(0)],
        )
        result = collection.get(where_document={"$contains": "unique_banana_phrase"})
        assert "fts_1" in result.ids

        # Update via upsert
        collection.upsert(
            documents=["replaced with unique_mango_phrase instead"],
            ids=["fts_1"],
            metadatas=[{"wing": "test"}],
            embeddings=[_make_embedding(1)],
        )
        old_result = collection.get(where_document={"$contains": "unique_banana_phrase"})
        assert "fts_1" not in old_result.ids

        new_result = collection.get(where_document={"$contains": "unique_mango_phrase"})
        assert "fts_1" in new_result.ids

        # Delete
        collection.delete(ids=["fts_1"])
        after_delete = collection.get(where_document={"$contains": "unique_mango_phrase"})
        assert "fts_1" not in after_delete.ids


# ---------------------------------------------------------------------------
# Test 21-25: Event bus
# ---------------------------------------------------------------------------


class TestEventBus:
    def test_event_bus_publish(self, backend):
        """Events are stored in the database."""
        from mempalace.backends.sqlite_backend import SqliteBackend

        be, palace, tmp = backend
        col = be.get_collection(
            palace=palace, collection_name="mempalace_drawers", create=True
        )
        be.publish_event(
            palace=palace,
            event_type="memory_added",
            payload={"id": "test_1", "wing": "tech"},
            session_id="session_a",
        )

        events = be.poll_events(palace=palace, session_id="session_b")
        assert len(events) >= 1
        assert events[0]["event_type"] == "memory_added"
        assert events[0]["payload"]["id"] == "test_1"

    def test_event_bus_consume(self, backend):
        """Consumed events are not returned on subsequent polls."""
        be, palace, tmp = backend
        be.get_collection(palace=palace, collection_name="mempalace_drawers", create=True)

        be.publish_event(
            palace=palace,
            event_type="decision_made",
            payload={"decision": "use_sqlite"},
            session_id="session_a",
        )

        # First poll — should see the event
        events1 = be.poll_events(palace=palace, session_id="session_b")
        assert len(events1) == 1

        # Second poll — should NOT see it again
        events2 = be.poll_events(palace=palace, session_id="session_b")
        assert len(events2) == 0

    def test_event_bus_gc(self, backend):
        """Old events are pruned by garbage collection."""
        import sqlite3
        import time

        be, palace, tmp = backend
        be.get_collection(palace=palace, collection_name="mempalace_drawers", create=True)

        # Insert an event with an old timestamp directly
        db_path = os.path.join(tmp, "mempalace.db")
        conn = sqlite3.connect(db_path)
        old_time = time.time() - 90000  # 25 hours ago
        conn.execute(
            "INSERT INTO event_bus(session_id, event_type, payload, created_at) VALUES (?, ?, ?, ?)",
            ("old_session", "old_event", "{}", old_time),
        )
        conn.commit()
        conn.close()

        # Publish a fresh event to trigger GC
        be.publish_event(
            palace=palace,
            event_type="fresh_event",
            payload={},
            session_id="session_a",
        )
        be.gc_events(palace=palace)

        # Poll — should only see the fresh event
        events = be.poll_events(palace=palace, session_id="session_c")
        for e in events:
            assert e["event_type"] != "old_event"

    def test_event_bus_cross_session(self, backend):
        """Session A doesn't see own events; Session B does."""
        be, palace, tmp = backend
        be.get_collection(palace=palace, collection_name="mempalace_drawers", create=True)

        be.publish_event(
            palace=palace,
            event_type="task_claimed",
            payload={"task": "build_feature"},
            session_id="session_a",
        )

        # Session A should NOT see its own event
        events_a = be.poll_events(palace=palace, session_id="session_a")
        assert len(events_a) == 0

        # Session B SHOULD see it
        events_b = be.poll_events(palace=palace, session_id="session_b")
        assert len(events_b) == 1
        assert events_b[0]["session_id"] == "session_a"

    def test_schema_version(self, backend):
        """Fresh DB has version 1 in schema_version table."""
        import sqlite3

        be, palace, tmp = backend
        be.get_collection(palace=palace, collection_name="mempalace_drawers", create=True)

        db_path = os.path.join(tmp, "mempalace.db")
        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        conn.close()
        assert row is not None
        assert row[0] == 1


# ---------------------------------------------------------------------------
# Test: Count
# ---------------------------------------------------------------------------


class TestCount:
    def test_count(self, populated_collection):
        """Count returns correct number of items."""
        assert populated_collection.count() == 100

    def test_estimated_count(self, populated_collection):
        """Estimated count matches actual count."""
        assert populated_collection.estimated_count() == 100
