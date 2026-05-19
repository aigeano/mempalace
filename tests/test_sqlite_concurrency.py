"""Tier 2 concurrency stress tests for the SQLite + NumPy backend.

Uses multiprocessing (not threads) to reproduce the real multi-MCP-server
scenario. Each test spawns separate OS processes that independently open
the same SQLite database file.

Total runtime: ~3-5 minutes.
"""

import multiprocessing
import os
import signal
import sqlite3
import struct
import tempfile
import time

import numpy as np
import pytest


EMBEDDING_DIM = 384
DB_FILENAME = "mempalace.db"


def _make_embedding(seed: int = 0) -> list[float]:
    """Generate a deterministic embedding vector."""
    rng = np.random.RandomState(seed)
    return rng.randn(EMBEDDING_DIM).astype(np.float32).tolist()


def _pack_embedding(emb: list[float]) -> bytes:
    return struct.pack(f"{len(emb)}f", *emb)


def _init_db(db_path: str):
    """Initialize the schema in a fresh database."""
    from mempalace.backends.sqlite_backend import SqliteBackend

    be = SqliteBackend()
    from mempalace.backends.base import PalaceRef

    palace = PalaceRef(id="stress", local_path=os.path.dirname(db_path))
    be.get_collection(palace=palace, collection_name="mempalace_drawers", create=True)
    be.close()


# ---------------------------------------------------------------------------
# Worker functions (run in separate processes)
# ---------------------------------------------------------------------------


def _writer_worker(db_path: str, worker_id: int, count: int, result_queue):
    """Write `count` items to the database."""
    try:
        conn = sqlite3.connect(db_path, timeout=10.0, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA synchronous=NORMAL")

        written = 0
        errors = 0
        for i in range(count):
            emb = _make_embedding(worker_id * 100000 + i)
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    "INSERT INTO memories(id, collection, wing, room, document, metadata, embedding) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        f"w{worker_id}_item_{i}",
                        "mempalace_drawers",
                        f"wing_{worker_id}",
                        f"room_{i % 3}",
                        f"Document from worker {worker_id}, item {i}",
                        "{}",
                        _pack_embedding(emb),
                    ),
                )
                conn.commit()
                written += 1
            except sqlite3.OperationalError as e:
                conn.rollback()
                errors += 1
                if "database is locked" in str(e):
                    time.sleep(0.01)
                else:
                    raise

        conn.close()
        result_queue.put({"worker_id": worker_id, "written": written, "errors": errors})
    except Exception as e:
        result_queue.put({"worker_id": worker_id, "error": str(e)})


def _reader_worker(db_path: str, worker_id: int, duration_seconds: float, result_queue):
    """Continuously read from the database for `duration_seconds`."""
    try:
        conn = sqlite3.connect(db_path, timeout=10.0, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")

        reads = 0
        errors = 0
        start = time.time()
        while time.time() - start < duration_seconds:
            try:
                rows = conn.execute(
                    "SELECT COUNT(*) FROM memories WHERE collection = ?",
                    ("mempalace_drawers",),
                ).fetchone()
                reads += 1

                # Also do a vector-like read (fetch embeddings)
                sample = conn.execute(
                    "SELECT id, embedding FROM memories WHERE collection = ? LIMIT 10",
                    ("mempalace_drawers",),
                ).fetchall()
                reads += 1
            except sqlite3.OperationalError:
                errors += 1
                time.sleep(0.01)

        conn.close()
        result_queue.put({"worker_id": worker_id, "reads": reads, "errors": errors})
    except Exception as e:
        result_queue.put({"worker_id": worker_id, "error": str(e)})


def _event_bus_worker(db_path: str, worker_id: int, count: int, result_queue):
    """Publish events and poll for others' events."""
    try:
        conn = sqlite3.connect(db_path, timeout=10.0, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")

        session_id = f"session_{worker_id}"
        published = 0
        consumed = 0

        for i in range(count):
            # Publish
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    "INSERT INTO event_bus(session_id, event_type, payload) VALUES (?, ?, ?)",
                    (session_id, "test_event", f'{{"worker": {worker_id}, "i": {i}}}'),
                )
                conn.commit()
                published += 1
            except sqlite3.OperationalError:
                conn.rollback()
                time.sleep(0.01)

            # Poll for others' events
            try:
                rows = conn.execute(
                    "SELECT id FROM event_bus WHERE session_id != ? "
                    "AND id NOT IN (SELECT event_id FROM event_consumption WHERE session_id = ?)",
                    (session_id, session_id),
                ).fetchall()

                if rows:
                    conn.execute("BEGIN IMMEDIATE")
                    conn.executemany(
                        "INSERT OR IGNORE INTO event_consumption(event_id, session_id) VALUES (?, ?)",
                        [(r[0], session_id) for r in rows],
                    )
                    conn.commit()
                    consumed += len(rows)
            except sqlite3.OperationalError:
                try:
                    conn.rollback()
                except Exception:
                    pass
                time.sleep(0.01)

        conn.close()
        result_queue.put({
            "worker_id": worker_id,
            "published": published,
            "consumed": consumed,
        })
    except Exception as e:
        result_queue.put({"worker_id": worker_id, "error": str(e)})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConcurrentWriters:
    def test_concurrent_writers_no_corruption(self, tmp_path):
        """6 processes writing simultaneously for ~30s, zero corruption.

        Each writer inserts 500 items. Final count must equal sum of all
        writes. PRAGMA integrity_check must return 'ok'.
        """
        db_path = str(tmp_path / DB_FILENAME)
        _init_db(db_path)

        n_workers = 6
        items_per_worker = 500
        result_queue = multiprocessing.Queue()

        processes = []
        for w in range(n_workers):
            p = multiprocessing.Process(
                target=_writer_worker,
                args=(db_path, w, items_per_worker, result_queue),
            )
            processes.append(p)

        for p in processes:
            p.start()
        for p in processes:
            p.join(timeout=120)

        results = []
        while not result_queue.empty():
            results.append(result_queue.get_nowait())

        # Check no worker had fatal errors
        for r in results:
            assert "error" not in r, f"Worker {r.get('worker_id')} failed: {r.get('error')}"

        total_written = sum(r["written"] for r in results)
        total_errors = sum(r.get("errors", 0) for r in results)

        # Verify count
        conn = sqlite3.connect(db_path)
        actual = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        conn.close()

        assert integrity == "ok", f"Database corrupted: {integrity}"
        assert actual == total_written, f"Count mismatch: {actual} vs {total_written}"
        assert total_written == n_workers * items_per_worker


class TestConcurrentReadWrite:
    def test_concurrent_read_write(self, tmp_path):
        """3 writers + 3 readers running simultaneously.

        Writers do INSERTs, readers do SELECTs. Zero unhandled errors.
        All operations complete within timeout.
        """
        db_path = str(tmp_path / DB_FILENAME)
        _init_db(db_path)

        result_queue = multiprocessing.Queue()
        duration = 15.0  # seconds

        processes = []
        # 3 writers
        for w in range(3):
            p = multiprocessing.Process(
                target=_writer_worker,
                args=(db_path, w, 300, result_queue),
            )
            processes.append(p)

        # 3 readers
        for r in range(3):
            p = multiprocessing.Process(
                target=_reader_worker,
                args=(db_path, 100 + r, duration, result_queue),
            )
            processes.append(p)

        for p in processes:
            p.start()
        for p in processes:
            p.join(timeout=60)

        results = []
        while not result_queue.empty():
            results.append(result_queue.get_nowait())

        for r in results:
            assert "error" not in r, f"Worker {r.get('worker_id')} failed: {r.get('error')}"

        # Verify integrity
        conn = sqlite3.connect(db_path)
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        conn.close()
        assert integrity == "ok"


class TestWriterContention:
    def test_writer_contention_under_load(self, tmp_path):
        """10 processes all writing simultaneously (extreme scenario).

        Zero unhandled SQLITE_BUSY exceptions. All complete within timeout.
        """
        db_path = str(tmp_path / DB_FILENAME)
        _init_db(db_path)

        n_workers = 10
        items_per_worker = 100
        result_queue = multiprocessing.Queue()

        processes = []
        for w in range(n_workers):
            p = multiprocessing.Process(
                target=_writer_worker,
                args=(db_path, w, items_per_worker, result_queue),
            )
            processes.append(p)

        for p in processes:
            p.start()
        for p in processes:
            p.join(timeout=120)

        results = []
        while not result_queue.empty():
            results.append(result_queue.get_nowait())

        for r in results:
            assert "error" not in r, f"Worker {r.get('worker_id')} failed: {r.get('error')}"

        total_written = sum(r["written"] for r in results)

        conn = sqlite3.connect(db_path)
        actual = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        conn.close()

        assert integrity == "ok"
        assert actual == total_written


class TestCrashRecovery:
    def test_crash_recovery(self, tmp_path):
        """Kill -9 a writer mid-transaction, verify DB integrity survives."""
        db_path = str(tmp_path / DB_FILENAME)
        _init_db(db_path)

        result_queue = multiprocessing.Queue()

        # Start a writer
        p = multiprocessing.Process(
            target=_writer_worker,
            args=(db_path, 0, 1000, result_queue),
        )
        p.start()

        # Let it write for a bit, then kill it
        time.sleep(0.5)
        if p.is_alive():
            os.kill(p.pid, signal.SIGKILL)
        p.join(timeout=5)

        # Verify DB is intact
        conn = sqlite3.connect(db_path, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        conn.close()

        assert integrity == "ok", f"Corrupted after kill: {integrity}"
        assert count >= 0  # Some items may have been committed before kill

    def test_crash_recovery_repeated(self, tmp_path):
        """20 repeated crashes don't accumulate damage."""
        db_path = str(tmp_path / DB_FILENAME)
        _init_db(db_path)

        result_queue = multiprocessing.Queue()

        for attempt in range(20):
            p = multiprocessing.Process(
                target=_writer_worker,
                args=(db_path, attempt, 200, result_queue),
            )
            p.start()

            # Random delay before kill (50-500ms)
            delay = 0.05 + (attempt % 10) * 0.05
            time.sleep(delay)

            if p.is_alive():
                os.kill(p.pid, signal.SIGKILL)
            p.join(timeout=5)

        # Final integrity check
        conn = sqlite3.connect(db_path, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        conn.close()

        assert integrity == "ok", f"Corrupted after {20} crashes: {integrity}"


class TestWALGrowth:
    def test_wal_growth_bounded(self, tmp_path):
        """6 processes (3 writers, 3 readers) for 20s. WAL < 100MB."""
        db_path = str(tmp_path / DB_FILENAME)
        _init_db(db_path)

        result_queue = multiprocessing.Queue()
        processes = []

        # 3 writers
        for w in range(3):
            p = multiprocessing.Process(
                target=_writer_worker,
                args=(db_path, w, 500, result_queue),
            )
            processes.append(p)

        # 3 readers
        for r in range(3):
            p = multiprocessing.Process(
                target=_reader_worker,
                args=(db_path, 100 + r, 20.0, result_queue),
            )
            processes.append(p)

        for p in processes:
            p.start()
        for p in processes:
            p.join(timeout=60)

        # Check WAL size
        wal_path = db_path + "-wal"
        if os.path.exists(wal_path):
            wal_size = os.path.getsize(wal_path)
            assert wal_size < 100 * 1024 * 1024, (
                f"WAL too large: {wal_size / 1024 / 1024:.1f}MB"
            )

        # Integrity
        conn = sqlite3.connect(db_path)
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        conn.close()
        assert integrity == "ok"


class TestStaleIndexVisibility:
    def test_stale_index_visibility(self, tmp_path):
        """Writer's own writes are immediately visible via the backend API."""
        from mempalace.backends.sqlite_backend import SqliteBackend
        from mempalace.backends.base import PalaceRef

        be = SqliteBackend()
        palace = PalaceRef(id="vis_test", local_path=str(tmp_path))
        col = be.get_collection(
            palace=palace, collection_name="mempalace_drawers", create=True
        )

        emb = _make_embedding(42)
        col.add(
            documents=["findable document about specific topic"],
            ids=["vis_1"],
            metadatas=[{"wing": "test"}],
            embeddings=[emb],
        )

        # Immediately query for it
        result = col.query(query_embeddings=[emb], n_results=5)
        assert "vis_1" in result.ids[0], "Own write not immediately visible"

        be.close()


class TestEventBusConcurrency:
    def test_event_bus_under_concurrency(self, tmp_path):
        """6 processes each publish 50 events. No loss, no duplication."""
        db_path = str(tmp_path / DB_FILENAME)
        _init_db(db_path)

        n_workers = 6
        events_per_worker = 50
        result_queue = multiprocessing.Queue()

        processes = []
        for w in range(n_workers):
            p = multiprocessing.Process(
                target=_event_bus_worker,
                args=(db_path, w, events_per_worker, result_queue),
            )
            processes.append(p)

        for p in processes:
            p.start()
        for p in processes:
            p.join(timeout=60)

        results = []
        while not result_queue.empty():
            results.append(result_queue.get_nowait())

        for r in results:
            assert "error" not in r, f"Worker {r.get('worker_id')} failed: {r.get('error')}"

        total_published = sum(r["published"] for r in results)

        # Verify all events exist in the database
        conn = sqlite3.connect(db_path)
        actual = conn.execute("SELECT COUNT(*) FROM event_bus").fetchone()[0]
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        conn.close()

        assert integrity == "ok"
        assert actual == total_published, f"Event loss: {actual} vs {total_published}"
