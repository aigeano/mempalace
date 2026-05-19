"""Invariant tests: search uses cosine similarity.

The SQLite+NumPy backend computes cosine similarity via dot product on
L2-normalized vectors. These tests verify that the palace module and
backend produce working collections where semantic search returns
meaningful similarity scores (not flat zeros).
"""

from mempalace.backends.base import PalaceRef
from mempalace.backends.sqlite_backend import SqliteBackend
from mempalace.embedding import get_embedding_function
from mempalace.palace import get_collection


_test_embed_fn = None


def _get_test_embed_fn():
    global _test_embed_fn
    if _test_embed_fn is None:
        _test_embed_fn = get_embedding_function()
    return _test_embed_fn


def test_palace_module_get_collection_returns_searchable(tmp_path):
    """The public ``mempalace.palace.get_collection`` must produce a
    collection that supports upsert + query with cosine distances."""
    col = get_collection(str(tmp_path), "mempalace_drawers", create=True)
    col.upsert(
        ids=["d1"],
        documents=["the cat sat on the mat"],
        metadatas=[{"wing": "test", "room": "r1"}],
    )
    results = col.query(query_texts=["cat sitting"], n_results=1)
    assert len(results["ids"][0]) == 1
    assert results["distances"][0][0] >= 0


def test_fresh_palace_via_full_stack_gets_cosine(tmp_path):
    """End-to-end: build a palace with the public API, upsert docs, and
    confirm that similar documents have higher similarity than dissimilar."""
    col = get_collection(str(tmp_path), "mempalace_drawers", create=True)
    col.upsert(
        ids=["d_cat", "d_math"],
        documents=["the cat sat on the mat", "integral calculus theory"],
        metadatas=[
            {"wing": "test", "room": "r1"},
            {"wing": "test", "room": "r2"},
        ],
    )
    results = col.query(query_texts=["feline sitting"], n_results=2)
    ids = results["ids"][0]
    dists = results["distances"][0]
    cat_idx = ids.index("d_cat")
    math_idx = ids.index("d_math")
    assert dists[cat_idx] < dists[math_idx], (
        "cat doc should be closer to 'feline sitting' than calculus doc"
    )


def test_sqlite_backend_collection_supports_cosine_query(tmp_path):
    """Direct SqliteBackend collection query returns cosine distances."""
    backend = SqliteBackend()
    palace = PalaceRef(id=str(tmp_path), local_path=str(tmp_path))
    col = backend.get_collection(
        palace=palace,
        collection_name="mempalace_drawers",
        create=True,
        options={"embed_fn": _get_test_embed_fn()},
    )
    col.upsert(
        ids=["a", "b"],
        documents=["python programming language", "french cuisine recipes"],
        metadatas=[{"wing": "w", "room": "r"}] * 2,
    )
    results = col.query(query_texts=["coding in python"], n_results=2)
    assert results["ids"][0][0] == "a"
    backend.close()
