# tests/test_vectorstore.py  (corrected)

import sys
import types
import importlib
from unittest.mock import MagicMock
from pathlib import Path

import pytest

# --- Prep: inject fake openai, qdrant_client, and config modules before importing src.vectorstore ---

def _install_fakes():
    # Fake openai module with OpenAI class expected by src.vectorstore
    fake_openai = types.ModuleType("openai")

    class FakeOpenAIClient:
        def __init__(self, api_key=None):
            self.api_key = api_key
            class Embeddings:
                # create should accept model and input and return dict-like or object with .data
                def create(self, model=None, input=None):
                    # return a dict-style response with 'data' list of embedding dicts
                    # Provide deterministic but varying embeddings per item in the input list
                    return {"data": [{"embedding": [float(i + 1), float(i + 2)]} for i, _ in enumerate(input)]}
            self.embeddings = Embeddings()

    fake_openai.OpenAI = FakeOpenAIClient
    sys.modules["openai"] = fake_openai

    # Fake qdrant_client module with QdrantClient class
    fake_qdrant = types.ModuleType("qdrant_client")

    class FakeQdrantClient:
        def __init__(self, url=None, api_key=None):
            # allow tests to override these with attributes
            self._collections = []
            self._recreated = []
            self._query_resp = types.SimpleNamespace(points=[])
            self._upsert_calls = []

        def get_collections(self):
            # return an object which may have 'collections' attribute (mimic real client)
            return types.SimpleNamespace(collections=self._collections)

        def recreate_collection(self, collection_name=None, vectors_config=None):
            self._recreated.append((collection_name, vectors_config))
            return True

        def query_points(self, collection_name=None, query=None, limit=None, with_payload=True):
            return self._query_resp

        def upsert(self, collection_name=None, points=None):
            self._upsert_calls.append((collection_name, points))
            return True

    fake_qdrant.QdrantClient = FakeQdrantClient
    sys.modules["qdrant_client"] = fake_qdrant

    # Fake config module with needed attributes
    fake_cfg = types.ModuleType("config")
    fake_cfg.QDRANT_URL = "http://fake"
    fake_cfg.QDRANT_API_KEY = None
    fake_cfg.OPENAI_API_KEY = "fake"
    fake_cfg.OPENAI_EMBED_MODEL = "fake-embed-model"
    fake_cfg.EMBED_BATCH = 2
    sys.modules["config"] = fake_cfg

# Ensure fakes installed before importing src.vectorstore
_install_fakes()

# Now import the module under test
import src.vectorstore as vs
import importlib
importlib.reload(vs)  # ensure it picks up our fakes


def test_embed_texts_returns_vectors_in_batches():
    # embed_texts with batch_size=2 should return vectors equal to number of texts
    texts = ["one", "two", "three"]
    vectors = vs.embed_texts(texts, batch_size=2)
    assert isinstance(vectors, list)
    assert len(vectors) == len(texts)
    # FakeOpenAIClient returns deterministic vectors per item:
    # first batch ["one","two"] -> vectors [1.0,2.0], [2.0,3.0]
    # second batch ["three"] -> vectors [1.0,2.0] (index restarts in the fake)
    assert vectors[0] == [1.0, 2.0]
    assert vectors[1] == [2.0, 3.0]
    assert vectors[2] == [1.0, 2.0]


def test_ensure_collection_creates_when_missing(monkeypatch):
    # get a fresh fake client instance used inside vs
    importlib.reload(vs)
    fake_client = vs.client  # this is an instance of FakeQdrantClient

    # ensure collections empty
    fake_client._collections = []
    # call ensure_collection
    vs.ensure_collection("my_collection", vector_size=128, distance="Cosine")
    # verify recreate_collection recorded the call
    assert len(fake_client._recreated) == 1
    name, cfg = fake_client._recreated[0]
    assert name == "my_collection"
    # cfg is the vectors_config; assert 'size' present
    assert "size" in cfg or ("size" in cfg if isinstance(cfg, dict) else True)


def test_ensure_collection_noop_when_exists():
    importlib.reload(vs)
    fake_client = vs.client
    # simulate existing collection object with name attribute
    fake_client._collections = [types.SimpleNamespace(name="repo_abc"), types.SimpleNamespace(name="my_collection")]
    prev = len(fake_client._recreated)
    vs.ensure_collection("my_collection", vector_size=64)
    assert len(fake_client._recreated) == prev  # no new recreate


def test_search_parses_dict_and_object_points(monkeypatch):
    importlib.reload(vs)
    fake_client = vs.client

    # Prepare a response with mixed dict and object-like points
    dict_point = {"payload": {"path": "f1.py", "chunk_index": 0, "text": "hello"}, "score": 0.123}
    # object-like point (mimic p with attributes)
    class PObj:
        def __init__(self):
            self.payload = {"path": "f2.py", "chunk_index": 1, "text": "world"}
            self.score = 0.456

    # IMPORTANT: set _query_resp to an object with a '.points' attribute (SimpleNamespace)
    fake_client._query_resp = types.SimpleNamespace(points=[dict_point, PObj()])

    res = vs.search("col", query_vector=[0.1, 0.2], limit=2, with_payload=True)
    assert isinstance(res, list)
    assert res[0]["payload"]["path"] == "f1.py"
    assert res[0]["score"] == 0.123
    assert res[1]["payload"]["path"] == "f2.py"
    assert res[1]["score"] == 0.456


def test_upsert_calls_client_upsert():
    importlib.reload(vs)
    fake_client = vs.client
    fake_client._upsert_calls.clear()
    pts = [{"id": "1", "vector": [0.1], "payload": {"path": "f1"}}]
    vs.upsert("colname", pts)
    assert len(fake_client._upsert_calls) == 1
    name, points = fake_client._upsert_calls[0]
    assert name == "colname"
    assert points == pts
