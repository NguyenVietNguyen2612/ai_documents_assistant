import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.ablation_study import (
    InMemoryCosineVectorStore,
    compute_query_metrics,
    load_beir_dataset,
)


def test_load_beir_dataset_and_sort_order():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "qrels").mkdir()

        corpus_rows = [
            {"_id": "d1", "title": "Title One", "text": "Body about deepfake."},
            {"_id": "d2", "title": "Title Two", "text": "Body about frequency."},
        ]
        with open(root / "corpus.jsonl", "w", encoding="utf-8") as f:
            for row in corpus_rows:
                f.write(json.dumps(row) + "\n")

        queries_rows = [
            {"_id": "10", "text": "Query ten"},
            {"_id": "2", "text": "Query two"},
            {"_id": "99", "text": "Unused query without qrels"},
        ]
        with open(root / "queries.jsonl", "w", encoding="utf-8") as f:
            for row in queries_rows:
                f.write(json.dumps(row) + "\n")

        qrel_rows = [
            {"query-id": "10", "corpus-id": "d2", "score": "1"},
            {"query-id": "2", "corpus-id": "d1", "score": "1"},
        ]
        with open(root / "qrels" / "test.jsonl", "w", encoding="utf-8") as f:
            for row in qrel_rows:
                f.write(json.dumps(row) + "\n")

        corpus, queries, qrels, _ = load_beir_dataset(
            dataset_name="custom/test",
            split="test",
            dataset_dir=str(root),
        )

        assert len(corpus) == 2
        assert corpus[0]["text"] == "Title One\nBody about deepfake."
        assert [q["id"] for q in queries] == ["2", "10"]
        assert qrels["2"] == {"d1": 1}


def test_compute_query_metrics_values():
    metrics = compute_query_metrics(
        retrieved_ids=["d_wrong", "d_rel_1", "d_other", "d_rel_2", "d_last"],
        relevant_qrels={"d_rel_1": 1, "d_rel_2": 1},
        top_k=5,
    )
    assert metrics["first_rank"] == 2
    assert metrics["hit_1"] == 0.0
    assert metrics["hit_3"] == 1.0
    assert metrics["hit_k"] == 1.0
    assert abs(metrics["mrr_k"] - 0.5) < 1e-6
    assert abs(metrics["recall_k"] - 1.0) < 1e-6
    assert 0.0 < metrics["ndcg_k"] < 1.0


def test_in_memory_cosine_vector_store_ranking():
    vs = InMemoryCosineVectorStore()
    vs.create_collection(dimension=3)
    vs.insert(
        [
            {
                "id": "doc_a",
                "vector": [1.0, 0.0, 0.0],
                "text": "Alpha",
                "document_id": "doc_a",
                "document_name": "a",
                "page": 1,
                "chunk_index": 0,
            },
            {
                "id": "doc_b",
                "vector": [0.0, 1.0, 0.0],
                "text": "Beta",
                "document_id": "doc_b",
                "document_name": "b",
                "page": 1,
                "chunk_index": 0,
            },
        ]
    )
    res = vs.search(query_vector=[0.9, 0.1, 0.0], limit=2)
    assert len(res) == 1
    assert [hit["id"] for hit in res[0]] == ["doc_a", "doc_b"]
    assert res[0][0]["distance"] > res[0][1]["distance"]


if __name__ == "__main__":
    test_load_beir_dataset_and_sort_order()
    test_compute_query_metrics_values()
    test_in_memory_cosine_vector_store_ranking()
    print("All ablation benchmark unit tests passed!")
