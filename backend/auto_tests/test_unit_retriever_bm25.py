import pytest
from unittest.mock import MagicMock
from app.services.bm25_service import BM25Service
from app.services.retriever import Retriever
from app.services.reranker_service import RerankerService


def test_bm25_service_index_and_search():
    service = BM25Service()
    docs = [
        {"id": "doc_1", "text": "Deepfake detection using frequency domain and FreqNet", "document_id": "d1", "document_name": "paper1.pdf", "page": 1, "chunk_index": 0},
        {"id": "doc_2", "text": "Kubernetes deployment for high availability backend", "document_id": "d2", "document_name": "k8s.pdf", "page": 1, "chunk_index": 0},
        {"id": "doc_3", "text": "Adaptive Soft Masking improves Deepfake detection accuracy", "document_id": "d1", "document_name": "paper1.pdf", "page": 2, "chunk_index": 1},
    ]
    service.index_documents(docs)

    # 1. Search keyword "Deepfake"
    results = service.search("Deepfake detection", top_k=2)
    assert len(results) == 2
    assert results[0]["id"] in ["doc_1", "doc_3"]
    assert results[0]["distance"] > 0

    # 2. Search keyword "Kubernetes"
    k8s_results = service.search("Kubernetes backend", top_k=5)
    assert len(k8s_results) == 1
    assert k8s_results[0]["id"] == "doc_2"


def test_bm25_service_add_and_remove():
    service = BM25Service()
    docs = [
        {"id": "doc_1", "text": "Python fastapi backend", "document_id": "d1", "document_name": "doc1.pdf", "page": 1, "chunk_index": 0},
    ]
    service.index_documents(docs)

    # Add document
    new_doc = [{"id": "doc_2", "text": "Docker container setup", "document_id": "d2", "document_name": "doc2.pdf", "page": 1, "chunk_index": 0}]
    service.add_documents(new_doc)
    assert len(service.documents) == 2

    # Search new doc
    res = service.search("Docker setup", top_k=5)
    assert len(res) == 1
    assert res[0]["id"] == "doc_2"

    # Remove document by document_id
    service.remove_documents_by_id("d2")
    assert len(service.documents) == 1
    res_after_remove = service.search("Docker setup", top_k=5)
    assert len(res_after_remove) == 0


def test_reranker_service_selects_top_3():
    # Giả lập RerankerService với mô hình CrossEncoder BGE
    reranker = RerankerService()
    mock_model = MagicMock()
    # Giả lập 5 ứng viên với các điểm số khác nhau từ BGE
    mock_model.predict.return_value = [0.12, 0.95, 0.45, 0.88, 0.23]
    reranker.model = mock_model
    reranker._is_loaded = True

    candidates = [
        {"id": f"chunk_{i}", "entity": {"text": f"Nội dung chunk {i}"}}
        for i in range(5)
    ]

    selected = reranker.rerank(
        query="Truy vấn kiểm tra",
        candidates=candidates,
        top_k=3,
    )

    # Kiểm tra chỉ lấy ra đúng Top 3 candidates
    assert len(selected) == 3
    # Candidate có điểm cao nhất (0.95 ở vị trí 1) phải đứng đầu
    assert selected[0]["id"] == "chunk_1"
    assert selected[0]["rerank_score"] == 0.95
    # Candidate có điểm cao thứ 2 (0.88 ở vị trí 3) phải đứng nhì
    assert selected[1]["id"] == "chunk_3"
    assert selected[1]["rerank_score"] == 0.88
    # Candidate có điểm cao thứ 3 (0.45 ở vị trí 2) phải đứng ba
    assert selected[2]["id"] == "chunk_2"
    assert selected[2]["rerank_score"] == 0.45


def test_retriever_pipeline_with_reranker_top_3():
    mock_embedding = MagicMock()
    mock_embedding.embed_chunks.return_value = [0.1] * 384

    mock_vector_store = MagicMock()
    # Giả lập 10 candidates từ Vector Search
    mock_vector_store.search.return_value = [[
        {"id": f"chunk_{i}", "distance": 0.9 - i * 0.05, "entity": {"text": f"Text {i}", "document_name": "doc.pdf", "page": 1}}
        for i in range(10)
    ]]

    # Giả lập RerankerService
    reranker = RerankerService()
    mock_model = MagicMock()
    # Chấm điểm cao cho chunk_7, chunk_2, chunk_5
    scores = [0.1] * 10
    scores[7] = 0.99
    scores[2] = 0.95
    scores[5] = 0.90
    mock_model.predict.return_value = scores
    reranker.model = mock_model
    reranker._is_loaded = True

    retriever = Retriever(
        embedding_service=mock_embedding,
        vector_store=mock_vector_store,
        reranker_service=reranker,
    )

    # Thực hiện retrieve với top_k=3
    results = retriever.retrieve("Truy vấn", top_k=3)

    assert isinstance(results, list)
    assert len(results) == 1
    hits = results[0]
    assert len(hits) == 3

    # Kiểm tra thứ tự do Reranker quyết định
    assert hits[0]["id"] == "chunk_7"
    assert hits[1]["id"] == "chunk_2"
    assert hits[2]["id"] == "chunk_5"
