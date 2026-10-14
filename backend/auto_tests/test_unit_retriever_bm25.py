import pytest
from unittest.mock import MagicMock
from app.services.bm25_service import BM25Service
from app.services.retriever import Retriever


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


def test_retriever_rrf_and_top_10():
    # Tạo mock cho embedding_service và vector_store
    mock_embedding = MagicMock()
    mock_embedding.embed_chunks.return_value = [0.1] * 384

    mock_vector_store = MagicMock()
    # Giả lập 10 candidates từ Vector Search
    mock_vector_store.search.return_value = [[
        {"id": f"v_{i}", "distance": 0.9 - i * 0.05, "entity": {"text": f"Vector chunk {i}", "document_name": "v.pdf", "page": 1}}
        for i in range(10)
    ]]

    # Giả lập BM25 Service với 10 candidates
    bm25 = BM25Service()
    bm25_docs = [
        {"id": f"v_{i}", "text": f"Hybrid chunk {i}", "document_id": "d", "document_name": "v.pdf", "page": 1, "chunk_index": i}
        for i in range(5)
    ] + [
        {"id": f"b_{i}", "text": f"BM25 specific chunk {i}", "document_id": "d", "document_name": "b.pdf", "page": 1, "chunk_index": i}
        for i in range(5)
    ]
    bm25.index_documents(bm25_docs)

    retriever = Retriever(
        embedding_service=mock_embedding,
        vector_store=mock_vector_store,
        bm25_service=bm25,
    )

    # Thực hiện retrieve với top_k=10
    results = retriever.retrieve("Hybrid chunk", top_k=10)

    # Kiểm tra kết quả trả về đúng định dạng danh sách lồng nhau [[...]]
    assert isinstance(results, list)
    assert len(results) == 1
    hits = results[0]
    assert len(hits) == 10

    # Kiểm tra các hit đều có rrf_score và id
    for hit in hits:
        assert "id" in hit
        assert "rrf_score" in hit
        assert "entity" in hit
