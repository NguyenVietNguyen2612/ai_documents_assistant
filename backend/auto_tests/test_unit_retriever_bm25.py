try:
    import pytest
except ImportError:
    pytest = None
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


def test_retriever_ablation_modes_vector_vs_hybrid_vs_rerank():
    """Kiểm tra 3 chế độ truy xuất phục vụ Ablation Study: vector_only, hybrid_rrf, hybrid_rerank."""
    mock_embedding = MagicMock()
    mock_embedding.embed_chunks.return_value = [0.1] * 384

    mock_vector_store = MagicMock()
    # Vector search trả về chunk_v1, chunk_v2, chunk_v3
    mock_vector_store.search.return_value = [[
        {"id": "chunk_v1", "distance": 0.92, "entity": {"text": "Vector 1", "document_name": "cs338.pdf", "page": 1}},
        {"id": "chunk_v2", "distance": 0.85, "entity": {"text": "Vector 2", "document_name": "cs338.pdf", "page": 2}},
        {"id": "chunk_v3", "distance": 0.78, "entity": {"text": "Vector 3", "document_name": "cs338.pdf", "page": 3}},
    ]]

    # BM25 trả về chunk_b1 (trùng với chunk_v2) và chunk_b2
    mock_bm25 = MagicMock()
    mock_bm25.search.return_value = [
        {"id": "chunk_v2", "distance": 12.5, "score": 12.5, "entity": {"text": "Vector 2", "document_name": "cs338.pdf", "page": 2}},
        {"id": "chunk_b2", "distance": 10.1, "score": 10.1, "entity": {"text": "BM25 exact keyword HFRI HFRF FCL", "document_name": "cs338.pdf", "page": 2}},
    ]

    # Reranker đảo chunk_b2 lên vị trí số 1
    mock_reranker = MagicMock()
    def fake_rerank(query, candidates, top_k=3):
        by_id = {c["id"]: c for c in candidates}
        ordered = []
        if "chunk_b2" in by_id:
            item = dict(by_id["chunk_b2"])
            item["rerank_score"] = 0.99
            ordered.append(item)
        for c in candidates:
            if c["id"] != "chunk_b2" and len(ordered) < top_k:
                ordered.append(c)
        return ordered[:top_k]

    mock_reranker.rerank.side_effect = fake_rerank

    retriever = Retriever(
        embedding_service=mock_embedding,
        vector_store=mock_vector_store,
        bm25_service=mock_bm25,
        reranker_service=mock_reranker,
    )

    # 1. Mode vector_only: Không gọi BM25 hay Reranker
    res_vec = retriever.retrieve("What are HFRI HFRF FCL?", top_k=2, mode="vector_only")[0]
    assert len(res_vec) == 2
    assert res_vec[0]["id"] == "chunk_v1"
    mock_bm25.search.assert_not_called()
    mock_reranker.rerank.assert_not_called()

    # 2. Mode hybrid_rrf: Gọi cả Vector + BM25 và hợp nhất RRF (chunk_v2 xuất hiện ở cả 2 bên nên đứng đầu RRF), không gọi Reranker
    res_rrf = retriever.retrieve("What are HFRI HFRF FCL?", top_k=2, mode="hybrid_rrf")[0]
    assert len(res_rrf) == 2
    assert res_rrf[0]["id"] == "chunk_v2"
    mock_bm25.search.assert_called_once()
    mock_reranker.rerank.assert_not_called()

    # 3. Mode hybrid_rerank: Gọi đầy đủ Stage 1 (RRF) + Stage 2 (BGE Reranker), đưa chunk_b2 lên vị trí #1
    res_rerank = retriever.retrieve("What are HFRI HFRF FCL?", top_k=2, mode="hybrid_rerank")[0]
    assert len(res_rerank) == 2
    assert res_rerank[0]["id"] == "chunk_b2"
    mock_reranker.rerank.assert_called_once()


def test_small_to_big_window_expansion_merges_adjacent_chunks():
    """Kiểm tra cơ chế Small-to-Big Window Expansion lấy và nối liền các chunk liền kề."""
    mock_embedding = MagicMock()
    mock_vector_store = MagicMock()
    mock_bm25 = MagicMock()
    mock_reranker = MagicMock()

    # Giả lập VectorStore trả về chunk liền kề 1, 2, 3
    mock_vector_store.get_neighboring_chunks.return_value = [
        {"id": "c1", "chunk_index": 1, "page": 1, "text": "Đoạn 1 mở đầu định nghĩa FreqNet."},
        {"id": "c2", "chunk_index": 2, "page": 1, "text": "Đoạn 2 là phần trọng tâm về HFRI và HFRF."},
        {"id": "c3", "chunk_index": 3, "page": 2, "text": "Đoạn 3 kết luận với khối FCL."},
    ]

    retriever = Retriever(
        embedding_service=mock_embedding,
        vector_store=mock_vector_store,
        bm25_service=mock_bm25,
        reranker_service=mock_reranker,
    )

    hits = [{
        "id": "c2",
        "entity": {
            "document_id": "doc_123",
            "document_name": "CS338.pdf",
            "page": 1,
            "chunk_index": 2,
            "text": "Đoạn 2 là phần trọng tâm về HFRI và HFRF.",
        }
    }]

    expanded = retriever.expand_window(hits, window_size=1)
    assert len(expanded) == 1
    ent = expanded[0]["entity"]

    assert ent["has_expanded"] is True
    assert ent["chunk_range"] == "1-3"
    assert "Đoạn 1 mở đầu" in ent["expanded_text"]
    assert "Đoạn 2 là phần trọng tâm" in ent["expanded_text"]
    assert "Đoạn 3 kết luận" in ent["expanded_text"]
    assert ent["page_range"] == "1-2"
    mock_vector_store.get_neighboring_chunks.assert_called_once_with(
        document_id="doc_123",
        chunk_index=2,
        window_size=1,
    )


def test_context_builder_small_to_big_and_corrective_rag_warning():
    """Kiểm tra ContextBuilder hiển thị cửa sổ Small-to-Big và nhãn Corrective RAG."""
    from app.services.context_builder import ContextBuilder

    cb = ContextBuilder()

    # Case 1: Điểm tin cậy cao (>= 0.70) và có Small-to-Big window
    high_conf_hits = [[{
        "id": "c2",
        "normalized_score": 0.88,
        "entity": {
            "document_name": "CS338.pdf",
            "page": 1,
            "page_range": "1-2",
            "chunk_range": "1-3",
            "has_expanded": True,
            "expanded_text": "Văn bản mở rộng liên tục từ chunk 1 đến 3.",
            "text": "Văn bản gốc chunk 2.",
        }
    }]]

    context_high = cb.build(high_conf_hits)
    assert "Cửa sổ Small-to-Big: Chunks 1-3" in context_high
    assert "Trang 1-2" in context_high
    assert "Văn bản mở rộng liên tục" in context_high
    assert "ĐỘ TIN CẬY CAO" in context_high

    # Case 2: Điểm tin cậy thấp (< 0.40) -> Kích hoạt cảnh báo Corrective RAG & Query Rewriting
    low_conf_hits = [[{
        "id": "c1",
        "normalized_score": 0.25,
        "entity": {
            "document_name": "Random.pdf",
            "page": 1,
            "chunk_index": 0,
            "text": "Nội dung mơ hồ không khớp.",
        }
    }]]

    context_low = cb.build(low_conf_hits)
    assert "CẢNH BÁO ĐỘ TIN CẬY THẤP" in context_low
    assert "QUERY REWRITING" in context_low
    assert "gọi lại `retrieve_documents`" in context_low


