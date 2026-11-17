import argparse
import contextlib
import csv
import io
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from huggingface_hub import snapshot_download

# Đảm bảo Python có thể import được thư mục 'app' ở thư mục cha
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.services.bm25_service import BM25Service
from app.services.embedding_service import EmbeddingService
from app.services.reranker_service import RerankerService
from app.services.retriever import Retriever
from app.services.vector_store import VectorStore


DEFAULT_DATASET = "mteb/scifact"
DEFAULT_SPLIT = "test"
DEFAULT_TOP_K = 5
MILVUS_TEXT_MAX_BYTES = 9500


def _truncate_utf8(text: str, max_bytes: int = MILVUS_TEXT_MAX_BYTES) -> str:
    """Cắt an toàn theo UTF-8 bytes để không vượt giới hạn VARCHAR(10000) của Milvus."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


class InMemoryCosineVectorStore:
    """
    Fallback VectorStore trong bộ nhớ (Normalized Cosine Similarity bằng NumPy)
    khi Milvus server chưa được khởi động. Giữ nguyên schema trả về của Milvus.
    """

    def __init__(self, collection_name: str = "in_memory_benchmark"):
        self.collection_name = collection_name
        self._records: List[Dict[str, Any]] = []
        self._norm_matrix: Optional[np.ndarray] = None

    def create_collection(self, dimension: int):
        self._records = []
        self._norm_matrix = None

    def create_index(self):
        pass

    def load_collection(self):
        pass

    def insert(self, data: List[Dict[str, Any]]):
        if not data:
            return
        self._records.extend(data)
        mat = np.asarray([item["vector"] for item in self._records], dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        self._norm_matrix = mat / norms

    def search(
        self,
        query_vector: List[float],
        limit: int = 15,
        document_name: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
    ) -> List[List[Dict[str, Any]]]:
        if self._norm_matrix is None or len(self._records) == 0:
            return [[]]

        q = np.asarray(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm

        scores = self._norm_matrix @ q

        if document_ids:
            allowed = set(document_ids)
            candidate_indices = [
                i for i, rec in enumerate(self._records) if rec.get("document_id") in allowed
            ]
            if not candidate_indices:
                return [[]]
            sub_scores = scores[candidate_indices]
            k = min(limit, len(candidate_indices))
            top_local = np.argpartition(-sub_scores, k - 1)[:k] if k < len(sub_scores) else np.arange(len(sub_scores))
            top_local = top_local[np.argsort(-sub_scores[top_local])]
            top_indices = [candidate_indices[idx] for idx in top_local]
        else:
            k = min(limit, len(self._records))
            top_indices = np.argpartition(-scores, k - 1)[:k] if k < len(scores) else np.arange(len(scores))
            top_indices = top_indices[np.argsort(-scores[top_indices])]

        hits = []
        for idx in top_indices:
            rec = self._records[int(idx)]
            hits.append(
                {
                    "id": rec["id"],
                    "distance": float(scores[int(idx)]),
                    "entity": {
                        "text": rec["text"],
                        "document_id": rec["document_id"],
                        "document_name": rec["document_name"],
                        "page": rec["page"],
                        "chunk_index": rec["chunk_index"],
                    },
                }
            )
        return [hits]

    def get_neighboring_chunks(
        self,
        document_id: str,
        chunk_index: int,
        window_size: int = 1,
    ) -> List[Dict[str, Any]]:
        return []


def _resolve_dataset_dir(dataset_name: str, dataset_dir: Optional[str] = None) -> Path:
    """Lấy đường dẫn bộ dữ liệu BEIR chuẩn từ thư mục local hoặc HuggingFace Hub cache."""
    if dataset_dir:
        path = Path(dataset_dir).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục dataset: {path}")
        return path

    try:
        local_path = snapshot_download(
            repo_id=dataset_name,
            repo_type="dataset",
            local_files_only=True,
        )
        return Path(local_path)
    except Exception:
        print(f"[Dataset] Chưa có cache local cho '{dataset_name}', đang tải từ HuggingFace Hub...")
        downloaded_path = snapshot_download(
            repo_id=dataset_name,
            repo_type="dataset",
        )
        return Path(downloaded_path)


def _load_qrels(qrels_dir: Path, split: str) -> Dict[str, Dict[str, int]]:
    """Đọc file qrels chuẩn BEIR/MTEB (hỗ trợ cả .jsonl và .tsv)."""
    jsonl_path = qrels_dir / f"{split}.jsonl"
    tsv_path = qrels_dir / f"{split}.tsv"

    qrels: Dict[str, Dict[str, int]] = {}
    if jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                qid = str(row["query-id"])
                cid = str(row["corpus-id"])
                score = int(row.get("score", 1))
                if score > 0:
                    qrels.setdefault(qid, {})[cid] = score
    elif tsv_path.exists():
        with open(tsv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                qid = str(row["query-id"])
                cid = str(row["corpus-id"])
                score = int(row.get("score", 1))
                if score > 0:
                    qrels.setdefault(qid, {})[cid] = score
    else:
        raise FileNotFoundError(
            f"Không tìm thấy file qrels cho split '{split}' trong {qrels_dir} (.jsonl hoặc .tsv)"
        )
    return qrels


def load_beir_dataset(
    dataset_name: str = DEFAULT_DATASET,
    split: str = DEFAULT_SPLIT,
    dataset_dir: Optional[str] = None,
    max_queries: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Dict[str, int]], Path]:
    """
    Nạp bộ dữ liệu chuẩn định dạng BEIR/MTEB (mặc định: mteb/scifact).
    Trả về: (corpus_docs, eval_queries, qrels, resolved_path)
    """
    base_dir = _resolve_dataset_dir(dataset_name=dataset_name, dataset_dir=dataset_dir)
    corpus_path = base_dir / "corpus.jsonl"
    queries_path = base_dir / "queries.jsonl"
    qrels_dir = base_dir / "qrels"

    if not corpus_path.exists() or not queries_path.exists():
        raise FileNotFoundError(
            f"Thư mục dataset {base_dir} thiếu corpus.jsonl hoặc queries.jsonl."
        )

    corpus_docs: List[Dict[str, Any]] = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            doc_id = str(item["_id"])
            title = (item.get("title") or "").strip()
            body = (item.get("text") or "").strip()
            full_text = f"{title}\n{body}".strip() if title else body
            corpus_docs.append(
                {
                    "id": doc_id,
                    "document_id": doc_id,
                    "document_name": f"{dataset_name.split('/')[-1]}:{doc_id}",
                    "page": 1,
                    "chunk_index": 0,
                    "title": title,
                    "text": full_text,
                }
            )

    qrels = _load_qrels(qrels_dir=qrels_dir, split=split)

    queries_map: Dict[str, str] = {}
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            qid = str(item["_id"])
            if qid in qrels:
                queries_map[qid] = (item.get("text") or "").strip()

    def _sort_key(qid: str):
        return (0, int(qid)) if qid.isdigit() else (1, qid)

    sorted_qids = sorted(queries_map.keys(), key=_sort_key)
    if max_queries is not None and max_queries > 0:
        sorted_qids = sorted_qids[:max_queries]

    eval_queries = [{"id": qid, "question": queries_map[qid]} for qid in sorted_qids]
    return corpus_docs, eval_queries, qrels, base_dir


def _get_or_compute_embeddings(
    corpus_docs: List[Dict[str, Any]],
    embedding_service: EmbeddingService,
    dataset_name: str,
    cache_dir: Optional[str],
) -> np.ndarray:
    """Tính toán hoặc tải từ cache bộ vector embeddings cho toàn bộ corpus."""
    safe_ds = dataset_name.replace("/", "__")
    cache_file = None
    if cache_dir:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        cache_file = cache_path / f"{safe_ds}_embeddings_{len(corpus_docs)}.npy"
        if cache_file.exists():
            cached = np.load(cache_file)
            if cached.shape[0] == len(corpus_docs):
                print(f"[Cache] Đã nạp sẵn {cached.shape[0]} corpus embeddings từ {cache_file}")
                return cached

    print(f"[Setup] Đang tạo vector embeddings cho {len(corpus_docs)} tài liệu...")
    t0 = time.perf_counter()
    texts = [item["text"] for item in corpus_docs]
    vectors = embedding_service.model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
    )
    arr = np.asarray(vectors, dtype=np.float32)
    elapsed = time.perf_counter() - t0
    print(f"[Setup] Hoàn tất nhúng {len(corpus_docs)} tài liệu trong {elapsed:.1f}s (dim={arr.shape[1]}).")

    if cache_file is not None:
        np.save(cache_file, arr)
        print(f"[Cache] Đã lưu cache embeddings tại {cache_file}")

    return arr


def _init_vector_store(
    backend_mode: str,
    collection_name: str,
) -> Tuple[Any, str]:
    """Khởi tạo VectorStore (Milvus ưu tiên hoặc fallback InMemoryCosineVectorStore)."""
    if backend_mode == "memory":
        return InMemoryCosineVectorStore(collection_name=collection_name), "In-Memory Cosine (NumPy)"

    try:
        vs = VectorStore(collection_name=collection_name)
        vs.client.has_collection(collection_name)
        return vs, f"Milvus ({vs.uri})"
    except Exception as e:
        if backend_mode == "milvus":
            raise RuntimeError(f"Không thể kết nối Milvus server ở chế độ --backend milvus: {e}") from e
        print(
            f"[Warning] Milvus server chưa sẵn sàng ({e}). "
            "Tự động chuyển sang In-Memory Cosine VectorStore để thực thi benchmark."
        )
        return InMemoryCosineVectorStore(collection_name=collection_name), "In-Memory Cosine (NumPy Fallback)"


def compute_query_metrics(
    retrieved_ids: List[str],
    relevant_qrels: Dict[str, int],
    top_k: int,
) -> Dict[str, Any]:
    """Tính các chỉ số chuẩn Information Retrieval: Hit@1, Hit@3, Hit@K, MRR@K, nDCG@K, Recall@K."""
    top_ids = retrieved_ids[:top_k]

    first_rank: Optional[int] = None
    for idx, doc_id in enumerate(top_ids, start=1):
        if doc_id in relevant_qrels:
            first_rank = idx
            break

    hit_1 = 1.0 if (len(top_ids) >= 1 and top_ids[0] in relevant_qrels) else 0.0
    hit_3 = 1.0 if any(doc_id in relevant_qrels for doc_id in top_ids[:3]) else 0.0
    hit_k = 1.0 if first_rank is not None else 0.0
    mrr_k = (1.0 / first_rank) if first_rank is not None else 0.0

    matched_rel = sum(1 for doc_id in top_ids if doc_id in relevant_qrels)
    recall_k = matched_rel / len(relevant_qrels) if relevant_qrels else 0.0

    dcg = 0.0
    for idx, doc_id in enumerate(top_ids, start=1):
        rel = relevant_qrels.get(doc_id, 0)
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(idx + 1)

    ideal_rels = sorted(relevant_qrels.values(), reverse=True)[:top_k]
    idcg = sum((2**rel - 1) / math.log2(idx + 1) for idx, rel in enumerate(ideal_rels, start=1))
    ndcg_k = (dcg / idcg) if idcg > 0 else 0.0

    return {
        "first_rank": first_rank,
        "hit_1": hit_1,
        "hit_3": hit_3,
        "hit_k": hit_k,
        "mrr_k": mrr_k,
        "recall_k": recall_k,
        "ndcg_k": ndcg_k,
    }


def _compute_win_loss(
    per_query_records: Dict[str, Dict[str, Any]],
    eval_queries: List[Dict[str, Any]],
    candidate_mode: str,
    baseline_mode: str,
    top_k: int,
) -> Dict[str, int]:
    """So sánh thứ hạng tài liệu đúng đầu tiên giữa 2 chế độ (Win / Tie / Loss)."""
    wins = 0
    ties = 0
    losses = 0
    miss_rank = top_k + 1

    for q in eval_queries:
        qid = q["id"]
        r_cand = per_query_records[qid][candidate_mode]["first_rank"] or miss_rank
        r_base = per_query_records[qid][baseline_mode]["first_rank"] or miss_rank
        if r_cand < r_base:
            wins += 1
        elif r_cand == r_base:
            ties += 1
        else:
            losses += 1
    return {"wins": wins, "ties": ties, "losses": losses}


def run_ablation_benchmark(
    dataset_name: str = DEFAULT_DATASET,
    split: str = DEFAULT_SPLIT,
    dataset_dir: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
    max_queries: Optional[int] = None,
    backend_mode: str = "auto",
    cache_dir: Optional[str] = None,
    report_path: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    print("=========================================================================")
    print("  ABLATION STUDY: DANH GIA RETRIEVAL & RERANKER TREN BO DU LIEU CHUAN  ")
    print("=========================================================================")

    corpus_docs, eval_queries, qrels, resolved_dir = load_beir_dataset(
        dataset_name=dataset_name,
        split=split,
        dataset_dir=dataset_dir,
        max_queries=max_queries,
    )
    total_qrels = sum(len(qrels[q["id"]]) for q in eval_queries)
    print(
        f"[Dataset] {dataset_name} (split='{split}') | Corpus: {len(corpus_docs)} docs | "
        f"Queries: {len(eval_queries)} | Relevance pairs: {total_qrels}"
    )

    # 1. Khởi tạo các dịch vụ thực tế của hệ thống
    embedding_service = EmbeddingService()
    reranker_service = RerankerService()
    reranker_service._ensure_model_loaded()
    # Warm-up Cross-Encoder trước khi đo Latency
    if reranker_service.model is not None:
        reranker_service.model.predict([("warmup query", "warmup document text")])

    embeddings = _get_or_compute_embeddings(
        corpus_docs=corpus_docs,
        embedding_service=embedding_service,
        dataset_name=dataset_name,
        cache_dir=cache_dir,
    )

    benchmark_collection = f"ablation_benchmark_{dataset_name.split('/')[-1]}"
    vector_store, backend_label = _init_vector_store(
        backend_mode=backend_mode,
        collection_name=benchmark_collection,
    )
    bm25_service = BM25Service()

    is_milvus = isinstance(vector_store, VectorStore)
    if is_milvus and vector_store.client.has_collection(benchmark_collection):
        vector_store.client.drop_collection(benchmark_collection)

    dim = int(embeddings.shape[1])
    vector_store.create_collection(dimension=dim)
    vector_store.create_index()

    records = []
    for item, vec in zip(corpus_docs, embeddings):
        text_val = _truncate_utf8(item["text"]) if is_milvus else item["text"]
        records.append(
            {
                "id": item["id"],
                "vector": vec.tolist() if is_milvus else vec,
                "text": text_val,
                "document_id": item["document_id"],
                "document_name": item["document_name"],
                "page": item["page"],
                "chunk_index": item["chunk_index"],
            }
        )

    if is_milvus:
        batch_size = 500
        for start in range(0, len(records), batch_size):
            vector_store.insert(records[start : start + batch_size])
    else:
        vector_store.insert(records)

    vector_store.load_collection()
    bm25_service.index_documents(corpus_docs)

    retriever = Retriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
        bm25_service=bm25_service,
        reranker_service=reranker_service,
    )

    modes = [
        ("vector_only", "1. Dense Vector Only (Bi-Encoder Cosine)"),
        ("hybrid_rrf", "2. Hybrid Search (Dense Vector + BM25 + RRF)"),
        ("hybrid_rerank", "3. 2-Stage Hybrid + BGE Cross-Encoder Reranker"),
    ]

    summary_metrics: Dict[str, Dict[str, Any]] = {}
    per_query_records: Dict[str, Dict[str, Any]] = {
        q["id"]: {"question": q["question"], "relevant_ids": list(qrels[q["id"]].keys())}
        for q in eval_queries
    }
    initial_k = max(10, top_k * 3)

    try:
        for mode_key, mode_label in modes:
            print(f"\n--- Đang chạy Benchmark chế độ: {mode_label} (top_k={top_k}, stage1_k={initial_k}) ---")
            hit_1_list = []
            hit_3_list = []
            hit_k_list = []
            mrr_list = []
            ndcg_list = []
            recall_list = []
            latencies_ms = []

            for idx, q_item in enumerate(eval_queries, start=1):
                qid = q_item["id"]
                question = q_item["question"]
                rel_dict = qrels[qid]

                t0 = time.perf_counter()
                with contextlib.redirect_stdout(io.StringIO()):
                    raw_results = retriever.retrieve(
                        query=question,
                        top_k=top_k,
                        mode=mode_key,
                        expand_window=False,
                    )
                results = (
                    raw_results[0]
                    if raw_results and isinstance(raw_results[0], list)
                    else (raw_results or [])
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                latencies_ms.append(elapsed_ms)

                retrieved_ids = [str(r.get("id")) for r in results]
                metrics = compute_query_metrics(retrieved_ids, rel_dict, top_k=top_k)

                hit_1_list.append(metrics["hit_1"])
                hit_3_list.append(metrics["hit_3"])
                hit_k_list.append(metrics["hit_k"])
                mrr_list.append(metrics["mrr_k"])
                ndcg_list.append(metrics["ndcg_k"])
                recall_list.append(metrics["recall_k"])

                per_query_records[qid][mode_key] = {
                    "first_rank": metrics["first_rank"],
                    "top_id": retrieved_ids[0] if retrieved_ids else "None",
                    "ndcg_k": round(metrics["ndcg_k"], 4),
                    "latency_ms": round(elapsed_ms, 1),
                }

                if idx % 25 == 0 or idx == len(eval_queries):
                    cur_mrr = sum(mrr_list) / len(mrr_list)
                    cur_ndcg = sum(ndcg_list) / len(ndcg_list)
                    cur_h1 = (sum(hit_1_list) / len(hit_1_list)) * 100
                    print(
                        f"  [{idx}/{len(eval_queries)}] Running Hit@1={cur_h1:.1f}% | "
                        f"MRR@{top_k}={cur_mrr:.4f} | nDCG@{top_k}={cur_ndcg:.4f} | "
                        f"AvgLat={sum(latencies_ms)/len(latencies_ms):.1f}ms"
                    )

            n_q = len(eval_queries)
            sorted_lat = sorted(latencies_ms)
            p95_lat = sorted_lat[min(n_q - 1, int(n_q * 0.95))]

            summary_metrics[mode_key] = {
                "label": mode_label,
                "hit_rate_at_1": round((sum(hit_1_list) / n_q) * 100, 2),
                "hit_rate_at_3": round((sum(hit_3_list) / n_q) * 100, 2),
                "hit_rate_at_k": round((sum(hit_k_list) / n_q) * 100, 2),
                "mrr_at_k": round(sum(mrr_list) / n_q, 4),
                "ndcg_at_k": round(sum(ndcg_list) / n_q, 4),
                "recall_at_k": round((sum(recall_list) / n_q) * 100, 2),
                "avg_latency_ms": round(sum(latencies_ms) / n_q, 1),
                "p95_latency_ms": round(p95_lat, 1),
            }
    finally:
        if is_milvus and vector_store.client.has_collection(benchmark_collection):
            vector_store.client.drop_collection(benchmark_collection)
            print(f"\n[Cleanup] Đã xóa collection tạm '{benchmark_collection}' khỏi Milvus.")

    win_vs_rrf = _compute_win_loss(
        per_query_records, eval_queries, "hybrid_rerank", "hybrid_rrf", top_k=top_k
    )
    win_vs_vec = _compute_win_loss(
        per_query_records, eval_queries, "hybrid_rerank", "vector_only", top_k=top_k
    )

    out_report = Path(
        report_path or os.path.join(os.path.dirname(__file__), "ablation_report.md")
    )
    _write_markdown_report(
        report_path=out_report,
        dataset_name=dataset_name,
        split=split,
        corpus_size=len(corpus_docs),
        query_count=len(eval_queries),
        total_qrels=total_qrels,
        top_k=top_k,
        initial_k=initial_k,
        max_queries=max_queries,
        backend_label=backend_label,
        reranker_model=reranker_service.model_name,
        summary_metrics=summary_metrics,
        win_vs_rrf=win_vs_rrf,
        win_vs_vec=win_vs_vec,
        per_query_records=per_query_records,
        eval_queries=eval_queries,
    )
    print(f"\n[Done] Đã xuất báo cáo Ablation Study chi tiết tại: {out_report}")
    return summary_metrics


def _write_markdown_report(
    report_path: Path,
    dataset_name: str,
    split: str,
    corpus_size: int,
    query_count: int,
    total_qrels: int,
    top_k: int,
    initial_k: int,
    max_queries: Optional[int],
    backend_label: str,
    reranker_model: str,
    summary_metrics: Dict[str, Dict[str, Any]],
    win_vs_rrf: Dict[str, int],
    win_vs_vec: Dict[str, int],
    per_query_records: Dict[str, Dict[str, Any]],
    eval_queries: List[Dict[str, Any]],
):
    v = summary_metrics["vector_only"]
    h = summary_metrics["hybrid_rrf"]
    r = summary_metrics["hybrid_rerank"]

    ndcg_gain_vs_vec = round((r["ndcg_at_k"] - v["ndcg_at_k"]) * 100, 2)
    ndcg_gain_vs_rrf = round((r["ndcg_at_k"] - h["ndcg_at_k"]) * 100, 2)
    hit1_gain_vs_rrf = round(r["hit_rate_at_1"] - h["hit_rate_at_1"], 2)

    lines = [
        "# 📊 Báo cáo Thực nghiệm Ablation Study: Kiến trúc Retrieval 2-Stage Hybrid + Reranking",
        "",
        (
            f"> **Bối cảnh thực nghiệm:** Đánh giá định lượng trên bộ dữ liệu chuẩn quốc tế "
            f"**BEIR `{dataset_name}`** (split `{split}`): `{query_count} câu truy vấn` "
            f"(`{total_qrels} cặp ground-truth qrels`) trên toàn bộ kho văn bản `{corpus_size} tài liệu khoa học`. "
            f"Cấu hình truy xuất: `top_k={top_k}` (Giai đoạn 1 thu hồi `initial_k={initial_k}` ứng viên), "
            f"Vector Backend: `{backend_label}`."
        ),
        "",
        "---",
        "",
        "## 1. Bảng So sánh Tổng hợp (Quantitative Benchmark Summary)",
        "",
        (
            f"| Kiến trúc Truy xuất (Retrieval Mode) | Hit@1 (%) | Hit@3 (%) | Hit@{top_k} (%) | "
            f"MRR@{top_k} | nDCG@{top_k} | Recall@{top_k} (%) | Avg Latency (ms) | P95 Latency (ms) |"
        ),
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        (
            f"| **1. Dense Vector Only** (`paraphrase-multilingual-MiniLM-L12-v2`) | "
            f"{v['hit_rate_at_1']:.2f}% | {v['hit_rate_at_3']:.2f}% | {v['hit_rate_at_k']:.2f}% | "
            f"`{v['mrr_at_k']:.4f}` | `{v['ndcg_at_k']:.4f}` | {v['recall_at_k']:.2f}% | "
            f"`{v['avg_latency_ms']} ms` | `{v['p95_latency_ms']} ms` |"
        ),
        (
            f"| **2. Hybrid Search (Dense Vector + BM25 + RRF)** | "
            f"{h['hit_rate_at_1']:.2f}% | {h['hit_rate_at_3']:.2f}% | {h['hit_rate_at_k']:.2f}% | "
            f"`{h['mrr_at_k']:.4f}` | `{h['ndcg_at_k']:.4f}` | {h['recall_at_k']:.2f}% | "
            f"`{h['avg_latency_ms']} ms` | `{h['p95_latency_ms']} ms` |"
        ),
        (
            f"| **3. 2-Stage Hybrid + BGE Cross-Encoder Reranker** (`{reranker_model}`) | "
            f"**{r['hit_rate_at_1']:.2f}%** | **{r['hit_rate_at_3']:.2f}%** | **{r['hit_rate_at_k']:.2f}%** | "
            f"**`{r['mrr_at_k']:.4f}`** | **`{r['ndcg_at_k']:.4f}`** | **{r['recall_at_k']:.2f}%** | "
            f"`{r['avg_latency_ms']} ms` | `{r['p95_latency_ms']} ms` |"
        ),
        "",
        "---",
        "",
        "## 2. Phân tích Thắng / Hòa / Thua của Reranker (Rank Improvement Analysis)",
        "",
        "| So sánh đối chiếu | Cải thiện thứ hạng (Win) | Giữ nguyên (Tie) | Tụt hạng (Loss) |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Hybrid + Reranker** vs **Dense Vector Only** | **{win_vs_vec['wins']}** | {win_vs_vec['ties']} | {win_vs_vec['losses']} |",
        f"| **Hybrid + Reranker** vs **Hybrid RRF** | **{win_vs_rrf['wins']}** | {win_vs_rrf['ties']} | {win_vs_rrf['losses']} |",
        "",
        "### Các ca điển hình Reranker cải thiện thứ hạng (Top Case Studies)",
        "",
        "| Query ID | Câu truy vấn | Vector Only | Hybrid RRF | Hybrid + BGE Rerank | Latency (2-Stage) |",
        "| :---: | :--- | :---: | :---: | :---: | :---: |",
    ]

    miss_val = top_k + 1

    def _gain_score(q_item: Dict[str, Any]) -> Tuple[int, int]:
        rec = per_query_records[q_item["id"]]
        rk_rerank = rec["hybrid_rerank"]["first_rank"] or miss_val
        rk_rrf = rec["hybrid_rrf"]["first_rank"] or miss_val
        rk_vec = rec["vector_only"]["first_rank"] or miss_val
        return ((rk_rrf - rk_rerank) + (rk_vec - rk_rerank), -rk_rerank)

    sorted_cases = sorted(eval_queries, key=_gain_score, reverse=True)
    shown_cases = [
        q
        for q in sorted_cases
        if (per_query_records[q["id"]]["hybrid_rerank"]["first_rank"] or miss_val)
        < max(
            per_query_records[q["id"]]["hybrid_rrf"]["first_rank"] or miss_val,
            per_query_records[q["id"]]["vector_only"]["first_rank"] or miss_val,
        )
    ][:10]

    if not shown_cases:
        shown_cases = eval_queries[:10]

    def _fmt_rank(rk: Optional[int]) -> str:
        if rk is None:
            return "❌ Miss"
        return f"**Rank #{rk}**" if rk == 1 else f"Rank #{rk}"

    for q_item in shown_cases:
        qid = q_item["id"]
        qtext = q_item["question"].replace("|", "\\|")
        if len(qtext) > 100:
            qtext = qtext[:97] + "..."
        rec = per_query_records[qid]
        lines.append(
            f"| `{qid}` | {qtext} | {_fmt_rank(rec['vector_only']['first_rank'])} | "
            f"{_fmt_rank(rec['hybrid_rrf']['first_rank'])} | "
            f"{_fmt_rank(rec['hybrid_rerank']['first_rank'])} | "
            f"`{rec['hybrid_rerank']['latency_ms']} ms` |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 3. Nhận xét Kỹ thuật & Lý giải Kiến trúc (Engineering Takeaways)",
            "",
            "1. **Hạn chế của Bi-Encoder Vector Search đơn lẻ (`vector_only`):**",
            (
                f"   - Trên tập dữ liệu học thuật `{dataset_name}` với `{corpus_size}` tài liệu, "
                f"Bi-Encoder đa ngữ (`paraphrase-multilingual-MiniLM-L12-v2`) đạt `Hit@1 = {v['hit_rate_at_1']:.2f}%` "
                f"và `nDCG@{top_k} = {v['ndcg_at_k']:.4f}` do việc nén toàn bộ đoạn văn thành một vector cố định làm mờ các thuật ngữ chuyên ngành chính xác."
            ),
            "2. **Hiệu quả thu hồi từ Hybrid Search + RRF (`hybrid_rrf`):**",
            (
                f"   - Kết hợp **BM25 Okapi** (khớp chính xác thuật ngữ hiếm/định danh khoa học) và Dense Vector giúp tăng "
                f"`Recall@{top_k}` từ `{v['recall_at_k']:.2f}%` lên `{h['recall_at_k']:.2f}%` và `nDCG@{top_k}` lên `{h['ndcg_at_k']:.4f}`."
            ),
            "3. **Giá trị của BGE Cross-Encoder Reranker ở Giai đoạn 2 (`hybrid_rerank`):**",
            (
                f"   - Cross-Encoder (`{reranker_model}`) đọc đồng thời cặp `(Query, Document)` qua toàn bộ các tầng Self-Attention, "
                f"đẩy `Hit@1` lên **`{r['hit_rate_at_1']:.2f}%`** (tăng `{hit1_gain_vs_rrf:+.2f}` điểm % so với Hybrid RRF) "
                f"và `nDCG@{top_k}` đạt **`{r['ndcg_at_k']:.4f}`** (`{ndcg_gain_vs_rrf:+.2f}` điểm % so với Hybrid RRF, "
                f"`{ndcg_gain_vs_vec:+.2f}` điểm % so với Vector Only)."
            ),
            "",
            "## 4. Hướng dẫn Tái lập Thực nghiệm (Reproducibility)",
            "",
            "```bash",
            "cd backend",
            (
                f"./.venv/bin/python evaluation/ablation_study.py --dataset {dataset_name} "
                f"--split {split} --top-k {top_k}"
                + (f" --max-queries {max_queries}" if max_queries else "")
            ),
            "```",
            "",
        ]
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ablation Study đánh giá Retrieval & Reranker trên bộ dữ liệu chuẩn BEIR/MTEB."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Tên dataset BEIR/MTEB trên HuggingFace Hub (mặc định: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help=f"Split đánh giá trong qrels (mặc định: {DEFAULT_SPLIT}).",
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Đường dẫn thư mục dataset BEIR cục bộ (tùy chọn, thay cho tải từ HuggingFace).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Số kết quả đầu ra Top-K cần đánh giá (mặc định: {DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Giới hạn số lượng câu hỏi đánh giá (mặc định: toàn bộ split).",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "milvus", "memory"],
        default="auto",
        help="Backend VectorStore: auto (ưu tiên Milvus, fallback NumPy), milvus, hoặc memory.",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.path.join(os.path.dirname(__file__), ".cache"),
        help="Thư mục lưu cache embeddings corpus để tăng tốc các lần chạy sau.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Tắt cache embeddings corpus.",
    )
    parser.add_argument(
        "--report-path",
        default=os.path.join(os.path.dirname(__file__), "ablation_report.md"),
        help="Đường dẫn xuất file báo cáo Markdown.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_ablation_benchmark(
        dataset_name=args.dataset,
        split=args.split,
        dataset_dir=args.dataset_dir,
        top_k=args.top_k,
        max_queries=args.max_queries,
        backend_mode=args.backend,
        cache_dir=None if args.no_cache else args.cache_dir,
        report_path=args.report_path,
    )
