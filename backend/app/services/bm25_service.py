import math
import re
from typing import List, Dict, Any, Optional


class SimpleBM25Okapi:
    """
    Thuật toán BM25Okapi thuần Python để đảm bảo hệ thống luôn chạy ổn định
    ngay cả khi chưa cài đặt thư viện ngoài (rank-bm25).
    """

    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.corpus_size = len(corpus)
        self.avgdl = sum(float(len(x)) for x in corpus) / self.corpus_size if self.corpus_size > 0 else 0.0
        self.doc_lens = [len(x) for x in corpus]
        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}
        self.nd: Dict[str, int] = {}
        self._initialize(corpus)

    def _initialize(self, corpus: List[List[str]]):
        for document in corpus:
            frequencies: Dict[str, int] = {}
            for word in document:
                frequencies[word] = frequencies.get(word, 0) + 1
            self.doc_freqs.append(frequencies)

            for word in frequencies:
                self.nd[word] = self.nd.get(word, 0) + 1

        negative_idfs = []
        for word, freq in self.nd.items():
            idf = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)
            self.idf[word] = idf
            if idf < 0:
                negative_idfs.append(word)

        self.average_idf = sum(self.idf.values()) / len(self.idf) if self.idf else 0.0
        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query: List[str]) -> List[float]:
        score = [0.0] * self.corpus_size
        doc_len = self.doc_lens
        for q in query:
            q_freq = self.nd.get(q, 0)
            if q_freq == 0:
                continue
            q_idf = self.idf.get(q, 0.0)
            for i, doc_freq in enumerate(self.doc_freqs):
                freq = doc_freq.get(q, 0)
                if freq == 0:
                    continue
                score[i] += (
                    q_idf
                    * (freq * (self.k1 + 1.0))
                    / (freq + self.k1 * (1.0 - self.b + self.b * (doc_len[i] / (self.avgdl or 1.0))))
                )
        return score


class BM25Service:
    """
    Service quản lý chỉ mục Lexical Search (BM25) cho các đoạn văn bản (chunks).
    """

    def __init__(self):
        self.documents: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25_model: Optional[SimpleBM25Okapi] = None

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tách từ hỗ trợ cả tiếng Anh, tiếng Việt và ký tự số."""
        if not text:
            return []
        return re.findall(r"[\w]+", text.lower())

    def index_documents(self, documents: List[Dict[str, Any]]):
        """Khởi tạo toàn bộ chỉ mục BM25 từ danh sách documents."""
        self.documents = list(documents)
        self.tokenized_corpus = [
            self.tokenize(doc.get("text", "")) for doc in self.documents
        ]
        if self.tokenized_corpus:
            self.bm25_model = SimpleBM25Okapi(self.tokenized_corpus)
            print(f"[BM25Service] Indexed {len(self.documents)} chunks into BM25 index.")
        else:
            self.bm25_model = None

    def add_documents(self, new_documents: List[Dict[str, Any]]):
        """Bổ sung thêm chunks mới (ví dụ khi upload tài liệu mới)."""
        if not new_documents:
            return
        self.documents.extend(new_documents)
        self.tokenized_corpus.extend(
            [self.tokenize(doc.get("text", "")) for doc in new_documents]
        )
        if self.tokenized_corpus:
            self.bm25_model = SimpleBM25Okapi(self.tokenized_corpus)
            print(f"[BM25Service] Added {len(new_documents)} chunks. Total BM25 corpus: {len(self.documents)} chunks.")

    def remove_documents_by_id(self, document_id: str):
        """Xóa toàn bộ các chunk thuộc về document_id được chỉ định."""
        self.documents = [
            doc for doc in self.documents if doc.get("document_id") != document_id
        ]
        self.tokenized_corpus = [
            self.tokenize(doc.get("text", "")) for doc in self.documents
        ]
        if self.tokenized_corpus:
            self.bm25_model = SimpleBM25Okapi(self.tokenized_corpus)
        else:
            self.bm25_model = None
        print(f"[BM25Service] Removed chunks for doc_id='{document_id}'. Remaining chunks: {len(self.documents)}.")

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Tìm kiếm BM25 theo query và trả về Top K ứng viên có format tương thích với Milvus.
        """
        if not self.bm25_model or not self.documents:
            return []

        tokens = self.tokenize(query)
        if not tokens:
            return []

        scores = self.bm25_model.get_scores(tokens)

        # Lọc các kết quả có điểm > 0 và sắp xếp giảm dần
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        results = []
        for idx in ranked_indices:
            score = scores[idx]
            if score <= 0.0:
                break
            doc = self.documents[idx]
            results.append({
                "id": doc.get("id"),
                "distance": float(score),
                "score": float(score),
                "entity": {
                    "text": doc.get("text", ""),
                    "document_id": doc.get("document_id", ""),
                    "document_name": doc.get("document_name", ""),
                    "page": doc.get("page", 1),
                    "chunk_index": doc.get("chunk_index", 0),
                },
            })
            if len(results) >= top_k:
                break

        return results

    def sync_from_milvus(self, vector_store):
        """
        Đồng bộ dữ liệu hiện có trong Milvus vào BM25 khi ứng dụng khởi động.
        """
        try:
            if not hasattr(vector_store, "client") or not vector_store.client.has_collection(vector_store.collection_name):
                return
            
            # Lấy toàn bộ chunk từ collection hiện có (tối đa 10000 chunks)
            res = vector_store.client.query(
                collection_name=vector_store.collection_name,
                filter="id != ''",
                output_fields=["id", "text", "document_id", "document_name", "page", "chunk_index"],
                limit=10000,
            )
            if res:
                self.index_documents(res)
                print(f"[BM25Service] Synced {len(res)} chunks from Milvus to BM25 index.")
        except Exception as e:
            print(f"[BM25Service] Warning during sync from Milvus: {e}")


# Singleton instance dùng chung trong toàn bộ app
bm25_service = BM25Service()
