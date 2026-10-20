import os
import json
import time
import uuid
from typing import Optional, Dict, Any, List
from pymilvus import MilvusClient, DataType

from app.services.embedding_service import EmbeddingService


class SemanticCacheService:
    """
    Semantic Cache powered by Milvus Vector Search.
    Caches (query_vector, question, answer, sources, thoughts, scope_doc)
    to serve semantically equivalent questions (Cosine Similarity >= 0.95)
    in < 50ms with 0 LLM token cost.
    """

    DEFAULT_THRESHOLD = 0.95

    def __init__(
        self,
        uri: str = None,
        collection_name: str = "semantic_cache",
        embedding_service: Optional[EmbeddingService] = None,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        self.uri = uri or os.getenv("MILVUS_URI", "http://localhost:19530")
        self.collection_name = collection_name
        self.threshold = threshold
        self.client = None
        self.embedding_service = embedding_service
        self._init_client()

    def _get_embedding_service(self) -> EmbeddingService:
        if self.embedding_service is None:
            self.embedding_service = EmbeddingService()
        return self.embedding_service

    def _init_client(self):
        try:
            self.client = MilvusClient(uri=self.uri)
            self._create_collection()
        except Exception as e:
            print(f"[SemanticCache] Warning: Could not connect to Milvus at {self.uri}: {e}")
            self.client = None

    def _create_collection(self):
        if not self.client:
            return
        try:
            if self.client.has_collection(self.collection_name):
                return

            schema = self.client.create_schema(
                auto_id=False,
                enable_dynamic_field=False,
            )
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
            schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=384)
            schema.add_field(field_name="query", datatype=DataType.VARCHAR, max_length=2000)
            schema.add_field(field_name="answer", datatype=DataType.VARCHAR, max_length=30000)
            schema.add_field(field_name="thoughts_json", datatype=DataType.VARCHAR, max_length=20000)
            schema.add_field(field_name="sources_json", datatype=DataType.VARCHAR, max_length=10000)
            schema.add_field(field_name="scope_doc", datatype=DataType.VARCHAR, max_length=500)
            schema.add_field(field_name="created_at", datatype=DataType.VARCHAR, max_length=50)

            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                metric_type="COSINE",
                index_type="FLAT",
                index_name="cache_vector_idx",
            )

            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params,
            )
            self.client.load_collection(collection_name=self.collection_name)
            print(f"[SemanticCache] Initialized collection '{self.collection_name}' with COSINE metric.")
        except Exception as e:
            print(f"[SemanticCache] Error creating collection '{self.collection_name}': {e}")

    def lookup(
        self,
        query: str,
        scope_doc: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Tra cứu câu hỏi trong Semantic Cache.
        Trả về kết quả nếu cosine similarity >= similarity_threshold (mặc định 0.95).
        """
        if not self.client:
            return None

        threshold = similarity_threshold or self.threshold
        clean_query = query.strip()
        if not clean_query or len(clean_query) < 4:
            return None

        try:
            t0 = time.perf_counter()
            query_vector = self._get_embedding_service().embed_text(clean_query)

            filter_expr = None
            if scope_doc:
                clean_scope = scope_doc.strip().replace('"', '\\"')
                filter_expr = f'scope_doc == "{clean_scope}"'

            search_res = self.client.search(
                collection_name=self.collection_name,
                data=[query_vector],
                limit=1,
                filter=filter_expr,
                output_fields=["id", "query", "answer", "thoughts_json", "sources_json", "scope_doc", "created_at"],
                search_params={"metric_type": "COSINE"},
            )

            if not search_res or not search_res[0]:
                return None

            top_hit = search_res[0][0]
            similarity = float(top_hit.get("distance", 0.0))
            if similarity < threshold:
                return None

            entity = top_hit.get("entity", {})
            dur_ms = round((time.perf_counter() - t0) * 1000, 2)

            parsed_thoughts = []
            if entity.get("thoughts_json"):
                try:
                    parsed_thoughts = json.loads(entity["thoughts_json"])
                except Exception:
                    parsed_thoughts = []

            parsed_sources = []
            if entity.get("sources_json"):
                try:
                    parsed_sources = json.loads(entity["sources_json"])
                except Exception:
                    parsed_sources = []

            print(
                f"[SemanticCache] Cache HIT! Query: '{clean_query[:50]}' "
                f"matched '{entity.get('query', '')[:50]}' (Similarity: {similarity:.4f} >= {threshold}) in {dur_ms}ms ⚡"
            )

            return {
                "hit": True,
                "similarity": similarity,
                "latency_ms": dur_ms,
                "cached_query": entity.get("query"),
                "answer": entity.get("answer"),
                "thoughts": parsed_thoughts,
                "sources": parsed_sources,
                "scope_doc": entity.get("scope_doc"),
            }
        except Exception as e:
            print(f"[SemanticCache] Lookup error: {e}")
            return None

    def store(
        self,
        query: str,
        answer: str,
        thoughts: Optional[List[Dict[str, Any]]] = None,
        sources: Optional[List[Dict[str, Any]]] = None,
        scope_doc: Optional[str] = None,
    ) -> bool:
        """
        Lưu cặp (câu hỏi, câu trả lời, nguồn trích dẫn, suy luận) vào Semantic Cache.
        """
        if not self.client:
            return False

        clean_query = query.strip()
        clean_answer = answer.strip()
        if not clean_query or not clean_answer:
            return False

        try:
            cache_id = str(uuid.uuid4())
            query_vector = self._get_embedding_service().embed_text(clean_query)
            thoughts_json = json.dumps(thoughts or [], ensure_ascii=False)
            sources_json = json.dumps(sources or [], ensure_ascii=False)

            entry = {
                "id": cache_id,
                "vector": query_vector,
                "query": clean_query[:1990],
                "answer": clean_answer[:29900],
                "thoughts_json": thoughts_json[:19900],
                "sources_json": sources_json[:9900],
                "scope_doc": (scope_doc or "").strip()[:490],
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            self.client.insert(collection_name=self.collection_name, data=[entry])
            self.client.flush(collection_name=self.collection_name)
            print(f"[SemanticCache] Cached response for query: '{clean_query[:50]}' (ID: {cache_id}).")
            return True
        except Exception as e:
            print(f"[SemanticCache] Store error: {e}")
            return False

    def clear(self) -> bool:
        """Xóa toàn bộ cache khi kho tri thức có cập nhật lớn."""
        if not self.client:
            return False
        try:
            if self.client.has_collection(self.collection_name):
                self.client.drop_collection(self.collection_name)
            self._create_collection()
            print("[SemanticCache] Cache cleared successfully.")
            return True
        except Exception as e:
            print(f"[SemanticCache] Clear cache error: {e}")
            return False


semantic_cache_service = SemanticCacheService()
