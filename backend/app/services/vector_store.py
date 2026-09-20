import os
from pymilvus import (
    MilvusClient,
    DataType,
)


class VectorStore:
    def __init__(
        self,
        uri: str = None,
        collection_name: str = "document_chunks",
    ):
        self.uri = uri or os.getenv("MILVUS_URI", "http://localhost:19530")
        self.client = MilvusClient(uri=self.uri)
        self.collection_name = collection_name

    def create_collection(self, dimension: int):
        if self.client.has_collection(
            collection_name=self.collection_name
        ):
            return

        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
        )

        schema.add_field(
            field_name="id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=100,
        )

        schema.add_field(
            field_name="vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=dimension,
        )

        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=10000,
        )

        schema.add_field(
            field_name="document_id",
            datatype=DataType.VARCHAR,
            max_length=100,
        )

        schema.add_field(
            field_name="document_name",
            datatype=DataType.VARCHAR,
            max_length=500,
        )

        schema.add_field(
            field_name="page",
            datatype=DataType.INT64,
        )

        schema.add_field(
            field_name="chunk_index",
            datatype=DataType.INT64,
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
        )

    def create_index(self):
        index_params = self.client.prepare_index_params()

        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )

        self.client.create_index(
            collection_name=self.collection_name,
            index_params=index_params,
        )

    def load_collection(self):
        self.client.load_collection(
            collection_name=self.collection_name
        )
    
    def insert(self, data):
        result = self.client.insert(
            collection_name=self.collection_name,
            data=data,
        )

        self.client.flush(
            collection_name=self.collection_name
        )

        return result

    def search(
        self,
        query_vector: list[float],
        limit: int = 15,
        document_name: str = None,
        document_ids: list[str] = None,
    ):
        search_params = {
            "collection_name": self.collection_name,
            "data": [query_vector],
            "limit": limit,
            "output_fields": [
                "text",
                "document_id",
                "document_name",
                "page",
                "chunk_index",
            ],
        }
        
        # 1. Ưu tiên lọc theo document_ids (chính xác và không phụ thuộc hoa/thường)
        if document_ids and len(document_ids) > 0:
            if len(document_ids) == 1:
                search_params["filter"] = f"document_id == '{document_ids[0]}'"
            else:
                ids_str = ", ".join([f"'{d_id}'" for d_id in document_ids])
                search_params["filter"] = f"document_id in [{ids_str}]"
        # 2. Fallback nếu truyền chuỗi document_name trực tiếp
        elif document_name:
            doc_str = str(document_name).strip()
            # Nới lỏng tìm kiếm cả chữ thường lẫn chữ hoa
            search_params["filter"] = f"document_name like '%{doc_str}%' or document_name like '%{doc_str.lower()}%' or document_name like '%{doc_str.upper()}%'"
            
        return self.client.search(**search_params)
    
    def get_by_id(self, record_id: str):
        return self.client.get(
            collection_name=self.collection_name,
            ids=[record_id],
        )

    def delete_by_document_id(self, document_id: str):
        """
        Deletes all chunks in the collection that belong to the specified document_id.
        """
        return self.client.delete(
            collection_name=self.collection_name,
            filter=f"document_id == '{document_id}'"
        )

    def get_all_document_ids(self) -> list[str]:
        if not self.client.has_collection(self.collection_name):
            return []
        res = self.client.query(
            collection_name=self.collection_name, 
            filter="id != ''", 
            output_fields=["document_id"]
        )
        return list(set(r.get("document_id") for r in res if r.get("document_id")))

    