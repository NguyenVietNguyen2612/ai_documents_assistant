from pymilvus import (
    MilvusClient,
    DataType,
)


class VectorStore:
    def __init__(
        self,
        uri: str = "http://localhost:19530",
        collection_name: str = "document_chunks",
    ):
        self.client = MilvusClient(uri=uri)
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
        limit: int = 5,
    ):
        return self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=limit,
            output_fields=[
                "text",
                "document_id",
                "document_name",
                "page",
                "chunk_index",
            ],
        )
    
    def get_by_id(self, record_id: str):
        return self.client.get(
            collection_name=self.collection_name,
            ids=[record_id],
        )

    