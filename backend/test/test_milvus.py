from pymilvus import MilvusClient


URI = "http://localhost:19530"
COLLECTION_NAME = "document_chunks"


def main():
    client = MilvusClient(uri=URI)

    print("Connected to Milvus.")

    exists = client.has_collection(
        collection_name=COLLECTION_NAME
    )

    print(f"Collection '{COLLECTION_NAME}' exists: {exists}")


if __name__ == "__main__":
    main()