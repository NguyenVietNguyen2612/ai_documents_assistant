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
    
    if exists:
        stats = client.get_collection_stats(collection_name=COLLECTION_NAME)
        print("\n--- Collection Stats ---")
        print(f"Total entities (chunks): {stats.get('row_count')}")
        
        # Optional: Retrieve a few sample records
        print("\n--- Sample Data (First 2 chunks) ---")
        res = client.query(
            collection_name=COLLECTION_NAME,
            filter="",  # empty filter to get all, but limit
            limit=2,
            output_fields=["document_name", "chunk_index", "text"]
        )
        for r in res:
            print(f"- Doc: {r.get('document_name')}, Chunk {r.get('chunk_index')}")
            print(f"  Text: {r.get('text')[:100]}...")


if __name__ == "__main__":
    main()