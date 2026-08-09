from app.services.vector_store import VectorStore


def main():
    vector_store = VectorStore()

    dimension = 384  # thay bằng dimension embedding thực tế của bạn

    vector_store.create_collection(
        dimension=dimension
    )

    print("Collection created successfully.")


if __name__ == "__main__":
    main()