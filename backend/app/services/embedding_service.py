from sentence_transformers import SentenceTransformer


class EmbeddingService:

    def __init__(self):
        model_name = "paraphrase-multilingual-MiniLM-L12-v2"
        try:
            self.model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            self.model = SentenceTransformer(model_name)

    def embed_text(
        self,
        text: str
    ) -> list[float]:

        embedding = self.model.encode(text)

        return embedding.tolist()

    def embed_chunks(
        self,
        chunks: list[str]
    ) -> list[list[float]]:

        embeddings = self.model.encode(
            chunks
        )

        return embeddings.tolist()