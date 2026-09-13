from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class Retriever:

    def __init__(self):
        print("Loading retrieval model...")

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.documents = []
        self.index = None

        print("Retrieval model loaded successfully!")

    def add_documents(self, documents):

        self.documents = documents

        embeddings = self.model.encode(
            documents,
            convert_to_numpy=True
        )

        embeddings = embeddings.astype("float32")

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dimension)

        self.index.add(embeddings)

    def search(self, query, top_k=3):

        if self.index is None or not self.documents:
            return []

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True
        )

        query_embedding = query_embedding.astype("float32")

        distances, indices = self.index.search(
            query_embedding,
            min(top_k, len(self.documents))
        )

        results = []

        for distance, index in zip(
            distances[0],
            indices[0]
        ):
            results.append({
                "document": self.documents[index],
                "distance": float(distance)
            })

        return results


if __name__ == "__main__":

    retriever = Retriever()

    documents = [
        "The Federal University of Technology, Akure (FUTA) was established in 1981.",
        "FUTA is located in Akure, Ondo State, Nigeria.",
        "The University of Lagos was established in 1962.",
        "The University of Ibadan was established in 1948."
    ]

    retriever.add_documents(documents)

    question = "When was FUTA established?"

    results = retriever.search(question)

    print("\n==============================")
    print("RETRIEVAL RESULTS")
    print("==============================")

    for result in results:
        print(result["document"])
        print("Distance:", result["distance"])
        print()