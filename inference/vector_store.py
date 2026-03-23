import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from core.config import settings
import pickle
import os

class VectorStore:
    def __init__(self):
        # Load the embedding model
        # This converts text into vectors (numbers)
        print("Loading embedding model...")
        self.embedder = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )
        self.index = None
        self.documents = []
        print("Embedding model loaded!")

    def build_index(self, documents):
        # Step 1: Extract text from documents
        texts = [doc["text"] for doc in documents]
        self.documents = documents

        # Step 2: Convert texts to vectors
        print("Converting documents to vectors...")
        embeddings = self.embedder.encode(
            texts,
            show_progress_bar=True
        )

        # Step 3: Convert to float32 (FAISS requirement)
        embeddings = np.array(
            embeddings, dtype=np.float32
        )

        # Step 4: Create FAISS index
        # dimension = size of each vector
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)

        # Step 5: Add vectors to index
        self.index.add(embeddings)
        print(f"Index built with {self.index.ntotal} vectors!")

    def search(self, query, top_k=3):
        # Convert question to vector
        query_vector = self.embedder.encode([query])
        query_vector = np.array(
            query_vector, dtype=np.float32
        )

        # Search for similar vectors
        distances, indices = self.index.search(
            query_vector, top_k
        )

        # Return matching documents
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                results.append({
                    "document": self.documents[idx],
                    "distance": float(distances[0][i])
                })
        return results

    def save(self, path="models/vector_store"):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(
            self.index,
            f"{path}/index.faiss"
        )
        with open(f"{path}/documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)
        print("Vector store saved!")

    def load(self, path="models/vector_store"):
        self.index = faiss.read_index(
            f"{path}/index.faiss"
        )
        with open(f"{path}/documents.pkl", "rb") as f:
            self.documents = pickle.load(f)
        print("Vector store loaded!")

if __name__ == "__main__":
    from data.data_loader import load_financial_products

    # Load products
    docs = load_financial_products()

    # Build vector store
    store = VectorStore()
    store.build_index(docs)

    # Test search
    query = "What is the best credit card for travel?"
    print(f"\nSearching for: {query}")
    results = store.search(query, top_k=2)

    for r in results:
        print(f"\nFound: {r['document']['metadata']['name']}")
        print(f"Distance: {r['distance']:.4f}")
