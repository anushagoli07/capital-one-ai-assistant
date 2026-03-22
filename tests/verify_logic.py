import os
import json
from governance.policy import GovernancePolicy
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DeterministicFakeEmbedding

# Mock RAG Engine for verification in environments without local Torch/CUDA
class MockRAGEngine:
    def __init__(self):
        self.governance = GovernancePolicy()
        self.embeddings = DeterministicFakeEmbedding(size=384)
        self.vector_store = None

    def build_vector_store_from_json(self, json_path: str):
        with open(json_path, 'r') as f:
            products = json.load(f)
        texts = [f"Name: {p['name']} | Description: {p['description']}" for p in products]
        metadatas = [{"name": p["name"]} for p in products]
        self.vector_store = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)

    def query(self, question: str):
        # 1. Governance check
        is_safe, msg = self.governance.validate_query(question)
        if not is_safe:
            return {"answer": f"Blocked: {msg}", "safety": "Blocked"}
        
        # 2. Search
        docs = self.vector_store.similarity_search(question, k=2)
        sources = [doc.metadata['name'] for doc in docs]
        
        # 3. Mock LLM Response
        mock_answer = f"Based on our products, I recommend {sources[0]} for your needs."
        return {"answer": mock_answer, "sources": sources, "safety": "Cleared"}

def verify_flow():
    print("--- Starting Capital One AI Assistant Logic Verification ---")
    engine = MockRAGEngine()
    engine.build_vector_store_from_json("data/financial_products.json")
    
    # Test 1: On-topic query
    q1 = "What's the best credit card for travel?"
    res1 = engine.query(q1)
    print(f"Query: {q1}\nResponse: {res1['answer']}\nSources: {res1['sources']}\nSafety: {res1['safety']}\n")
    
    # Test 2: PII query
    q2 = "My SSN is 111-22-3333, which card can I get?"
    res2 = engine.query(q2)
    print(f"Query: {q2}\nResponse: {res2['answer']}\nSafety: {res2['safety']}\n")

if __name__ == "__main__":
    verify_flow()
