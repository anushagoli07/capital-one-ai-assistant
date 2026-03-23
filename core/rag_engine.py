from inference.vector_store import VectorStore
from inference.llm_engine import LLMEngine
from core.config import settings
import time

class RAGEngine:
    def __init__(self):
        print("Initializing RAG Engine...")

        # Initialize vector store (FAISS)
        self.vector_store = VectorStore()

        # Initialize LLM
        self.llm = LLMEngine()

        print("RAG Engine ready!")

    def build_knowledge_base(self, documents):
        # Build FAISS index from documents
        print("Building knowledge base...")
        self.vector_store.build_index(documents)
        print("Knowledge base ready!")

    def query(self, question):
        print(f"\nProcessing query: {question}")
        start_time = time.time()

        # Step 1: Retrieve relevant documents from FAISS
        print("Step 1: Searching knowledge base...")
        results = self.vector_store.search(
            question,
            top_k=settings.top_k_results
        )

        # Step 2: Build context from retrieved documents
        context = ""
        sources = []
        for r in results:
            context += r["document"]["text"] + "\n\n"
            sources.append(r["document"]["metadata"]["name"])

        # Step 3: Generate answer using LLM
        print("Step 2: Generating answer...")
        llm_result = self.llm.generate(question, context)

        # Step 4: Calculate total latency
        total_latency = (time.time() - start_time) * 1000

        return {
            "question": question,
            "answer": llm_result["answer"],
            "sources": sources,
            "latency_ms": round(total_latency, 2),
            "retrieval_count": len(results)
        }

if __name__ == "__main__":
    from data.data_loader import load_financial_products

    # Load documents
    docs = load_financial_products()

    # Initialize RAG engine
    rag = RAGEngine()
    rag.build_knowledge_base(docs)

    # Test queries
    questions = [
        "What is the best credit card for travel?",
        "I want a card with no annual fee",
        "What savings account has the best interest rate?"
    ]

    for question in questions:
        result = rag.query(question)
        print(f"\nQ: {result['question']}")
        print(f"A: {result['answer']}")
        print(f"Sources: {result['sources']}")
        print(f"Latency: {result['latency_ms']}ms")
        print("-" * 50)
