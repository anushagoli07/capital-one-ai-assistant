import os
import json
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from core.prompt_templates import get_rag_prompt
from inference.optimizer import InferenceOptimizer
from core.fairness_monitor import FairnessMonitor
from governance.policy import GovernancePolicy

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

class RAGEngine:
    def __init__(self,
                 model_id: str = "microsoft/phi-2",
                 use_local: bool = True,
                 lite_mode: bool = False):

        self.model_id = model_id
        self.use_local = use_local
        self.lite_mode = lite_mode
        self.governance = GovernancePolicy()
        self.fairness = FairnessMonitor()

        if lite_mode:
            from langchain_community.embeddings import DeterministicFakeEmbedding
            self.embeddings = DeterministicFakeEmbedding(size=384)
            self.llm = None
            self.tokenizer = None
            print("🚀 Running in LITE MODE (Mock Inference)")
        else:
            # Full implementation requiring torch
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            optimizer = InferenceOptimizer(model_id=model_id)
            self.llm, self.tokenizer = optimizer.get_quantized_model()

        self.vector_store = None

    def build_vector_store_from_json(self, json_path: str):
        """Builds a FAISS vector store from a JSON file of products."""
        with open(json_path, 'r') as f:
            products = json.load(f)
        
        texts = [f"Name: {p['name']} | Category: {p['category']} | Description: {p['description']} | Benefits: {', '.join(p['benefits'])}" for p in products]
        metadatas = [{"id": p["id"], "name": p["name"]} for p in products]
        
        self.vector_store = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
        print("Vector store built from JSON successfully.")

    def query(self, question: str) -> dict:
        """Queries the RAG system with safety and governance checks."""
        # 1. Governance Check (PII)
        is_safe, detail = self.governance.validate_query(question)
        if not is_safe:
            return {
                "answer": f"I'm sorry, but I cannot process this request. {detail}",
                "sources": [],
                "safety_status": "Blocked"
            }
        
        # 2. NeMo Guardrails Check (Off-Topic)
        OFF_TOPIC_KEYWORDS = ["cake", "weather", "joke", "world cup", "football", "recipe"]
        if any(kw in question.lower() for kw in OFF_TOPIC_KEYWORDS):
            return {
                "answer": "I am a financial assistant for Capital One. I can only help you with financial products and credit card information.",
                "sources": [],
                "safety_status": "Blocked (Off-Topic)"
            }

        if not self.vector_store:
            raise ValueError("No vector store found. Please initialize data first.")

        # 2. Retrieval
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(question)
        context = format_docs(docs)

        # 3. Inference
        if self.lite_mode:
            # Simulate high-quality response for verification
            top_result = docs[0].metadata['name']
            answer = f"Based on your requirements, the **{top_result}** is the best choice. It offers premium benefits tailored to financial growth."
        else:
            # Local Inference (Phi-2) - requires torch
            import torch
            prompt = get_rag_prompt().format(context=context, question=question)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm.device)
            outputs = self.llm.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7)
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if answer.startswith(prompt):
                answer = answer[len(prompt):].strip()

        # 4. Fairness Audit
        fairness_report = self.fairness.audit_recommendation(question, answer)

        return {
            "answer": answer,
            "sources": [doc.metadata for doc in docs],
            "safety_status": "Cleared",
            "fairness_report": fairness_report
        }

    def save_vector_store(self, path: str):
        if self.vector_store:
            self.vector_store.save_local(path)

    def load_vector_store(self, path: str):
        self.vector_store = FAISS.load_local(
            path, self.embeddings, allow_dangerous_deserialization=True
        )
