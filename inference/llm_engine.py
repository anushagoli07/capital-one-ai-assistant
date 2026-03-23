from transformers import pipeline
from core.config import settings
import time

class LLMEngine:
    def __init__(self):
        print("Loading LLM model...")
        # Load flan-t5-base model
        # pipeline = easy way to use HuggingFace models
        self.pipe = pipeline(
            "text-generation",
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            max_new_tokens=settings.max_new_tokens
        )
        print(f"LLM loaded: {settings.hf_model_name}")

    def generate(self, question, context):
        # Build prompt with question and context
        prompt = f"""You are a helpful Capital One financial advisor.
        Use the following product information to answer the customer question.
        Be specific and helpful.

        Product Information:
        {context}

        Customer Question: {question}

        Answer:"""

        # Track latency
        start_time = time.time()

        # Generate answer
        full_text = self.pipe(prompt)[0]["generated_text"]
# Extract only the answer part after "Answer:"
        result = full_text.split("Answer:")[-1].strip()

        # Calculate latency
        latency = time.time() - start_time

        return {
            "answer": result,
            "latency_ms": round(latency * 1000, 2),
            "model": settings.hf_model_name
        }

if __name__ == "__main__":
    # Test the LLM
    engine = LLMEngine()

    # Sample context from FAISS
    context = """
Product Name: Capital One Venture Rewards Credit Card
Rewards: 2x miles on every purchase
Best For: Travel rewards, frequent travelers
Annual Fee: $95
Benefits: Travel insurance, no foreign transaction fees
    """

    question = "What is the best credit card for travel?"
    print(f"\nQuestion: {question}")

    result = engine.generate(question, context)
    print(f"Answer: {result['answer']}")
    print(f"Latency: {result['latency_ms']}ms")
