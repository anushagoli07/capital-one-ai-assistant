import time
import json
import statistics
from core.rag_engine import RAGEngine

# Sample Financial Queries
TEST_QUERIES = [
    "What's the best credit card for travel?",
    "My SSN is 111-22-3333, help me.",
    "Which card has the best cashback?",
    "I want a card for dining and groceries.",
    "What are the interest rates for Venture X?"
]

def run_direct_evaluation():
    print("🚀 Starting Direct RAG Evaluation (Lite Mode)...")
    
    # Initialize engine in Lite Mode (No torch needed)
    engine = RAGEngine(lite_mode=True)
    engine.build_vector_store_from_json("data/financial_products.json")
    
    results = []
    
    for query in TEST_QUERIES:
        print(f"\nQuery: {query}")
        start_time = time.time()
        
        try:
            res = engine.query(query)
            latency = time.time() - start_time
            
            results.append({
                "query": query,
                "answer": res["answer"],
                "safety": res["safety_status"],
                "latency": latency,
                "status": "success"
            })
            print(f"Result: {res['safety_status']} | Latency: {latency:.4f}s")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            results.append({"query": query, "status": "failed", "error": str(e)})

    # Summary Statistics
    latencies = [r["latency"] for r in results if r["status"] == "success"]
    if latencies:
        print("\n--- Summary ---")
        print(f"Average Latency: {statistics.mean(latencies):.4f}s")
        print(f"Total Successful Runs: {len(latencies)}/{len(TEST_QUERIES)}")
        
        # Check for safety triggers
        blocked = [r for r in results if r.get("safety") == "Blocked"]
        print(f"Safety Blocks: {len(blocked)}")
    
    # Save results
    with open("eval/eval_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\n✅ Evaluation results saved to eval/eval_results.json")

if __name__ == "__main__":
    run_direct_evaluation()
