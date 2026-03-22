import time
import requests
import json
import statistics

# API Settings
API_URL = "http://localhost:8000"

# Sample Financial Queries
TEST_QUERIES = [
    "What was the total revenue in the latest fiscal year?",
    "Compare the net income with the previous year.",
    "What are the major operational risks mentioned?",
    "Calculate the EBITDA growth if possible.",
    "Draft a summary of the CEO's outlook."
]

def run_evaluation():
    print("🚀 Starting Financial RAG Evaluation...")
    results = []
    
    for query in TEST_QUERIES:
        print(f"Querying: {query}")
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{API_URL}/query", 
                json={"question": query, "model_provider": "openai"}
            )
            response.raise_for_status()
            data = response.json()
            
            latency = time.time() - start_time
            results.append({
                "query": query,
                "latency": latency,
                "source_count": len(data.get("sources", [])),
                "status": "success"
            })
            print(f"✅ Success | Latency: {latency:.2f}s")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            results.append({"query": query, "status": "failed", "error": str(e)})

    # Summary Statistics
    latencies = [r["latency"] for r in results if r["status"] == "success"]
    if latencies:
        print("\n--- Summary ---")
        print(f"Average Latency: {statistics.mean(latencies):.2f}s")
        print(f"Max Latency: {max(latencies):.2f}s")
        print(f"Total Successful Runs: {len(latencies)}/{len(TEST_QUERIES)}")
    
    # Save results
    with open("eval/eval_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    run_evaluation()
