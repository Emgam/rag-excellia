import requests
import time
import itertools
import json

BASE_URL = "http://localhost:8000"

# Define the parameters you want to test
# NOTE: Testing chunk_size requires re-ingesting documents, so we test top_k and thresholds first.
PARAM_GRID = {
    "top_k": [3, 5, 7, 10],
    "threshold": [0.10, 0.15, 0.20, 0.25] # We can't change .env dynamically easily, but we can pass top_k
}

# A small representative sample of your queries (mix of simple, medium, complex)
TEST_QUERIES = [
    "What is the seating capacity of the A350?",
    "How do I check in online?",
    "Compare CloudLux and CloudElite",
    "What is the capital of Australia?", # Should refuse
    "What are the excess baggage fees?"
]

def run_grid_search():
    results = []
    
    for top_k in PARAM_GRID["top_k"]:
        passed = 0
        total_time = 0
        
        print(f"\n Testing top_k={top_k} ---")
        for query in TEST_QUERIES:
            try:
                start = time.time()
                # Pass top_k dynamically to your API
                res = requests.post(f"{BASE_URL}/query", json={"question": query, "top_k": top_k}, timeout=120)
                elapsed = time.time() - start
                total_time += elapsed
                
                if res.status_code == 200:
                    data = res.json()
                    # Basic pass condition: got sources and answer
                    if len(data.get("sources", [])) > 0 and "could not find" not in data.get("answer", "").lower():
                        passed += 1
            except Exception as e:
                print(f"Error: {e}")
                
        avg_time = total_time / len(TEST_QUERIES)
        results.append({
            "top_k": top_k,
            "pass_rate": passed / len(TEST_QUERIES),
            "avg_time": avg_time
        })
        print(f"Pass Rate: {passed}/{len(TEST_QUERIES)} | Avg Time: {avg_time:.2f}s")

    # Find the best one
    best = max(results, key=lambda x: x['pass_rate'])
    print("\n=== OPTIMAL PARAMETERS ===")
    print(f"Best top_k: {best['top_k']} with Pass Rate: {best['pass_rate']*100}%")
    return best

if __name__ == "__main__":
    run_grid_search()