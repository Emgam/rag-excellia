import time
import requests
from retrieval.hybrid import hybrid_search  # Adjust import to your project structure

query = "What is the core function of the system?"

# 1. Measure Retrieval Speed
start_retrieval = time.time()
docs = hybrid_search(query)  # Runs BM25 + Qdrant Dense search
retrieval_time = time.time() - start_retrieval

# 2. Measure Ollama Generation Speed
start_gen = time.time()
response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "qwen2.5:3b-instruct-q4_k_m",
        "prompt": f"Context: {docs}\nQuestion: {query}",
        "stream": False,
        "options": {"num_thread": 4, "num_predict": 100}
    }
)
gen_time = time.time() - start_gen

print(f"⏱️ Retrieval Time: {retrieval_time:.2f}s")
print(f"⏱️ Generation Time: {gen_time:.2f}s")
print(f"⏱️ Total Time: {retrieval_time + gen_time:.2f}s")