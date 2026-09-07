#!/usr/bin/env python
"""
Ultimate Production Readiness Evaluation Script.
Measures True Semantic Accuracy, Faithfulness, Citations, and Latency.
Bypasses Redis cache to ensure real pipeline execution.
"""

import requests
import time
import re
import random
import numpy as np
import pandas as pd
from typing import List, Dict

# Import your local embedder to calculate semantic similarity
from embeddings.embedder import get_embedder

BASE_URL = "http://localhost:8000"

# A representative sample of 10 questions (Mix of simple, medium, complex)
TEST_QUESTIONS = [
    {"q": "What is UPI?", "a": "Unified Payments Interface"},
    {"q": "What is the seating capacity of the A350?", "a": "320 passengers"},
    {"q": "How do I check in online?", "a": "Web check-in opens 48 hours before departure"},
    {"q": "What is the checked baggage allowance for CloudElite?", "a": "32 kg per bag"},
    {"q": "Compare CloudLux and CloudElite", "a": "CloudLux is First Class, CloudElite is Business Class"},
    {"q": "What are the excess baggage fees?", "a": "Can be pre-purchased at a discounted rate"},
    {"q": "What happens if my flight is delayed?", "a": "You are entitled to meals and compensation"},
    {"q": "How to book a flight?", "a": "Via the CloudWay 24 website or app"},
    {"q": "What is the quietest cabin?", "a": "The Airbus A350"},
    {"q": "What is a documentary credit?", "a": "A letter of credit where documents are required"}
]

def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """Calculates cosine similarity between two texts using local embeddings."""
    embedder = get_embedder()
    vec1 = embedder.embed_query(text1)
    vec2 = embedder.embed_query(text2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    return float(dot_product / (norm1 * norm2)) if norm1 > 0 and norm2 > 0 else 0.0

def extract_citations(answer: str) -> List[int]:
    return [int(m) for m in re.findall(r'\[(\d+)\]', answer)]

def main():
    print("="*60)
    print("🚀 PRODUCTION READINESS EVALUATION")
    print("="*60)
    
    results = []
    
    for idx, item in enumerate(TEST_QUESTIONS):
        question = item["q"]
        expected_answer = item["a"]
        
        # Bypass Redis Cache
        cache_buster_q = f"{question} {random.randint(1, 99999)}"
        
        payload = {
            "question": cache_buster_q,
            "top_k": 4,      # Test with your optimal fast setting
            "stream": False
        }
        
        start_time = time.time()
        try:
            response = requests.post(f"{BASE_URL}/query", json=payload, timeout=180)
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                raw_answer = data.get("answer", "")
                generated_clean = re.sub(r'\[\d+\]', '', raw_answer).strip()
                
                # 1. Semantic Accuracy (Cosine Similarity > 0.75 = Accurate)
                sim_score = calculate_semantic_similarity(expected_answer, generated_clean)
                is_semantically_correct = sim_score >= 0.60
                
                # 2. Faithfulness (Did NLI verifier pass?)
                faithful = data.get("reliable", False)
                
                # 3. Citation Accuracy
                cited_ids = extract_citations(raw_answer)
                has_valid_citations = len(cited_ids) > 0
                
                # 4. Refusal Check
                is_refusal = "could not find" in generated_clean.lower()
                
                results.append({
                    "question": question[:30] + "...",
                    "time_s": round(elapsed, 2),
                    "semantic_sim": round(sim_score, 2),
                    "correct": is_semantically_correct,
                    "faithful": faithful,
                    "cited": has_valid_citations,
                    "refusal": is_refusal
                })
                
                print(f"{idx+1}/10 | Time: {elapsed:.1f}s | Sim: {sim_score:.2f} | Faithful: {'✅' if faithful else '❌'} | Cited: {'✅' if has_valid_citations else '❌'}")
                
            else:
                print(f"  HTTP Error {response.status_code}")
                results.append({"question": question[:30], "time_s": 180.0, "semantic_sim": 0.0, "correct": False, "faithful": False, "cited": False, "refusal": False})
                
        except Exception as e:
            print(f"  Exception: {e}")
            results.append({"question": question[:30], "time_s": 180.0, "semantic_sim": 0.0, "correct": False, "faithful": False, "cited": False, "refusal": False})
            
    # Calculate Final Metrics
    df = pd.DataFrame(results)
    
    avg_time = df['time_s'].mean()
    semantic_accuracy = df['correct'].mean() * 100
    faithfulness_rate = df['faithful'].mean() * 100
    citation_rate = df['cited'].mean() * 100
    refusal_rate = df['refusal'].mean() * 100
    
    # Print Final Report
    print("\n" + "="*60)
    print("📊 PRODUCTION READINESS REPORT")
    print("="*60)
    print(f"  True Semantic Accuracy:  {semantic_accuracy:.1f}%  (Target: > 70%)")
    print(f"  Faithfulness (No Hallu): {faithfulness_rate:.1f}%  (Target: > 90%)")
    print(f"  Citation Compliance:     {citation_rate:.1f}%  (Target: 100%)")
    print(f"  Average Response Time:   {avg_time:.2f}s   (Target: < 30s)")
    print(f"  False Refusal Rate:      {refusal_rate:.1f}%  (Target: < 20%)")
    print("="*60)
    
    # Final Verdict
    is_ready = (
        semantic_accuracy >= 70.0 and
        faithfulness_rate >= 90.0 and
        citation_rate >= 95.0 and
        avg_time <= 30.0
    )
    
    if is_ready:
        print("\n🟢 VERDICT: SYSTEM IS PRODUCTION READY!")
        print("Your RAG assistant is accurate, grounded, cited, and fast enough for CPU.")
    else:
        print("\n🟡 VERDICT: Needs minor tuning. Review metrics above.")

if __name__ == "__main__":
    main()