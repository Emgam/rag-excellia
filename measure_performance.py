#!/usr/bin/env python
"""
Measure performance of RAG system
"""

import requests
import time
import json
import sys

def measure_query(question="What is UPI?", top_k=3):
    """Measure performance of a single query"""
    
    print("=" * 70)
    print(f"🔬 Measuring Performance: '{question}'")
    print("=" * 70)
    
    # Start timer
    start = time.time()
    
    try:
        response = requests.post(
            "http://localhost:8000/query",
            json={"question": question, "top_k": top_k, "stream": False},
            timeout=60
        )
        elapsed = time.time() - start
        
        if response.status_code != 200:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            return
        
        data = response.json()
        
        print("\n📊 PERFORMANCE BREAKDOWN")
        print("-" * 50)
        print(f"🔹 Total Time (Client):    {elapsed:.2f}s")
        print(f"🔹 API Reported Time:      {data.get('response_time_seconds', 0):.2f}s")
        print(f"🔹 Confidence Score:       {data.get('confidence', 0):.4f}")
        print(f"🔹 Reliable:               {data.get('reliable', False)}")
        print(f"🔹 Sources Found:          {len(data.get('sources', []))}")
        print("-" * 50)
        
        print("\n📝 Answer Preview:")
        answer = data.get("answer", "")
        if len(answer) > 300:
            print(answer[:300] + "...")
        else:
            print(answer)
        
        print("=" * 70)
        
        return {
            "total_time": elapsed,
            "response_time": data.get("response_time_seconds", 0),
            "confidence": data.get("confidence", 0),
            "reliable": data.get("reliable", False),
            "answer": answer
        }
        
    except requests.exceptions.Timeout:
        print("❌ Request timed out after 60 seconds")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def measure_multiple(questions=None):
    """Measure multiple queries"""
    
    if questions is None:
        questions = [
            "What is UPI?",
            "What is Excellia Trade?",
            "What is a documentary credit?",
            "How do I apply for a cheque book?",
        ]
    
    results = []
    print("\n" + "=" * 70)
    print("📊 BULK PERFORMANCE TEST")
    print("=" * 70)
    
    for i, question in enumerate(questions, 1):
        print(f"\n🔄 Query {i}/{len(questions)}: '{question[:50]}...'")
        result = measure_query(question)
        if result:
            results.append(result)
    
    # Average
    if results:
        avg_time = sum(r["total_time"] for r in results) / len(results)
        avg_conf = sum(r["confidence"] for r in results) / len(results)
        
        print("\n" + "=" * 70)
        print("📊 AVERAGE PERFORMANCE")
        print("=" * 70)
        print(f"📈 Average Response Time: {avg_time:.2f}s")
        print(f"📈 Average Confidence:    {avg_conf:.4f}")
        print(f"✅ All Reliable:          {all(r['reliable'] for r in results)}")
        print("=" * 70)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run with custom question
        question = " ".join(sys.argv[1:])
        measure_query(question)
    else:
        # Run with default question
        measure_query("What is UPI?")  # ← FIXED: default question