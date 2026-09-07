import requests
import time
import re

BASE_URL = "http://localhost:8000"

QUICK_TESTS = [
    {"query": "What is the seating capacity of the A350?", "expect_refusal": False, "expect_sources": True, "type": "Simple"},
    {"query": "How do I check in online?", "expect_refusal": False, "expect_sources": True, "type": "Medium"},
    {"query": "Compare CloudLux and CloudElite", "expect_refusal": False, "expect_sources": True, "type": "Complex"},
    {"query": "What is the capital of Australia?", "expect_refusal": True, "expect_sources": False, "type": "Hallucination"},
    {"query": "A350?", "expect_refusal": False, "expect_sources": True, "type": "Edge/Short"}
]

print("🚀 RUNNING QUICK RAG TEST (5 Queries)\n" + "="*50)

for test in QUICK_TESTS:
    q = test["query"]
    print(f"\n🔹 [{test['type']}] {q}")
    
    try:
        start = time.time()
        res = requests.post(f"{BASE_URL}/query", json={"question": q, "top_k": 3}, timeout=120)
        elapsed = time.time() - start
        
        if res.status_code == 200:
            data = res.json()
            ans = data.get("answer", "")
            sources = data.get("sources", [])
            
            is_refusal = "could not find" in ans.lower()
            has_citations = bool(re.search(r'\[\d+\]', ans))
            has_sources = len(sources) > 0
            
            print(f"   Answer: {ans[:100]}...")
            print(f"   Sources: {len(sources)} | Citations: {'✅' if has_citations else '❌'} | Refusal: {'✅' if is_refusal else '❌'}")
            print(f"   Time: {elapsed:.2f}s")
            
            # Validation Logic
            if test["expect_refusal"]:
                if is_refusal and not has_sources: print("   ✅ PASS (Correctly refused)")
                else: print("   ❌ FAIL (Should have refused)")
            else:
                if not is_refusal and has_sources and has_citations: print("   ✅ PASS (Correctly answered with citations)")
                else: print("   ❌ FAIL (Missing citations, sources, or falsely refused)")
        else:
            print(f"   ❌ HTTP ERROR: {res.status_code}")
            
    except Exception as e:
        print(f"   ❌ EXCEPTION: {e}")

print("\n" + "="*50 + "\n")