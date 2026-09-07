#!/usr/bin/env python
"""Direct test of the query pipeline – bypasses FastAPI."""
import sys
import traceback

print("=== Starting direct test ===")

try:
    print("Importing retrieval.pipeline...")
    from retrieval.pipeline import retrieve
    print("  OK")
    
    print("Importing generation.answer_service...")
    from generation.answer_service import answer_question
    print("  OK")
    
    print("Calling retrieve...")
    chunks, confidence, reliable = retrieve("What is CloudWay-24?", top_k=5)
    print(f"  Retrieved {len(chunks)} chunks, reliable={reliable}")
    
    if chunks:
        print(f"  First chunk: {chunks[0].chunk.text[:100]}...")
    
    print("Calling answer_question...")
    result = answer_question("What is CloudWay-24?", top_k=5)
    print(f"  Answer: {result.answer[:200]}...")
    print(f"  Sources: {len(result.sources)}")
    
    print("\n=== Test completed successfully ===")
    
except Exception as e:
    print(f"\n=== ERROR ===")
    print(f"Error: {e}")
    print(f"Type: {type(e).__name__}")
    print("\nFull traceback:")
    traceback.print_exc()