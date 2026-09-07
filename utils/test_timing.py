#!/usr/bin/env python
"""
Test timing without circular imports
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.reranking.reranker import load_reranker
from utils.timer import print_timing_report, reset_timing, timeit
from generation.answer_service import answer_question
print("Warming up models...")
load_reranker()
def main():
    question = "What is UPI?"
    
    print("\n" + "=" * 70)
    print(f"🔬 TESTING: '{question}'")
    print("=" * 70)
    
    reset_timing()
    result = answer_question(question)
    
    print("\n" + "=" * 70)
    print("📝 ANSWER:")
    print("=" * 70)
    print(result.answer)
    print("=" * 70)
    
    print_timing_report()

if __name__ == "__main__":
    main()