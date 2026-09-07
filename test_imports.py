#!/usr/bin/env python
"""Test imports and pipeline directly."""
import sys
print("=== Starting import test ===")

try:
    print("Importing api.main...")
    from api.main import app
    print("  OK")
    
    print("Importing retrieval.pipeline...")
    from retrieval.pipeline import retrieve
    print("  OK")
    
    print("Importing generation.answer_service...")
    from generation.answer_service import answer_question
    print("  OK")
    
    print("=== All imports successful ===")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()