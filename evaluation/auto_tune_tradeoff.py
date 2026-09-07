#!/usr/bin/env python
"""
Automatically finds the optimal trade-off between Speed and Quality.
Tests different top_k and LLM max tokens.
Bypasses Redis cache and uses smarter word-overlap matching.
"""

import requests
import time
import pandas as pd
import re
import random

BASE_URL = "http://localhost:8000"

# Define the configurations you want to test
CONFIG_GRID = [
    {"name": "Max Quality (k=5, tok=300)", "top_k": 5, "max_tokens": 300},
    {"name": "Balanced (k=4, tok=200)",   "top_k": 4, "max_tokens": 200},
    {"name": "Fast (k=3, tok=150)",       "top_k": 3, "max_tokens": 150},
    {"name": "Ultra-Fast (k=2, tok=120)", "top_k": 2, "max_tokens": 120},
]

# A representative sample of 10 questions
TEST_QUESTIONS = [
    {"q": "What is UPI?", "a": "Unified Payments Interface"},
    {"q": "What is the seating capacity of the A350?", "a": "320 passengers"},
    {"q": "How do I check in online?", "a": "Web check-in"},
    {"q": "What is the checked baggage allowance for CloudElite?", "a": "32 kg"},
    {"q": "Compare CloudLux and CloudElite", "a": "First Class"},
    {"q": "What are the excess baggage fees?", "a": "pre-purchased"},
    {"q": "What happens if my flight is delayed?", "a": "compensation"},
    {"q": "How to book a flight?", "a": "CloudWay 24"},
    {"q": "What is the quietest cabin?", "a": "A350"},
    {"q": "What is a documentary credit?", "a": "Letter of Credit"}
]

def evaluate_config(config):
    print(f"\n{'='*60}")
    print(f"🧪 Testing Config: {config['name']}")
    print(f"{'='*60}")
    
    results = []
    for item in TEST_QUESTIONS:
        question = item["q"]
        expected_answer = item["a"].lower()
        
        # Add a random number to bypass Redis cache so the pipeline actually runs!
        cache_buster_question = f"{question} {random.randint(1, 99999)}"
        
        payload = {
            "question": cache_buster_question,
            "top_k": config["top_k"],
            "max_tokens": config["max_tokens"],
            "stream": False
        }
        
        start_time = time.time()
        try:
            response = requests.post(f"{BASE_URL}/query", json=payload, timeout=180)
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                raw_answer = data.get("answer", "")
                
                # Clean citations for printing and matching
                generated_clean = re.sub(r'\[\d+\]', '', raw_answer).strip()
                generated = generated_clean.lower()
                
                # SMARTER MATCHING: Check if >50% of expected words are in the answer
                expected_words = set(expected_answer.split())
                generated_words = set(generated.split())
                overlap = len(expected_words.intersection(generated_words))
                is_correct = (overlap / len(expected_words)) >= 0.5 if expected_words else False
                
                is_refusal = "could not find" in generated
                
                # Scoring logic
                if is_correct:
                    quality_score = 1.0
                elif is_refusal:
                    quality_score = 0.5
                else:
                    quality_score = 0.0
                    
                results.append({
                    "time": elapsed,
                    "quality": quality_score,
                    "correct": is_correct,
                    "refusal": is_refusal
                })
                
                # PRINT EXPECTED VS GOT FOR DEBUGGING
                print(f"\n  Q: {question}")
                print(f"  Expected: {expected_answer}")
                print(f"  Got: {generated_clean[:100]}...")
                print(f"  Time: {elapsed:.2f}s | Correct: {'✅' if is_correct else '❌'}")
                
            else:
                print(f"  HTTP Error {response.status_code}")
                results.append({"time": 180.0, "quality": 0.0, "correct": False, "refusal": False})
                
        except Exception as e:
            print(f"  Exception: {e}")
            results.append({"time": 180.0, "quality": 0.0, "correct": False, "refusal": False})
            
    # Calculate averages
    avg_time = sum([r["time"] for r in results]) / len(results)
    avg_quality = sum([r["quality"] for r in results]) / len(results)
    accuracy = sum([1 if r["correct"] else 0 for r in results]) / len(results)
    
    # Trade-off Score: We want high quality and low time.
    tradeoff_score = (avg_quality * 10.0) / avg_time if avg_time > 0 else 0
    
    return {
        "Config": config["name"],
        "Avg_Time_s": round(avg_time, 2),
        "Quality_Score": round(avg_quality, 2),
        "Accuracy_%": round(accuracy * 100, 1),
        "Tradeoff_Score": round(tradeoff_score, 3)
    }

def main():
    all_results = []
    
    for config in CONFIG_GRID:
        res = evaluate_config(config)
        all_results.append(res)
        
    # Create DataFrame for nice formatting
    df = pd.DataFrame(all_results)
    
    print("\n" + "="*80)
    print("📊 FINAL TRADE-OFF ANALYSIS")
    print("="*80)
    print(df.to_string(index=False))
    
    # Find the winner
    best = df.loc[df["Tradeoff_Score"].idxmax()]
    
    print("\n" + "="*80)
    print("🏆 OPTIMAL CONFIGURATION (Best Trade-off)")
    print("="*80)
    print(f"  Config: {best['Config']}")
    print(f"  Speed: {best['Avg_Time_s']}s | Quality: {best['Quality_Score']} | Accuracy: {best['Accuracy_%']}%")
    print(f"  Tradeoff Score: {best['Tradeoff_Score']}")

if __name__ == "__main__":
    main()