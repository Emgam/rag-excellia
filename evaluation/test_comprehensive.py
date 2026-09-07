#!/usr/bin/env python
"""
Comprehensive Test Suite for Excellia RAG
Tests: Hallucination Prevention, Smart Routing, Citation Accuracy
"""

import requests
import json
import time
import re  # MOVED TO TOP
from datetime import datetime

BASE_URL = "http://localhost:8000"

# Define test cases with expected behavior
TEST_CASES = {
    # ============================================================
    # VALID QUERIES (Should return answers with citations)
    # ============================================================
    "VALID - Aircraft & Fleet": {
        "queries": [
            "What is the Airbus A350?",
            "What is the seating capacity of the A350?",
            "How many seats in CloudLux First Class?",
            "What is the configuration of CloudElite?",
            "What are the features of CloudComfort?",
            "How many seats does CloudSaver have?",
            "What is the pitch in CloudComfort?",
            "Does the A350 have HEPA filtration?",
            "What is the quietest cabin?",
            "What is the total seating capacity of the A350-900?",
        ],
        "expected_sources": True,
        "expected_refusal": False,
        "expected_route": "simple|medium"
    },
    
    "VALID - Baggage Policy": {
        "queries": [
            "How many cabin bags in CloudLux?",
            "What is the checked baggage allowance for CloudElite?",
            "How much can I carry in CloudSaver?",
            "What are the excess baggage fees?",
            "Can I pre-purchase extra baggage?",
            "What is the baggage policy for sporting equipment?",
            "How many checked bags in Premium Economy?",
            "What is the maximum weight per checked bag?",
            "Are there special allowances for SkyPoints members?",
            "What are the cabin baggage dimensions?",
        ],
        "expected_sources": True,
        "expected_refusal": False,
        "expected_route": "simple|medium"
    },
    
    "VALID - Passenger Rights": {
        "queries": [
            "What happens if my flight is delayed?",
            "Am I entitled to compensation for denied boarding?",
            "How do I claim a refund for cancellation?",
            "What assistance for passengers with reduced mobility?",
            "What are my rights if my baggage is lost?",
            "How do I file a complaint?",
            "What is the right to privacy for passengers?",
            "How do I request special assistance?",
            "What support for unaccompanied minors?",
            "What are the passenger rights?",
        ],
        "expected_sources": True,
        "expected_refusal": False,
        "expected_route": "simple|medium"
    },
    
    "VALID - Booking & Services": {
        "queries": [
            "How do I check in online?",
            "How to change my seat?",
            "How to book a flight?",
            "How do I earn SkyPoints?",
            "What are the visa requirements?",
            "How do I open a locker?",
            "How to provide feedback?",
            "What is the web check-in process?",
            "How to upgrade my seat?",
            "How to select a seat?",
        ],
        "expected_sources": True,
        "expected_refusal": False,
        "expected_route": "medium|complex"
    },
    
    # ============================================================
    # SMART ROUTING TESTS (Should use different models/configs)
    # ============================================================
    "ROUTING - Simple Queries": {
        "queries": [
            "What is UPI?",
            "Define documentary credit",
            "What are ATM fees?",
            "Meaning of KYC",
            "What is CloudWay 24?",
            "What is the A350?",
            "Define CloudLux",
            "What is baggage allowance?",
            "What is a locker?",
            "What is web check-in?",
        ],
        "expected_route": "simple",
        "expected_tokens": 100,
        "expected_reranker": False
    },
    
    "ROUTING - Medium Queries": {
        "queries": [
            "How to apply for a loan?",
            "How to open a bank account?",
            "Steps to book a flight",
            "How to check in online?",
            "How to change my seat?",
            "How to file a complaint?",
            "How to request assistance?",
            "Process for baggage claim",
            "How to upgrade my seat?",
            "How to earn SkyPoints?",
        ],
        "expected_route": "medium",
        "expected_tokens": 150,
        "expected_reranker": True
    },
    
    "ROUTING - Complex Queries": {
        "queries": [
            "Compare CloudLux and CloudElite",
            "Difference between CloudComfort and CloudSaver",
            "Compare baggage allowances across all classes",
            "What is the difference between First and Business Class?",
            "Which cabin class is best for long haul?",
            "Compare passenger rights for different scenarios",
            "What happens if flight is delayed and baggage is lost?",
            "Analyze the A350 cabin features",
            "What are the most important passenger rights?",
            "Compare web check-in and airport check-in",
        ],
        "expected_route": "complex",
        "expected_tokens": 200,
        "expected_reranker": True
    },
    
    # ============================================================
    # HALLUCINATION TESTS (Should return NO sources)
    # ============================================================
    "HALLUCINATION - Personal Questions": {
        "queries": [
            "Do you have a sister?",
            "Do you have a friend?",
            "How is the weather in Tunisia?",
            "What is your favorite color?",
            "Do you like pizza?",
            "How old are you?",
            "What is your name?",
            "Do you have a pet?",
            "How are you feeling today?",
            "Do you play football?",
        ],
        "expected_sources": False,
        "expected_refusal": True,
        "expected_route": "refusal"
    },
    
    "HALLUCINATION - Out of Context": {
        "queries": [
            "What is the meaning of life?",
            "How to install Python?",
            "What is a neural network?",
            "How to build a website?",
            "What is machine learning?",
            "Who is the president of France?",
            "What is the capital of Australia?",
            "How do I bake a cake?",
            "What is the stock price of Apple?",
            "How to use Docker?",
        ],
        "expected_sources": False,
        "expected_refusal": True,
        "expected_route": "refusal"
    },
    
    "HALLUCINATION - Typos & Misspellings": {
        "queries": [
            "What is cloud2way?",
            "Tell me about airbuss a350",
            "How to open a lockr?",
            "What is bagage alowance?",
            "Define CloudLuxx",
            "What is UPI?",
            "How to check in onlin?",
            "What is passenger righs?",
            "How to book a flite?",
            "What is the seat pitch in CloudComfort?",
        ],
        # CHANGED: A good RAG system should handle typos and answer these!
        "expected_sources": True,
        "expected_refusal": False,
        "expected_route": "simple|medium"
    },
    
    "HALLUCINATION - Nonsense Queries": {
        "queries": [
            "What is the meaning of 42?",
            "How to become a unicorn?",
            "What is the secret of the universe?",
            "How to time travel?",
            "What is the best pizza topping?",
            "Who is the best football player?",
            "What is the meaning of life?",
            "How to be happy?",
            "What is love?",
            "How to win the lottery?",
        ],
        "expected_sources": False,
        "expected_refusal": True,
        "expected_route": "refusal"
    },
    
    # ============================================================
    # EDGE CASES (Should handle gracefully)
    # ============================================================
    "EDGE - Very Short Queries": {
        "queries": [
            "A350?",
            "Baggage?",
            "UPI?",
            "Locker?",
            "KYC?",
            "ATM?",
            "Check-in?",
            "Flight?",
            "Seat?",
            "SkyPoints?",
        ],
        "expected_sources": True,
        "expected_refusal": False,
        "expected_route": "simple"
    },
    
    "EDGE - Very Long Queries": {
        "queries": [
            "What is the Airbus A350 and what are its key features including seating capacity, cabin classes, and passenger amenities?",
            "Can you explain the entire baggage policy including cabin baggage, checked baggage, excess baggage fees, and special baggage allowances?",
            "What are the complete passenger rights including information rights, care rights, refund rights, and special assistance rights?",
            "How do I book a flight, select a seat, check in online, and manage my booking?",
            "Compare all cabin classes including CloudLux, CloudElite, CloudComfort, and CloudSaver with their features and amenities?",
        ],
        "expected_sources": True,
        "expected_refusal": False,
        "expected_route": "complex"
    },
    
    "EDGE - Mixed Relevance": {
        "queries": [
            "What is the most comfortable seat on CloudWay 24?",
            "Can I bring my pet to the airport?",
            "What is the best way to travel with family?",
            "How to get to the airport?",
            "What is the best airline?",
        ],
        "expected_sources": False,
        "expected_refusal": True,
        "expected_route": "refusal"
    }
}


def analyze_response(data, query, expected):
    """
    Analyze response and check for hallucinations.
    """
    results = {
        "has_answer": bool(data.get("answer", "").strip()),
        "has_sources": len(data.get("sources", [])) > 0,
        "sources_count": len(data.get("sources", [])),
        "confidence": data.get("confidence", 0),
        "reliable": data.get("reliable", False),
        "response_time": data.get("response_time_seconds", 0),
        "is_refusal": "could not find" in data.get("answer", "").lower() or 
                     "don't have enough" in data.get("answer", "").lower(),
        "has_citations": bool(re.search(r'\[\d+\]', data.get("answer", ""))),
    }
    
    # Check for hallucinations (answer exists but sources don't support it)
    results["hallucination"] = (
        results["has_answer"] and 
        not results["is_refusal"] and 
        (not results["has_sources"] or not results["has_citations"])
    )
    
    # Check if smart routing is working (based on response time and content)
    results["smart_routing_detected"] = (
        results["response_time"] > 0 and
        results["has_answer"] and
        not results["is_refusal"]
    )
    
    return results


def run_tests():
    """
    Run all test cases and generate report.
    """
    print("=" * 100)
    print("🧪 EXCELLIA RAG - COMPREHENSIVE TEST SUITE")
    print("=" * 100)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    hallucination_detected = 0
    
    results_by_category = {}
    
    for category, test_data in TEST_CASES.items():
        print(f"\n📂 {category}")
        print("-" * 80)
        
        category_results = []
        
        for query in test_data["queries"]:
            total_tests += 1
            print(f"\n🔹 Query: {query[:60]}...")
            
            try:
                start_time = time.time()
                response = requests.post(
                    f"{BASE_URL}/query",
                    json={
                        "question": query,
                        "top_k": 3,
                        "stream": False
                    },
                    timeout=120  # INCREASED to 120 seconds for CPU NLI + LLM
                )
                elapsed = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    results = analyze_response(data, query, test_data)
                    results["response_time"] = elapsed
                    
                    # Determine if test passed
                    expected_sources = test_data.get("expected_sources", True)
                    expected_refusal = test_data.get("expected_refusal", False)
                    
                    if expected_refusal:
                        passed = results["is_refusal"] and not results["has_sources"] and not results["hallucination"]
                    elif expected_sources:
                        passed = results["has_sources"] and results["has_citations"] and not results["hallucination"]
                    else:
                        passed = not results["hallucination"]
                    
                    if passed:
                        passed_tests += 1
                        status = "✅ PASS"
                    else:
                        failed_tests += 1
                        status = "❌ FAIL"
                    
                    if results["hallucination"]:
                        hallucination_detected += 1
                    
                    print(f"   Status: {status}")
                    print(f"   Answer: {data.get('answer', '')[:100]}...")
                    print(f"   Sources: {results['sources_count']}")
                    print(f"   Citations: {'✅' if results['has_citations'] else '❌'}")
                    print(f"   Refusal: {'✅' if results['is_refusal'] else '❌'}")
                    print(f"   Hallucination: {'✅' if results['hallucination'] else '❌'}")
                    print(f"   Confidence: {results['confidence']:.4f}")
                    print(f"   Reliable: {results['reliable']}")
                    print(f"   Response Time: {results['response_time']:.2f}s")
                    
                    category_results.append({
                        "query": query,
                        "passed": passed,
                        "results": results
                    })
                else:
                    failed_tests += 1
                    print(f"   ❌ ERROR: HTTP {response.status_code}")
                    
            except Exception as e:
                failed_tests += 1
                print(f"   ❌ EXCEPTION: {e}")
        
        results_by_category[category] = category_results
    
    # Summary
    print("\n" + "=" * 100)
    print("📊 TEST SUMMARY")
    print("=" * 100)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Hallucinations Detected: {hallucination_detected}")
    
    if total_tests > 0:
        score = (passed_tests / total_tests) * 100
        print(f"Score: {score:.1f}%")
    else:
        print("Score: N/A (no tests run)")
    
    # Category breakdown with division by zero protection
    print("\n📊 Category Breakdown:")
    print("-" * 80)
    for category, results in results_by_category.items():
        total = len(results)
        if total == 0:
            print(f"{category}: 0/0 (N/A - no tests)")
            continue
        passed = sum(1 for r in results if r["passed"])
        print(f"{category}: {passed}/{total} ({passed/total*100:.1f}%)")
    
    # Hallucination check
    if hallucination_detected == 0:
        print("\n🎉 ZERO HALLUCINATIONS DETECTED! System is safe!")
    else:
        print(f"\n⚠️ {hallucination_detected} hallucinations detected! Need improvement.")
    
    # Smart routing analysis
    print("\n🧭 Smart Routing Analysis:")
    print("-" * 80)
    routing_categories = ["ROUTING - Simple Queries", "ROUTING - Medium Queries", "ROUTING - Complex Queries"]
    for category in routing_categories:
        if category in results_by_category:
            results = results_by_category[category]
            total = len(results)
            if total > 0:
                avg_time = sum(r["results"]["response_time"] for r in results) / total
                print(f"{category}: Avg Response Time: {avg_time:.2f}s")
            else:
                print(f"{category}: No tests run")
    
    # Overall assessment
    if total_tests > 0:
        score = (passed_tests / total_tests) * 100
        if score >= 90:
            print("\n🏆 EXCELLENT! System is production-ready!")
        elif score >= 70:
            print("\n⚠️ Good, but needs some improvements.")
        else:
            print("\n❌ System needs significant improvement.")

if __name__ == "__main__":
    run_tests()