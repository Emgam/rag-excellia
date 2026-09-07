# evaluation/analyze_documents.py
"""
Analyze document distribution to validate categories
"""

import os
from pathlib import Path
from collections import defaultdict
import json

def analyze_documents(docs_path="./documents"):
    """Analyze document distribution by category"""
    
    # Define category keywords (based on your actual content)
    category_keywords = {
        'banking': ['bank', 'account', 'savings', 'current', 'deposit', 'branch', 'atm', 'netbanking'],
        'aircraft': ['airbus', 'boeing', 'aircraft', 'a350', 'a380', '787', 'dreamliner', 'fleet'],
        'cabin_features': ['class', 'seat', 'cabin', 'cloudlux', 'cloudelite', 'cloudcomfort', 'cloudsaver'],
        'loans': ['loan', 'credit', 'mortgage', 'finance', 'interest', 'repayment', 'emi'],
        'services': ['baggage', 'boarding', 'check-in', 'lounge', 'meal', 'wifi', 'service'],
        'policies': ['policy', 'terms', 'conditions', 'discount', 'refund', 'cancellation'],
        'digital_banking': ['upi', 'app', 'mobile', 'digital', 'online', 'payment', 'netbanking'],
        'general': []  # Catch-all
    }
    
    file_counts = defaultdict(int)
    category_files = defaultdict(list)
    
    # Walk through documents
    for root, dirs, files in os.walk(docs_path):
        for file in files:
            if file.endswith(('.pdf', '.docx', '.md', '.html', '.pptx')):
                file_path = Path(root) / file
                file_str = str(file_path).lower()
                file_counts['total'] += 1
                
                # Categorize
                categorized = False
                for category, keywords in category_keywords.items():
                    if keywords and any(kw in file_str for kw in keywords):
                        category_files[category].append(file)
                        categorized = True
                        break
                
                if not categorized:
                    category_files['general'].append(file)
    
    return category_files, file_counts

def print_analysis(category_files):
    """Print category analysis"""
    print("\n" + "=" * 70)
    print("📊 DOCUMENT CATEGORY ANALYSIS")
    print("=" * 70)
    
    total = sum(len(files) for files in category_files.values())
    print(f"\n📁 Total Documents: {total}")
    print("\n📂 Distribution by Category:")
    print("-" * 50)
    
    for category, files in sorted(category_files.items(), key=lambda x: len(x[1]), reverse=True):
        count = len(files)
        percentage = (count / total * 100) if total > 0 else 0
        bar = "█" * int(percentage / 2) + "░" * (50 - int(percentage / 2))
        print(f"  {category:20} {count:4} ({percentage:5.1f}%) {bar}")
    
    # Show sample files per category
    print("\n📄 Sample Files Per Category:")
    print("-" * 50)
    for category, files in list(category_files.items())[:5]:
        if files:
            print(f"\n  {category.upper()}:")
            for f in files[:3]:
                print(f"    - {f}")

def get_recommended_config(category_files, target_questions=100):
    """Generate recommended configuration based on document distribution"""
    
    total_docs = sum(len(files) for files in category_files.values())
    
    # Calculate recommended questions per category
    # Proportional to document count, with minimum of 3 per category
    recommended = {}
    
    for category, files in category_files.items():
        if not files:
            continue
        
        proportion = len(files) / total_docs
        questions = max(3, int(proportion * target_questions))
        recommended[category] = questions
    
    # Adjust to hit target
    total_recommended = sum(recommended.values())
    if total_recommended > target_questions:
        # Scale down
        scale = target_questions / total_recommended
        for category in recommended:
            recommended[category] = max(3, int(recommended[category] * scale))
    
    print("\n" + "=" * 70)
    print("🎯 RECOMMENDED CONFIGURATION")
    print("=" * 70)
    print(f"\nBased on {total_docs} documents, for {target_questions} target questions:\n")
    print("self.config = {")
    print(f'    "target_questions": {target_questions},')
    print('    "questions_per_category": {')
    
    for category, count in sorted(recommended.items(), key=lambda x: x[1], reverse=True):
        print(f'        "{category}": {count},')
    
    print('    }')
    print('}')

if __name__ == "__main__":
    category_files, counts = analyze_documents()
    print_analysis(category_files)
    get_recommended_config(category_files, target_questions=100)