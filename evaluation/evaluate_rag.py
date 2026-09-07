# evaluation/evaluate_rag.py
"""
Evaluate RAG system using generated QA dataset
"""

import pandas as pd
import requests
import json
import time
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGEvaluator:
    def __init__(self, dataset_path: str = "evaluation/qa_eval_dataset.csv"):
        self.dataset_path = Path(dataset_path)
        self.dataset = None
        self.results = []
        self.api_url = "http://localhost:8000/query"
        self.metrics = {}
        
        # Load dataset
        self.load_dataset()
    
    def load_dataset(self) -> bool:
        """Load the QA dataset"""
        if not self.dataset_path.exists():
            logger.error(f"❌ Dataset not found: {self.dataset_path}")
            logger.info("Please generate questions first: python -m evaluation.generate_enterprise_qa")
            return False
        
        try:
            self.dataset = pd.read_csv(self.dataset_path)
            logger.info(f"✅ Loaded {len(self.dataset)} questions from {self.dataset_path}")
            
            # Show sample
            if len(self.dataset) > 0:
                logger.info(f"📝 Sample question: {self.dataset.iloc[0]['question'][:60]}...")
            
            return True
        except Exception as e:
            logger.error(f"❌ Error loading dataset: {e}")
            return False
    
    def evaluate_query(self, question: str, expected_answer: str, 
                       expected_category: str = "", expected_type: str = "") -> Dict:
        """
        Query RAG system and compare with expected answer
        """
        payload = {
            "question": question,
            "top_k": 5,
            "stream": False
        }
        
        start_time = time.time()
        try:
            response = requests.post(self.api_url, json=payload, timeout=300)
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                generated_answer = data.get("answer", "")
                confidence = data.get("confidence", 0)
                reliable = data.get("reliable", False)
                sources = data.get("sources", [])
                
                # Calculate answer similarity (simple word overlap)
                similarity = self.calculate_similarity(generated_answer, expected_answer)
                
                return {
                    "question": question,
                    "expected_answer": expected_answer,
                    "generated_answer": generated_answer,
                    "confidence": confidence,
                    "reliable": reliable,
                    "response_time": elapsed,
                    "sources": len(sources),
                    "source_list": sources[:3],  # First 3 sources
                    "similarity_score": similarity,
                    "category": expected_category,
                    "question_type": expected_type,
                    "status": "success"
                }
            else:
                return {
                    "question": question,
                    "expected_answer": expected_answer,
                    "generated_answer": "",
                    "error": f"HTTP {response.status_code}",
                    "response_time": elapsed,
                    "status": "failed"
                }
        except requests.exceptions.Timeout:
            return {
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": "",
                "error": "Timeout",
                "response_time": elapsed if 'elapsed' in locals() else 60,
                "status": "timeout"
            }
        except Exception as e:
            return {
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": "",
                "error": str(e),
                "response_time": time.time() - start_time if 'start_time' in locals() else 0,
                "status": "error"
            }
    
    def calculate_similarity(self, generated: str, expected: str) -> float:
        """
        Calculate simple similarity between generated and expected answers
        Uses word overlap for speed (no heavy models)
        """
        if not generated or not expected:
            return 0.0
        
        # Normalize
        generated_words = set(generated.lower().split())
        expected_words = set(expected.lower().split())
        
        if not expected_words:
            return 0.0
        
        # Calculate overlap
        overlap = len(generated_words.intersection(expected_words))
        total = len(expected_words)
        
        return min(1.0, overlap / total)
    
    def run_evaluation(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Run evaluation on all questions
        
        Args:
            limit: Limit number of questions to evaluate (for testing)
        """
        if self.dataset is None:
            logger.error("❌ No dataset loaded")
            return pd.DataFrame()
        
        # Limit questions if specified
        if limit and len(self.dataset) > limit:
            eval_dataset = self.dataset.head(limit)
            logger.info(f"🔍 Limiting evaluation to {limit} questions")
        else:
            eval_dataset = self.dataset
        
        logger.info(f"🔍 Evaluating {len(eval_dataset)} questions...")
        logger.info("-" * 50)
        
        # Progress bar
        for idx, row in tqdm(eval_dataset.iterrows(), total=len(eval_dataset), 
                            desc="Evaluating"):
            question = row.get('question', '')
            expected = row.get('answer', '')
            category = row.get('category', '')
            q_type = row.get('question_type', '')
            
            if not question or not expected:
                logger.warning(f"⚠️ Skipping row {idx}: missing question or answer")
                continue
            
            result = self.evaluate_query(question, expected, category, q_type)
            self.results.append(result)
            
            # Show sample results periodically
            if len(self.results) % 10 == 0:
                logger.info(f"   Processed {len(self.results)} questions...")
        
        return self.generate_report()
    
    def generate_report(self) -> pd.DataFrame:
        """Generate comprehensive evaluation report"""
        if not self.results:
            logger.warning("⚠️ No results to report")
            return pd.DataFrame()
        
        df = pd.DataFrame(self.results)
        
        print("\n" + "=" * 80)
        print("📊 RAG EVALUATION REPORT")
        print("=" * 80)
        
        # Overall stats
        total = len(df)
        success = df[df['status'] == 'success']
        failed = df[df['status'] != 'success']
        
        print(f"\n📈 Overall Statistics:")
        print(f"   Total queries: {total}")
        print(f"   ✅ Successful: {len(success)} ({len(success)/total*100:.1f}%)")
        print(f"   ❌ Failed: {len(failed)} ({len(failed)/total*100:.1f}%)")
        
        if not success.empty:
            # Performance metrics
            print(f"\n📊 Performance Metrics (Successful Queries):")
            print(f"   Average confidence: {success['confidence'].mean():.3f}")
            print(f"   Reliable responses: {success['reliable'].sum()}/{len(success)} ({success['reliable'].sum()/len(success)*100:.1f}%)")
            print(f"   Average response time: {success['response_time'].mean():.2f}s")
            print(f"   Fastest response: {success['response_time'].min():.2f}s")
            print(f"   Slowest response: {success['response_time'].max():.2f}s")
            print(f"   Average sources: {success['sources'].mean():.1f}")
            
            # Answer quality
            if 'similarity_score' in success.columns:
                print(f"\n📝 Answer Quality:")
                print(f"   Average similarity score: {success['similarity_score'].mean():.3f}")
                print(f"   High quality answers (similarity > 0.7): {len(success[success['similarity_score'] > 0.7])}/{len(success)}")
                print(f"   Low quality answers (similarity < 0.3): {len(success[success['similarity_score'] < 0.3])}/{len(success)}")
        
        # By category
        if 'category' in success.columns:
            print(f"\n📂 By Category:")
            for category in success['category'].unique():
                cat_data = success[success['category'] == category]
                print(f"   - {category}: {len(cat_data)} questions, "
                      f"avg confidence: {cat_data['confidence'].mean():.3f}, "
                      f"avg similarity: {cat_data['similarity_score'].mean():.3f}")
        
        # By question type
        if 'question_type' in success.columns:
            print(f"\n📝 By Question Type:")
            for q_type in success['question_type'].unique():
                type_data = success[success['question_type'] == q_type]
                print(f"   - {q_type}: {len(type_data)} questions, "
                      f"avg confidence: {type_data['confidence'].mean():.3f}")
        
        # Failed queries
        if not failed.empty:
            print(f"\n❌ Failed Queries ({len(failed)}):")
            for _, row in failed.head(5).iterrows():
                print(f"   - {row['question'][:60]}...")
                print(f"     Error: {row.get('error', 'Unknown')}")
        
        # Poor quality answers
        if not success.empty and 'similarity_score' in success.columns:
            poor = success[success['similarity_score'] < 0.3]
            if not poor.empty:
                print(f"\n⚠️ Poor Quality Answers ({len(poor)}):")
                for _, row in poor.head(3).iterrows():
                    print(f"\n   Q: {row['question'][:60]}...")
                    print(f"   Expected: {row['expected_answer'][:60]}...")
                    print(f"   Got: {row['generated_answer'][:60]}...")
                    print(f"   Similarity: {row['similarity_score']:.3f}")
        
        # Save results
        output_path = "evaluation/evaluation_results.csv"
        df.to_csv(output_path, index=False)
        print(f"\n📁 Results saved to: {output_path}")
        
        # Save summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_queries": total,
            "successful": len(success),
            "failed": len(failed),
            "avg_confidence": success['confidence'].mean() if not success.empty else 0,
            "reliable_percentage": success['reliable'].sum() / len(success) * 100 if not success.empty else 0,
            "avg_response_time": success['response_time'].mean() if not success.empty else 0,
            "avg_similarity": success['similarity_score'].mean() if not success.empty and 'similarity_score' in success.columns else 0,
        }
        
        with open("evaluation/evaluation_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"📁 Summary saved to: evaluation/evaluation_summary.json")
        
        return df
    
    def run_full_evaluation(self):
        """Run complete evaluation pipeline"""
        if self.dataset is None:
            if not self.load_dataset():
                logger.error("❌ Cannot run evaluation: dataset not loaded")
                return
        
        # Run evaluation
        df = self.run_evaluation()
        
        if df.empty:
            logger.error("❌ Evaluation failed")
        else:
            logger.info("✅ Evaluation complete!")
            return df

def main():
    """Main entry point"""
    print("=" * 80)
    print("🚀 RAG EVALUATION SYSTEM")
    print("=" * 80)
    
    # Check if API is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is running")
        else:
            print("⚠️ API health check failed")
    except:
        print("❌ API is not running!")
        print("   Please start the API: uvicorn api.main:app --reload --port 8000")
        return
    
    # Check if dataset exists
    dataset_path = Path("evaluation/qa_eval_dataset.csv")
    if not dataset_path.exists():
        print("❌ Dataset not found!")
        print("   Please generate questions first:")
        print("   python -m evaluation.generate_enterprise_qa")
        return
    
    # Run evaluation
    evaluator = RAGEvaluator(str(dataset_path))
    
    # Ask if user wants to limit queries
    limit = input("\n📊 Limit number of queries to test? (press Enter for all, or enter number): ").strip()
    if limit.isdigit():
        evaluator.run_evaluation(limit=int(limit))
    else:
        evaluator.run_evaluation()

if __name__ == "__main__":
    main()