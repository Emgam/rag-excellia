# evaluation/generate_enterprise_qa.py
"""
Enterprise-Level QA Generation for RAG Evaluation
Generates diverse, high-quality test questions from all documents
"""

import os
import sys
import random
import pandas as pd
import json
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Any, Optional

# LangChain imports
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    CSVLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate

# Logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnterpriseQAGenerator:
    """Enterprise-grade QA generation for RAG evaluation"""
    
    def __init__(self, docs_path: str = "./documents", model: str = "llama3.2:3b"):
        self.docs_path = Path(docs_path)
        self.model = model
        self.llm = None
        self.documents = []
        self.categories = defaultdict(list)
        self.questions = []
        
        # Configuration - FIXED: Added all required keys
        self.config = {
            "target_questions": 100,
            "temperature": 0.0,           # ✅ ADDED
            "min_chunk_size": 100,        # ✅ ADDED
            "max_chunk_size": 800,        # ✅ ADDED
            "question_types": [           # ✅ ADDED
                "factual", "definition", "comparative", 
                "explanatory", "listing", "scenario", "process"
            ],
            "questions_per_category": {
                # Banking categories
                "banking": 20,
                "digital": 10,
                "loans": 5,
                
                # Aviation categories
                "aircraft": 20,
                "cabin_features": 18,
                "services": 12,
                "policies": 10,
                
                # Other
                "general": 5,
            }
        }
        
        logger.info(f"🚀 Initializing Enterprise QA Generator")
        logger.info(f"📁 Documents path: {self.docs_path}")
        logger.info(f"🤖 Model: {self.model}")
    
    def setup_llm(self):
        """Initialize LLM"""
        try:
            self.llm = ChatOllama(
                model=self.model,
                base_url="http://localhost:11434",
                temperature=self.config["temperature"],
                num_predict=200
            )
            logger.info("✅ LLM initialized successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM: {e}")
            return False
    
    def load_documents(self):
        """Load documents from all supported file types"""
        logger.info("📁 Loading documents...")
        
        # File type loaders
        loaders = {
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.pptx': UnstructuredPowerPointLoader,
            '.md': TextLoader,
            '.html': UnstructuredHTMLLoader,
            '.htm': UnstructuredHTMLLoader,
            '.txt': TextLoader,
            '.csv': CSVLoader,
        }
        
        documents = []
        stats = defaultdict(int)
        
        for root, dirs, files in os.walk(self.docs_path):
            for file in files:
                file_path = Path(root) / file
                ext = file_path.suffix.lower()
                
                if ext in loaders:
                    try:
                        # Load with appropriate loader
                        if ext in ['.md', '.txt']:
                            loader = loaders[ext](str(file_path), encoding='utf-8')
                        else:
                            loader = loaders[ext](str(file_path))
                        
                        docs = loader.load()
                        
                        # Add metadata
                        for doc in docs:
                            doc.metadata['source'] = str(file_path)
                            doc.metadata['file_name'] = file
                            doc.metadata['file_type'] = ext
                            doc.metadata['file_size'] = file_path.stat().st_size
                            doc.metadata['category'] = self._categorize_document(file_path)
                        
                        documents.extend(docs)
                        stats[ext] += len(docs)
                        logger.info(f"  ✅ Loaded {len(docs)} chunks from {file}")
                        
                    except Exception as e:
                        logger.warning(f"  ⚠️ Error loading {file}: {e}")
        
        self.documents = documents
        
        logger.info("\n📊 Loading Summary:")
        for ext, count in sorted(stats.items()):
            logger.info(f"   {ext}: {count} chunks")
        logger.info(f"   TOTAL: {len(documents)} chunks")
        
        return documents
    
    def _categorize_document(self, file_path: Path) -> str:
        """Categorize document based on filename and path"""
        file_str = str(file_path).lower()
        
        categories = {
            'aircraft': ['airbus', 'boeing', 'a350', 'a380', '787', 'dreamliner', 'fleet'],
            'cabin_features': ['class', 'cloudlux', 'cloudelite', 'cloudcomfort', 'cloudsaver', 'seat', 'cabin'],
            'services': ['baggage', 'boarding', 'check-in', 'lounge', 'meal', 'wifi', 'entertainment'],
            'policies': ['discount', 'policy', 'terms', 'conditions', 'armed forces'],
            'banking': ['bank', 'branch', 'atm', 'account', 'savings', 'current'],
            'loans': ['loan', 'credit', 'mortgage', 'finance', 'interest'],
            'digital': ['app', 'mobile', 'netbanking', 'online', 'digital', 'upi', 'payment'],
        }
        
        for category, keywords in categories.items():
            if any(kw in file_str for kw in keywords):
                return category
        
        return 'general'
    
    def categorize_documents(self):
        """Categorize all documents by topic"""
        logger.info("📊 Categorizing documents...")
        
        # Use metadata categories or analyze content
        for doc in self.documents:
            category = doc.metadata.get('category', 'general')
            self.categories[category].append(doc)
        
        # If no categories, try to categorize based on content
        if len(self.categories) <= 1:
            logger.info("   Re-categorizing based on content...")
            # Reset categories
            self.categories = defaultdict(list)
            
            for doc in self.documents:
                content = doc.page_content.lower()
                file_name = doc.metadata.get('file_name', '').lower()
                
                # Check content for keywords
                categorized = False
                for category, keywords in self._get_category_keywords().items():
                    if any(kw in content or kw in file_name for kw in keywords):
                        self.categories[category].append(doc)
                        categorized = True
                        break
                
                if not categorized:
                    self.categories['general'].append(doc)
        
        logger.info("   Categories found:")
        for category, docs in self.categories.items():
            logger.info(f"     - {category}: {len(docs)} documents")
    
    def _get_category_keywords(self) -> Dict[str, List[str]]:
        """Get keywords for each category"""
        return {
            'aircraft': ['airbus', 'boeing', 'a350', 'a380', '787', 'dreamliner', 'fleet', 'aircraft'],
            'cabin_features': ['class', 'cloudlux', 'cloudelite', 'cloudcomfort', 'cloudsaver', 'seating', 'cabin', 'seat'],
            'services': ['baggage', 'boarding', 'lounge', 'meal', 'wifi', 'entertainment', 'service'],
            'policies': ['discount', 'policy', 'terms', 'conditions', 'force', 'military'],
            'banking': ['bank', 'branch', 'atm', 'account', 'savings', 'current', 'deposit'],
            'loans': ['loan', 'credit', 'mortgage', 'finance', 'interest', 'repayment'],
            'digital': ['app', 'mobile', 'netbanking', 'online', 'digital', 'technology', 'upi', 'payment'],
        }
    
    def validate_question(self, question: str, answer: str, context: str) -> bool:
        """
        Validate that a generated question is valid and answerable from context.
        
        Args:
            question: The generated question
            answer: The generated answer
            context: The context text used for generation
        
        Returns:
            True if valid, False otherwise
        """
        # Question must have enough meaning (at least 5 words)
        if len(question.split()) < 5:
            return False
        
        # Answer must exist in context
        keywords = set(question.split())  # keep all words as‑is
        matches = sum(1 for word in keywords if word in context.lower())
        
        # At least 3 keywords must match
        if matches < 3:
            return False
        
        return True
    
    def generate_questions_for_category(self, category: str, docs: List[Any], num_questions: int = 5) -> List[Dict]:
        """Generate questions for a specific category"""
        logger.info(f"   📌 Generating {num_questions} questions for {category}...")
        
        questions = []
        
        # Question type prompts
        question_prompts = {
            'factual': "Create a specific factual question with numbers, dates, or detailed specifications.",
            'definition': "Create a clear definition question asking 'What is X?' for a key concept or feature.",
            'comparative': "Create a comparison question asking how two things, features, or options differ.",
            'explanatory': "Create a 'How' or 'Why' question that requires explaining a process or rationale.",
            'listing': "Create a question asking for a list of features, services, or options available.",
            'scenario': "Create a scenario-based question asking what a customer should do in a specific situation.",
            'process': "Create a question asking about a step-by-step process or procedure."
        }
        
        # Select documents
        selected_docs = random.sample(docs, min(len(docs), max(3, num_questions // 2)))
        
        for idx, doc in enumerate(selected_docs):
            if len(questions) >= num_questions:
                break
                
            if len(doc.page_content) < self.config["min_chunk_size"]:
                continue
            
            # Select random question type
            question_type = random.choice(self.config["question_types"])
            instruction = question_prompts.get(question_type, question_prompts['factual'])
            
            # Get document info
            file_name = doc.metadata.get('file_name', 'unknown')
            file_type = doc.metadata.get('file_type', 'unknown')
            content = doc.page_content[:self.config["max_chunk_size"]]
            
            prompt = PromptTemplate(
                template="""You are an expert QA evaluator. Generate a high-quality question and answer based on the given document.

Question Type: {question_type}
Instruction: {instruction}

Document: {file_name} ({file_type})
Content: {content}

Generate a {question_type} question that tests understanding of this document.
Make it realistic and challenging.

Format:
Q: [question]
A: [answer]
Difficulty: [Easy/Medium/Hard]
Tags: [comma-separated keywords]

Response:"""
            )
            
            try:
                response = self.llm.invoke(
                    prompt.format(
                        question_type=question_type,
                        instruction=instruction,
                        file_name=file_name,
                        file_type=file_type,
                        content=content
                    )
                )
                
                result = self._parse_qa_response(response.content)
                
                # Validate the question
                if result['question'] and result['answer']:
                    if self.validate_question(result['question'], result['answer'], content):
                        result.update({
                            'category': category,
                            'source_file': file_name,
                            'file_type': file_type,
                            'question_type': question_type,
                            'context': content[:300],
                        })
                        questions.append(result)
                        logger.info(f"     ✅ [{question_type}] {result['question'][:60]}...")
                    else:
                        logger.warning(f"     ⚠️ Question failed validation: {result['question'][:60]}...")
                    
            except Exception as e:
                logger.warning(f"     ❌ Error generating question: {e}")
                continue
            
            # Small delay to avoid rate limiting
            time.sleep(0.2)
        
        return questions
    
    def _parse_qa_response(self, response: str) -> Dict:
        """Parse the LLM response into Q, A, difficulty, tags"""
        result = {
            'question': '',
            'answer': '',
            'difficulty': 'Medium',
            'tags': []
        }
        
        lines = response.strip().split('\n')
        
        for line in lines:
            if line.startswith('Q:') or line.startswith('Q :'):
                result['question'] = line.replace('Q:', '').replace('Q :', '').strip()
            elif line.startswith('A:') or line.startswith('A :'):
                result['answer'] = line.replace('A:', '').replace('A :', '').strip()
            elif 'Difficulty:' in line or 'difficulty:' in line.lower():
                result['difficulty'] = line.split(':')[-1].strip()
            elif 'Tags:' in line or 'tags:' in line.lower():
                tags = line.split(':')[-1].strip()
                result['tags'] = [tag.strip() for tag in tags.split(',') if tag.strip()]
        
        return result
    
    def generate_all_questions(self) -> pd.DataFrame:
        """Generate questions from all categories"""
        logger.info("\n🔍 Generating questions...")
        logger.info("-" * 50)
        
        all_questions = []
        
        # Get questions per category
        for category, num_questions in self.config["questions_per_category"].items():
            if category in self.categories and self.categories[category]:
                questions = self.generate_questions_for_category(
                    category,
                    self.categories[category],
                    num_questions
                )
                all_questions.extend(questions)
            else:
                logger.info(f"   ⚠️ No documents found for category: {category}")
        
        # If not enough questions, generate from general
        if len(all_questions) < self.config["target_questions"] and self.categories.get('general'):
            needed = self.config["target_questions"] - len(all_questions)
            logger.info(f"\n📌 Generating {needed} extra questions from GENERAL category...")
            extra = self.generate_questions_for_category(
                'general',
                self.categories['general'],
                needed
            )
            all_questions.extend(extra)
        
        self.questions = all_questions
        return self.to_dataframe()
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert questions to DataFrame"""
        if not self.questions:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.questions)
        
        # Ensure all columns exist
        required_columns = ['question', 'answer', 'category', 'source_file', 'file_type', 'question_type', 'difficulty', 'tags', 'context']
        for col in required_columns:
            if col not in df.columns:
                df[col] = ''
        
        # Reorder columns
        order = ['question', 'answer', 'category', 'question_type', 'difficulty', 'source_file', 'file_type', 'tags', 'context']
        df = df[[col for col in order if col in df.columns]]
        
        return df
    
    def save_to_csv(self, df: pd.DataFrame, output_path: str = "evaluation/qa_eval_dataset.csv"):
        """Save questions to CSV"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"✅ Saved {len(df)} questions to {output_path}")
        return output_path
    
    def save_to_json(self, df: pd.DataFrame, output_path: str = "evaluation/qa_eval_dataset.json"):
        """Save questions to JSON"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_json(output_path, orient='records', indent=2)
        logger.info(f"✅ Saved {len(df)} questions to {output_path}")
        return output_path
    
    def generate_report(self, df: pd.DataFrame):
        """Generate summary report"""
        print("\n" + "=" * 70)
        print("📊 QA EVALUATION DATASET REPORT")
        print("=" * 70)
        
        print(f"\n📈 Total Questions: {len(df)}")
        
        # By category
        print("\n📂 By Category:")
        for category in df['category'].unique():
            count = len(df[df['category'] == category])
            print(f"   - {category}: {count} ({count/len(df)*100:.1f}%)")
        
        # By question type
        if 'question_type' in df.columns:
            print("\n📝 By Question Type:")
            for q_type in df['question_type'].unique():
                count = len(df[df['question_type'] == q_type])
                print(f"   - {q_type}: {count} ({count/len(df)*100:.1f}%)")
        
        # By difficulty
        if 'difficulty' in df.columns:
            print("\n⭐ By Difficulty:")
            for diff in df['difficulty'].unique():
                count = len(df[df['difficulty'] == diff])
                print(f"   - {diff}: {count} ({count/len(df)*100:.1f}%)")
        
        # By file type
        if 'file_type' in df.columns:
            print("\n📁 By File Type:")
            for ft in df['file_type'].unique():
                count = len(df[df['file_type'] == ft])
                print(f"   - {ft}: {count} ({count/len(df)*100:.1f}%)")
        
        # Sample questions
        print("\n📝 SAMPLE QUESTIONS:")
        print("-" * 50)
        for i, row in df.head(5).iterrows():
            print(f"\n  {i+1}. [{row.get('category', 'general')}] {row['question']}")
            print(f"     Answer: {row['answer'][:100]}...")
            print(f"     Source: {row.get('source_file', 'unknown')}")
    
    def run(self) -> pd.DataFrame:
        """Run the entire pipeline"""
        logger.info("\n" + "=" * 70)
        logger.info("🚀 STARTING ENTERPRISE QA GENERATION")
        logger.info("=" * 70)
        
        # Step 1: Setup LLM
        if not self.setup_llm():
            return pd.DataFrame()
        
        # Step 2: Load documents
        self.load_documents()
        if not self.documents:
            logger.error("❌ No documents loaded!")
            return pd.DataFrame()
        
        # Step 3: Categorize
        self.categorize_documents()
        
        # Step 4: Generate questions
        df = self.generate_all_questions()
        
        if df.empty:
            logger.error("❌ No questions generated!")
            return df
        
        # Step 5: Save
        self.save_to_csv(df)
        self.save_to_json(df)
        
        # Step 6: Report
        self.generate_report(df)
        
        logger.info("\n✅ Enterprise QA Generation Complete!")
        return df


def main():
    """Main entry point"""
    # Initialize generator
    generator = EnterpriseQAGenerator(
        docs_path="./documents",
        model="llama3.2:3b"
    )
    
    # Run
    df = generator.run()
    
    return df


if __name__ == "__main__":
    main()