
#!/usr/bin/env python
"""
Evaluate chunk size on a diverse set of ~200 documents.
Computes Context Recall (Answer in Chunks), MRR, and retrieval latency.
"""

import itertools
import time
import pandas as pd
import json
import shutil
import random
from pathlib import Path
from typing import List, Dict

from config.settings import get_settings
from retrieval.pipeline import retrieve
from ingestion.ingest_service import ingest

# ------------------------------------------------------------------
# Load QA dataset
# ------------------------------------------------------------------
def load_qa_dataset(path: str = "evaluation/qa_eval_dataset.json") -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ------------------------------------------------------------------
# Select N diverse documents (Stratified by Domain and Format)
# ------------------------------------------------------------------
def select_documents(qa_data: List[Dict], docs_dir: Path, target_count: int = 200) -> List[Path]:
    """
    Selects ~200 diverse files from documents/CloudWay-24 and documents/ZX Bank.
    Prioritizes files referenced in the QA dataset, then fills via stratified
    round-robin across all formats (pdf, docx, html, md, pptx) to ensure diversity.
    """
    qa_files = {Path(item["source_file"]).name for item in qa_data}
    
    domains = ["CloudWay-24", "ZX Bank"]
    formats = ["pdf", "docx", "html", "md", "pptx", "txt"]
    
    selected = []
    random.seed(42)
    
    target_per_domain = target_count // len(domains)  # 100 per domain
    
    print("📂 Scanning folder structure for diverse document selection...")
    
    for domain in domains:
        domain_path = docs_dir / domain
        if not domain_path.exists():
            print(f"⚠️ Domain folder not found: {domain_path}")
            continue
            
        # Group files by format
        files_by_format = {fmt: [] for fmt in formats}
        
        for fmt in formats:
            fmt_dir = domain_path / fmt
            if fmt_dir.exists():
                # Find all files matching the format extension
                files_by_format[fmt].extend(fmt_dir.glob(f"*.{fmt}"))
                
        all_domain_files = [f for fmt_list in files_by_format.values() for f in fmt_list]
        
        # 1. Prioritize QA files in this domain
        domain_qa_files = [f for f in all_domain_files if f.name in qa_files]
        random.shuffle(domain_qa_files)
        
        needed = target_per_domain
        selected.extend(domain_qa_files[:needed])
        needed -= len(domain_qa_files[:needed])
        
        # 2. Fill the rest using stratified round-robin across formats
        if needed > 0:
            non_qa_by_format = {
                fmt: [f for f in files_by_format[fmt] if f.name not in qa_files] 
                for fmt in formats
            }
            
            # Shuffle each format bucket
            for fmt in formats:
                random.shuffle(non_qa_by_format[fmt])
            
            # Round-robin pick: PDF, DOCX, HTML, MD, PPTX, PDF, DOCX...
            fmt_idx = 0
            while needed > 0 and any(non_qa_by_format.values()):
                fmt = formats[fmt_idx % len(formats)]
                if non_qa_by_format[fmt]:
                    file = non_qa_by_format[fmt].pop()
                    selected.append(file)
                    needed -= 1
                fmt_idx += 1
                
    print(f"✅ Selected {len(selected)} diverse files across domains and formats.")
    return selected[:target_count]

def prepare_test_folder(selected_files: List[Path], target_dir: Path):
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(exist_ok=True)
    for f in selected_files:
        # Flatten the directory structure into the test folder
        shutil.copy2(f, target_dir / f.name)

# ------------------------------------------------------------------
# Evaluate one chunking configuration
# ------------------------------------------------------------------
def evaluate_config(chunk_size: int, overlap: int, qa_queries: List[Dict], test_dir: Path) -> Dict:
    settings = get_settings()
    
    # Save originals
    original_size = settings.CHUNK_SIZE_TOKENS
    original_overlap = settings.CHUNK_OVERLAP_TOKENS
    original_reranker = settings.RERANKER_ENABLED

    # Mutate settings for this run
    settings.CHUNK_SIZE_TOKENS = chunk_size
    settings.CHUNK_OVERLAP_TOKENS = overlap
    settings.RERANKER_ENABLED = False  # Test raw chunking power

    print(f"🔄 Re-ingesting with chunk_size={chunk_size}, overlap={overlap}...")
    start_ingest = time.time()
    ingest(path=str(test_dir), recreate=True)
    end_ingest = time.time()
    print(f"✅ Re-ingest took {end_ingest - start_ingest:.1f}s")

    context_hits = 0
    file_hits = 0
    reciprocal_ranks = []
    retrieval_times = []

    for item in qa_queries:
        question = item["question"]
        expected_source = Path(item["source_file"]).name
        expected_answer = item["answer"].lower().strip()

        start_ret = time.time()
        chunks, _, _ = retrieve(question, top_k=5)
        ret_time = time.time() - start_ret
        retrieval_times.append(ret_time)

        found_rank = None
        answer_in_context = False

        for rank, sc in enumerate(chunks[:5], start=1):
            chunk_text = sc.chunk.text.lower()
            
            # 1. Check if the chunk contains the expected answer (Context Recall)
            if expected_answer in chunk_text or any(word in chunk_text for word in expected_answer.split() if len(word) > 4):
                answer_in_context = True
                found_rank = rank 
                break
            
            # 2. Fallback: Check if it's from the right file
            meta = sc.chunk.metadata
            source = getattr(meta, "file_name", getattr(meta, "document", ""))
            if source and Path(source).name == expected_source and found_rank is None:
                found_rank = rank

        if answer_in_context:
            context_hits += 1
            
        if found_rank:
            file_hits += 1
            reciprocal_ranks.append(1.0 / found_rank)
        else:
            reciprocal_ranks.append(0.0)

    # Metrics
    context_recall = context_hits / len(qa_queries) if qa_queries else 0.0
    file_recall = file_hits / len(qa_queries) if qa_queries else 0.0
    mrr = sum(reciprocal_ranks) / len(qa_queries) if qa_queries else 0.0
    avg_retrieval_time = sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0.0

    # Restore settings
    settings.CHUNK_SIZE_TOKENS = original_size
    settings.CHUNK_OVERLAP_TOKENS = original_overlap
    settings.RERANKER_ENABLED = original_reranker

    return {
        "chunk_size": chunk_size,
        "overlap": overlap,
        "context_recall@5": context_recall,  
        "file_recall@5": file_recall,
        "mrr": mrr,
        "avg_retrieval_time": avg_retrieval_time,
        "ingest_time": end_ingest - start_ingest,
    }

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    docs_dir = Path("./documents")
    test_dir = Path("./test_200")
    qa_path = Path("evaluation/qa_eval_dataset.json")

    qa_data = load_qa_dataset(qa_path)

    selected_files = select_documents(qa_data, docs_dir, target_count=200)
    prepare_test_folder(selected_files, test_dir)
    print(f"📁 Test folder prepared: {test_dir}")

    selected_names = {f.name for f in selected_files}
    qa_subset = [q for q in qa_data if Path(q["source_file"]).name in selected_names]
    
    # Use up to 200 matching QA queries
    qa_subset = qa_subset[:200] 
    print(f"🧪 Using {len(qa_subset)} QA queries for evaluation.")

    # Reduced grid for CPU speed (testing all combos takes hours on CPU)
    chunk_sizes = [250, 350, 500]
    overlaps = [0, 50]

    results = []
    for cs, ov in itertools.product(chunk_sizes, overlaps):
        print(f"\n{'='*60}")
        print(f"Evaluating chunk_size={cs}, overlap={ov}")
        res = evaluate_config(cs, ov, qa_subset, test_dir)
        results.append(res)
        print(f"  Context Recall@5: {res['context_recall@5']:.3f} | File Recall: {res['file_recall@5']:.3f} | MRR: {res['mrr']:.3f}")

    df = pd.DataFrame(results)
    df.to_csv("chunk_size_evaluation_200.csv", index=False)

    # Find best by Context Recall
    best = df.loc[df["context_recall@5"].idxmax()]

    print("\n🏆 Best Configuration (by Context Recall):")
    print(best)

    print("\n✅ Recommendation:")
    print(f"  Use chunk_size = {best['chunk_size']:.0f}, overlap = {best['overlap']:.0f}")

if __name__ == "__main__":
    main()
