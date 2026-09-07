"""
Simple timing utilities for performance measurement.
No circular imports - this file should NOT import from retrieval.
"""

import time
import functools
from collections import defaultdict
from typing import Dict, Any, List

_timing_data = defaultdict(list)

class Timer:
    """Context manager for timing code blocks"""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time
        _timing_data[self.name].append(self.elapsed)
        indent = "  " * self.name.count("_")
        print(f"{indent}⏱️ {self.name}: {self.elapsed:.4f}s")
    
    def get_elapsed(self) -> float:
        return self.elapsed if self.elapsed else 0.0


def timeit(func):
    """Decorator to time function calls"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        _timing_data[func.__name__].append(elapsed)
        print(f"⏱️ {func.__name__}: {elapsed:.4f}s")
        return result
    return wrapper


def print_timing_report():
    """Print formatted timing report"""
    if not _timing_data:
        print("No timing data available")
        return
    
    print("\n" + "=" * 70)
    print("📊 TIMING REPORT")
    print("=" * 70)
    print(f"{'Function':<45} {'Count':<8} {'Total':<12} {'Avg':<12}")
    print("-" * 70)
    
    total_time = 0
    for name, times in sorted(_timing_data.items(), key=lambda x: sum(x[1]), reverse=True):
        count = len(times)
        total = sum(times)
        avg = total / count if count > 0 else 0
        total_time += total
        print(f"{name:<45} {count:<8} {total:.4f}s   {avg:.4f}s")
    
    print("-" * 70)
    print(f"{'TOTAL':<45} {'':<8} {total_time:.4f}s")
    print("=" * 70)


def reset_timing():
    """Reset all timing data"""
    global _timing_data
    _timing_data = defaultdict(list)


def get_timing_data() -> Dict[str, List[float]]:
    """Get raw timing data"""
    return dict(_timing_data)