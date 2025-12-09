#!/usr/bin/env python3
"""
RAG Performance Testing Script

This script measures the time spent in each node of the RAG pipeline
to identify performance bottlenecks.

Usage:
    python test_rag_performance.py
"""
import logging
import sys
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from src.vector_store import VectorStoreManager
from src.rag_chain_langgraph import DocumentRAGGraph, create_rag_graph, query_rag_graph

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TimingStats:
    """Collect and display timing statistics."""
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
        self.query_timings: List[Dict[str, float]] = []
    
    def record(self, name: str, duration: float) -> None:
        """Record a timing measurement."""
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(duration)
    
    def start_query(self) -> Dict[str, float]:
        """Start timing a new query."""
        query_timing = {}
        self.query_timings.append(query_timing)
        return query_timing
    
    def print_summary(self) -> None:
        """Print timing summary."""
        print("\n" + "=" * 70)
        print("TIMING SUMMARY")
        print("=" * 70)
        
        if not self.timings:
            print("No timings recorded.")
            return
        
        # Calculate totals and averages
        print(f"\n{'Component':<30} {'Total (s)':<12} {'Avg (s)':<12} {'Count':<8}")
        print("-" * 70)
        
        sorted_timings = sorted(
            self.timings.items(),
            key=lambda x: sum(x[1]),
            reverse=True
        )
        
        total_time = 0
        for name, times in sorted_timings:
            total = sum(times)
            avg = total / len(times)
            total_time += total
            print(f"{name:<30} {total:<12.3f} {avg:<12.3f} {len(times):<8}")
        
        print("-" * 70)
        print(f"{'TOTAL':<30} {total_time:<12.3f}")
        print("=" * 70)


# Global timing stats
timing_stats = TimingStats()


def timed_node(node_name: str):
    """Decorator to time a node function."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start
                timing_stats.record(node_name, duration)
                logger.info(f"[TIMING] {node_name}: {duration:.3f}s")
        return wrapper
    return decorator


class InstrumentedRAGGraph(DocumentRAGGraph):
    """RAG Graph with timing instrumentation on each node."""
    
    def _initialize(self) -> None:
        """Initialize with instrumented nodes."""
        # Call parent initialization
        super()._initialize()
        
        # Wrap all node methods with timing
        self._wrap_nodes_with_timing()
    
    def _wrap_nodes_with_timing(self) -> None:
        """Wrap node methods with timing instrumentation."""
        # Store original methods
        original_memory = self._create_memory_node
        original_query_analysis = self._create_query_analysis_node
        original_rewrite = self._create_rewrite_node
        original_retrieval = self._create_retrieval_node
        original_reranking_decision = self._create_reranking_decision_node
        original_reranking = self._create_reranking_node
        original_generation = self._create_generation_node
        original_citation = self._create_citation_node
        original_similar_drugs = self._create_similar_drugs_node
        
        # Create timed wrappers
        def timed_memory(state):
            start = time.perf_counter()
            result = original_memory(state)
            timing_stats.record("1_memory", time.perf_counter() - start)
            return result
        
        def timed_query_analysis(state):
            start = time.perf_counter()
            result = original_query_analysis(state)
            timing_stats.record("2_query_analysis", time.perf_counter() - start)
            return result
        
        def timed_rewrite(state):
            start = time.perf_counter()
            result = original_rewrite(state)
            timing_stats.record("3_query_rewrite", time.perf_counter() - start)
            return result
        
        def timed_retrieval(state):
            start = time.perf_counter()
            result = original_retrieval(state)
            timing_stats.record("4_retrieval", time.perf_counter() - start)
            return result
        
        def timed_reranking_decision(state):
            start = time.perf_counter()
            result = original_reranking_decision(state)
            timing_stats.record("5_reranking_decision", time.perf_counter() - start)
            return result
        
        def timed_reranking(state):
            start = time.perf_counter()
            result = original_reranking(state)
            timing_stats.record("6_reranking", time.perf_counter() - start)
            return result
        
        def timed_generation(state):
            start = time.perf_counter()
            result = original_generation(state)
            timing_stats.record("7_generation", time.perf_counter() - start)
            return result
        
        def timed_citation(state):
            start = time.perf_counter()
            result = original_citation(state)
            timing_stats.record("8_citation", time.perf_counter() - start)
            return result
        
        def timed_similar_drugs(state):
            start = time.perf_counter()
            result = original_similar_drugs(state)
            timing_stats.record("9_similar_drugs", time.perf_counter() - start)
            return result
        
        # Replace methods
        self._create_memory_node = timed_memory
        self._create_query_analysis_node = timed_query_analysis
        self._create_rewrite_node = timed_rewrite
        self._create_retrieval_node = timed_retrieval
        self._create_reranking_decision_node = timed_reranking_decision
        self._create_reranking_node = timed_reranking
        self._create_generation_node = timed_generation
        self._create_citation_node = timed_citation
        self._create_similar_drugs_node = timed_similar_drugs


def create_instrumented_rag_graph(
    vector_store_manager: VectorStoreManager,
    provider: str = None,
    memory_store: Optional[Dict] = None
) -> InstrumentedRAGGraph:
    """Create an instrumented RAG graph for performance testing."""
    return InstrumentedRAGGraph(
        vector_store_manager=vector_store_manager,
        provider=provider,
        memory_store=memory_store
    )


def run_performance_test():
    """Run performance tests with example queries."""
    # Example queries from the plan
    test_queries = [
        "Má gefa barni íbúfen?",
        "Hvaða áhrif hefur morfín á flogaveika?",
        "Hver er munurinn á innihaldsefnum voltaren forte og diklofenak teva?",
        "Hver er hámarksskammtur af Tegretol?",
    ]
    
    print("\n" + "=" * 70)
    print("RAG PERFORMANCE TEST")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  LLM Provider: {Config.LLM_PROVIDER}")
    print(f"  Enable Query Rewrite: {Config.ENABLE_QUERY_REWRITE}")
    print(f"  Enable Reranking: {Config.ENABLE_RERANKING}")
    print(f"  Retrieval Top-K: {Config.RETRIEVAL_TOP_K}")
    print(f"  Retrieval Initial-K: {Config.RETRIEVAL_INITIAL_K}")
    
    try:
        # Initialize vector store with timing
        print("\n[1/3] Initializing vector store...")
        vs_start = time.perf_counter()
        vector_store_manager = VectorStoreManager()
        vs_duration = time.perf_counter() - vs_start
        timing_stats.record("0_vector_store_init", vs_duration)
        
        doc_count = vector_store_manager.get_document_count()
        print(f"  Vector store initialized in {vs_duration:.3f}s with {doc_count} documents")
        
        if doc_count == 0:
            print("ERROR: No documents in vector store. Please run ingestion first.")
            return False
        
        # Create instrumented RAG graph
        print("\n[2/3] Creating instrumented RAG graph...")
        graph_start = time.perf_counter()
        rag_graph = create_instrumented_rag_graph(
            vector_store_manager=vector_store_manager,
            provider=Config.LLM_PROVIDER
        )
        graph_duration = time.perf_counter() - graph_start
        timing_stats.record("0_graph_init", graph_duration)
        print(f"  RAG graph created in {graph_duration:.3f}s")
        
        # Run test queries
        print("\n[3/3] Running test queries...")
        print("-" * 70)
        
        query_results = []
        
        for i, query in enumerate(test_queries, 1):
            print(f"\nQuery {i}/{len(test_queries)}: {query}")
            
            query_start = time.perf_counter()
            
            try:
                result = rag_graph.process_message(
                    question=query,
                    session_id=f"perf-test-{i}"
                )
                
                query_duration = time.perf_counter() - query_start
                timing_stats.record("total_query_time", query_duration)
                
                success = not result.get("error")
                answer_preview = result.get("answer", "")[:100] + "..." if result.get("answer") else "No answer"
                sources_count = len(result.get("sources", []))
                
                query_results.append({
                    "query": query,
                    "duration": query_duration,
                    "success": success,
                    "sources": sources_count
                })
                
                status = "✓" if success else "✗"
                print(f"  {status} Completed in {query_duration:.3f}s ({sources_count} sources)")
                
                if result.get("error"):
                    print(f"  Error: {result['error']}")
                
            except Exception as e:
                query_duration = time.perf_counter() - query_start
                print(f"  ✗ Failed after {query_duration:.3f}s: {e}")
                query_results.append({
                    "query": query,
                    "duration": query_duration,
                    "success": False,
                    "sources": 0
                })
        
        # Print results summary
        print("\n" + "=" * 70)
        print("QUERY RESULTS")
        print("=" * 70)
        print(f"\n{'#':<3} {'Query':<50} {'Time (s)':<10} {'Status':<8}")
        print("-" * 70)
        
        for i, result in enumerate(query_results, 1):
            query_short = result["query"][:47] + "..." if len(result["query"]) > 50 else result["query"]
            status = "OK" if result["success"] else "FAIL"
            print(f"{i:<3} {query_short:<50} {result['duration']:<10.3f} {status:<8}")
        
        avg_time = sum(r["duration"] for r in query_results) / len(query_results)
        print("-" * 70)
        print(f"{'AVG':<3} {'':<50} {avg_time:<10.3f}")
        
        # Print detailed timing breakdown
        timing_stats.print_summary()
        
        # Performance assessment
        print("\n" + "=" * 70)
        print("PERFORMANCE ASSESSMENT")
        print("=" * 70)
        
        if avg_time < 10:
            print(f"\n✓ GOOD: Average query time ({avg_time:.1f}s) is under 10s target")
        elif avg_time < 30:
            print(f"\n⚠ NEEDS WORK: Average query time ({avg_time:.1f}s) is between 10-30s")
        else:
            print(f"\n✗ POOR: Average query time ({avg_time:.1f}s) exceeds 30s")
        
        # Identify bottlenecks
        print("\nTop bottlenecks:")
        sorted_timings = sorted(
            timing_stats.timings.items(),
            key=lambda x: sum(x[1]),
            reverse=True
        )[:5]
        
        for name, times in sorted_timings:
            if name.startswith("0_"):
                continue  # Skip initialization timings
            avg = sum(times) / len(times)
            print(f"  - {name}: {avg:.3f}s avg")
        
        return True
        
    except Exception as e:
        logger.error(f"Performance test failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    Config.validate()
    success = run_performance_test()
    sys.exit(0 if success else 1)
