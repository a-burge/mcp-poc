"""
Performance test script for LangGraph RAG implementation.

Measures timing for each node in the RAG pipeline to identify bottlenecks.
"""
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List
from contextlib import contextmanager

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from src.vector_store import VectorStoreManager
from src.rag_chain_langgraph import create_rag_graph, query_rag_graph

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerformanceTimer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        return False
    
    def __repr__(self) -> str:
        return f"{self.name}: {self.duration:.3f}s" if self.duration else f"{self.name}: not finished"


def wrap_node_with_timing(original_node_func, node_name: str):
    """Wrap a LangGraph node function with timing."""
    def timed_node(state: Dict[str, Any]) -> Dict[str, Any]:
        with PerformanceTimer(node_name) as timer:
            result = original_node_func(state)
        # Store timing in state for later retrieval
        if "node_timings" not in state:
            state["node_timings"] = {}
        state["node_timings"][node_name] = timer.duration
        return result
    return timed_node


def test_query_performance(
    rag_graph,
    question: str,
    query_name: str,
    session_id: str = None
) -> Dict[str, Any]:
    """
    Test a single query and measure performance.
    
    Args:
        rag_graph: RAG graph instance
        question: Question to ask
        query_name: Name for this query (for reporting)
        session_id: Optional session ID
        
    Returns:
        Dictionary with results and timing information
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing query: {query_name}")
    logger.info(f"Question: {question}")
    logger.info(f"{'='*80}")
    
    # Wrap nodes with timing
    original_nodes = {
        "memory": rag_graph._create_memory_node,
        "query_analysis": rag_graph._create_query_analysis_node,
        "query_rewrite": rag_graph._create_rewrite_node,
        "retrieval": rag_graph._create_retrieval_node,
        "reranking_decision": rag_graph._create_reranking_decision_node,
        "reranking": rag_graph._create_reranking_node,
        "generation": rag_graph._create_generation_node,
        "citation": rag_graph._create_citation_node,
        "extract_similar_drugs": rag_graph._create_similar_drugs_node,
    }
    
    # Store original nodes
    wrapped_nodes = {}
    for node_name, node_func in original_nodes.items():
        wrapped_nodes[node_name] = wrap_node_with_timing(node_func, node_name)
    
    # Temporarily replace nodes with timed versions
    graph = rag_graph.graph.get_graph()
    for node_name, wrapped_func in wrapped_nodes.items():
        if node_name in graph.nodes:
            # Replace the node function
            setattr(rag_graph, f"_create_{node_name}_node", wrapped_func)
    
    # Measure total time
    total_timer = PerformanceTimer("TOTAL")
    total_timer.__enter__()
    
    try:
        result = query_rag_graph(
            rag_graph=rag_graph,
            question=question,
            session_id=session_id
        )
    finally:
        total_timer.__exit__(None, None, None)
        
        # Restore original nodes
        for node_name, original_func in original_nodes.items():
            setattr(rag_graph, f"_create_{node_name}_node", original_func)
    
    # Extract timing from result if available
    # Note: LangGraph state is internal, so we'll use a different approach
    # We'll patch the nodes directly to log timing
    
    return {
        "query_name": query_name,
        "question": question,
        "result": result,
        "total_time": total_timer.duration
    }


def test_query_with_detailed_timing(
    rag_graph,
    question: str,
    query_name: str,
    session_id: str = None
) -> Dict[str, Any]:
    """
    Test a query with detailed node-level timing using monkey patching.
    
    Args:
        rag_graph: RAG graph instance
        question: Question to ask
        query_name: Name for this query
        session_id: Optional session ID
        
    Returns:
        Dictionary with results and detailed timing
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing query: {query_name}")
    logger.info(f"Question: {question}")
    logger.info(f"{'='*80}")
    
    # Store timing data
    timings: Dict[str, float] = {}
    
    # Wrap each node method with timing
    original_methods = {}
    node_names = [
        "memory",
        "query_analysis", 
        "query_rewrite",
        "retrieval",
        "reranking_decision",
        "reranking",
        "generation",
        "citation",
        "extract_similar_drugs"
    ]
    
    def create_timed_wrapper(node_name: str, original_method):
        def timed_wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
            start = time.time()
            try:
                result = original_method(state)
            finally:
                elapsed = time.time() - start
                timings[node_name] = elapsed
                logger.info(f"[TIMING] {node_name}: {elapsed:.3f}s")
            return result
        return timed_wrapper
    
    # Patch methods
    for node_name in node_names:
        method_name = f"_create_{node_name}_node"
        if hasattr(rag_graph, method_name):
            original_methods[node_name] = getattr(rag_graph, method_name)
            setattr(
                rag_graph,
                method_name,
                create_timed_wrapper(node_name, original_methods[node_name])
            )
    
    # Measure total time
    total_start = time.time()
    
    try:
        result = query_rag_graph(
            rag_graph=rag_graph,
            question=question,
            session_id=session_id
        )
    finally:
        total_time = time.time() - total_start
        
        # Restore original methods
        for node_name, original_method in original_methods.items():
            method_name = f"_create_{node_name}_node"
            setattr(rag_graph, method_name, original_method)
    
    timings["TOTAL"] = total_time
    
    return {
        "query_name": query_name,
        "question": question,
        "result": result,
        "timings": timings
    }


def print_timing_summary(results: List[Dict[str, Any]]):
    """Print a formatted timing summary table."""
    print("\n" + "="*100)
    print("PERFORMANCE SUMMARY")
    print("="*100)
    
    # Collect all node names
    all_nodes = set()
    for result in results:
        if "timings" in result:
            all_nodes.update(result["timings"].keys())
    
    # Sort nodes (TOTAL last)
    sorted_nodes = sorted([n for n in all_nodes if n != "TOTAL"]) + ["TOTAL"]
    
    # Print header
    header = f"{'Query':<50}"
    for node in sorted_nodes:
        header += f"{node:>12}"
    print(header)
    print("-" * 100)
    
    # Print each query's timings
    for result in results:
        query_name = result.get("query_name", "Unknown")
        timings = result.get("timings", {})
        
        row = f"{query_name:<50}"
        for node in sorted_nodes:
            time_val = timings.get(node, 0.0)
            row += f"{time_val:>12.3f}"
        print(row)
    
    # Print averages
    if len(results) > 1:
        print("-" * 100)
        avg_row = f"{'AVERAGE':<50}"
        for node in sorted_nodes:
            avg_time = sum(r.get("timings", {}).get(node, 0.0) for r in results) / len(results)
            avg_row += f"{avg_time:>12.3f}"
        print(avg_row)
    
    print("="*100)
    
    # Print detailed breakdown for each query
    for result in results:
        query_name = result.get("query_name", "Unknown")
        timings = result.get("timings", {})
        
        print(f"\n{query_name}:")
        print(f"  Total time: {timings.get('TOTAL', 0.0):.3f}s")
        
        # Sort by time (descending)
        sorted_timings = sorted(
            [(k, v) for k, v in timings.items() if k != "TOTAL"],
            key=lambda x: x[1],
            reverse=True
        )
        
        for node_name, node_time in sorted_timings:
            percentage = (node_time / timings.get("TOTAL", 1.0)) * 100
            print(f"    {node_name}: {node_time:.3f}s ({percentage:.1f}%)")


def main():
    """Run performance tests."""
    logger.info("="*80)
    logger.info("RAG Performance Test")
    logger.info("="*80)
    
    # Validate configuration
    try:
        Config.validate()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    
    # Test queries
    test_queries = [
        ("Má gefa barni íbúfen?", "Query 1: Can child be given ibuprofen?"),
        ("Hvaða áhrif hefur morfín á flogaveika?", "Query 2: Effects of morphine on epilepsy"),
        ("Hver er munurinn á innihaldsefnum voltaren forte og diklofenak teva?", "Query 3: Difference between Voltaren Forte and Diclofenac Teva"),
        ("Hver er hámarksskammtur af Tegretol?", "Query 4: Maximum dose of Tegretol"),
    ]
    
    # Initialize vector store
    logger.info("Initializing vector store...")
    vector_store_manager = VectorStoreManager()
    doc_count = vector_store_manager.get_document_count()
    logger.info(f"Vector store initialized with {doc_count} documents")
    
    if doc_count == 0:
        logger.error("No documents in vector store. Please run ingestion first.")
        sys.exit(1)
    
    # Create RAG graph
    logger.info(f"Creating RAG graph with provider: {Config.LLM_PROVIDER}")
    logger.info(f"ENABLE_QUERY_REWRITE: {Config.ENABLE_QUERY_REWRITE}")
    logger.info(f"ENABLE_RERANKING: {Config.ENABLE_RERANKING}")
    
    rag_graph = create_rag_graph(
        vector_store_manager=vector_store_manager,
        provider=Config.LLM_PROVIDER
    )
    
    # Run tests
    results = []
    for question, query_name in test_queries:
        try:
            result = test_query_with_detailed_timing(
                rag_graph=rag_graph,
                question=question,
                query_name=query_name
            )
            results.append(result)
            
            # Check for errors
            if result["result"].get("error"):
                logger.error(f"Error in {query_name}: {result['result']['error']}")
            else:
                logger.info(f"✓ {query_name} completed in {result['timings'].get('TOTAL', 0.0):.3f}s")
                logger.info(f"  Answer length: {len(result['result'].get('answer', ''))} chars")
                logger.info(f"  Sources: {len(result['result'].get('sources', []))}")
        except Exception as e:
            logger.error(f"Test failed for {query_name}: {e}", exc_info=True)
            results.append({
                "query_name": query_name,
                "question": question,
                "result": {"error": str(e)},
                "timings": {}
            })
    
    # Print summary
    print_timing_summary(results)
    
    # Check if targets are met
    avg_total = sum(r.get("timings", {}).get("TOTAL", 0.0) for r in results) / len(results)
    target_time = 10.0
    
    logger.info(f"\nAverage total time: {avg_total:.3f}s")
    logger.info(f"Target time: {target_time:.3f}s")
    
    if avg_total <= target_time:
        logger.info("✓ Performance target met!")
    else:
        logger.warning(f"✗ Performance target not met. Need to reduce by {avg_total - target_time:.3f}s")
    
    return results


if __name__ == "__main__":
    results = main()
    sys.exit(0 if results else 1)

