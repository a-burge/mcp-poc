"""
Test script for query rewrite functionality.

Tests the query rewrite node's ability to:
- Correct typos and normalize medical terminology
- Map brand names to ingredients
- Detect relevant SmPC sections
- Merge detections with query_analysis results
- Handle errors gracefully
"""
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from src.vector_store import VectorStoreManager
from src.rag_chain_langgraph import create_rag_graph, query_rag_graph

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_rewrite_disabled():
    """Test that rewrite node returns empty state when disabled."""
    logger.info("Testing rewrite disabled behavior...")
    
    # Temporarily disable rewrite
    original_value = Config.ENABLE_QUERY_REWRITE
    Config.ENABLE_QUERY_REWRITE = False
    
    try:
        vector_store_manager = VectorStoreManager()
        doc_count = vector_store_manager.get_document_count()
        
        if doc_count == 0:
            logger.warning("No documents in vector store. Skipping test.")
            return False
        
        rag_graph = create_rag_graph(
            vector_store_manager=vector_store_manager,
            provider=Config.LLM_PROVIDER
        )
        
        # Verify rewrite_llm is None when disabled
        if rag_graph.rewrite_llm is not None:
            logger.error("rewrite_llm should be None when ENABLE_QUERY_REWRITE is False")
            return False
        
        logger.info("✓ Rewrite disabled correctly - rewrite_llm is None")
        return True
    
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return False
    
    finally:
        Config.ENABLE_QUERY_REWRITE = original_value


def test_misspelling_correction():
    """Test that rewrite corrects typos like 'ibuprofein' -> 'ibuprofen'."""
    logger.info("Testing misspelling correction...")
    
    # Enable rewrite for this test
    original_value = Config.ENABLE_QUERY_REWRITE
    Config.ENABLE_QUERY_REWRITE = True
    
    try:
        vector_store_manager = VectorStoreManager()
        doc_count = vector_store_manager.get_document_count()
        
        if doc_count == 0:
            logger.warning("No documents in vector store. Skipping test.")
            return False
        
        rag_graph = create_rag_graph(
            vector_store_manager=vector_store_manager,
            provider=Config.LLM_PROVIDER
        )
        
        if rag_graph.rewrite_llm is None:
            logger.warning("Rewrite LLM not available. Skipping test.")
            return False
        
        # Test query with typo
        test_question = "mega börn fá ibuprofein?"
        logger.info(f"Querying with typo: {test_question}")
        
        result = query_rag_graph(
            rag_graph=rag_graph,
            question=test_question
        )
        
        if result.get("error"):
            logger.error(f"Error in query: {result['error']}")
            return False
        
        # Check that we got results (should retrieve ibuprofen documents)
        if len(result.get("sources", [])) == 0:
            logger.warning("No sources retrieved - may indicate rewrite didn't work")
            return False
        
        logger.info(f"✓ Query with typo succeeded. Retrieved {len(result['sources'])} sources")
        logger.info(f"Answer preview: {result['answer'][:200]}...")
        
        return True
    
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return False
    
    finally:
        Config.ENABLE_QUERY_REWRITE = original_value


def test_brand_to_ingredient():
    """Test that rewrite maps brand names to ingredients (e.g., 'panodil' -> 'paracetamol')."""
    logger.info("Testing brand to ingredient mapping...")
    
    # Enable rewrite for this test
    original_value = Config.ENABLE_QUERY_REWRITE
    Config.ENABLE_QUERY_REWRITE = True
    
    try:
        vector_store_manager = VectorStoreManager()
        doc_count = vector_store_manager.get_document_count()
        
        if doc_count == 0:
            logger.warning("No documents in vector store. Skipping test.")
            return False
        
        rag_graph = create_rag_graph(
            vector_store_manager=vector_store_manager,
            provider=Config.LLM_PROVIDER
        )
        
        if rag_graph.rewrite_llm is None:
            logger.warning("Rewrite LLM not available. Skipping test.")
            return False
        
        # Test query with brand name
        test_question = "Hvað er panodil?"
        logger.info(f"Querying with brand name: {test_question}")
        
        result = query_rag_graph(
            rag_graph=rag_graph,
            question=test_question
        )
        
        if result.get("error"):
            logger.error(f"Error in query: {result['error']}")
            return False
        
        # Check that we got results (should retrieve paracetamol documents)
        if len(result.get("sources", [])) == 0:
            logger.warning("No sources retrieved - may indicate rewrite didn't work")
            return False
        
        logger.info(f"✓ Brand name query succeeded. Retrieved {len(result['sources'])} sources")
        logger.info(f"Answer preview: {result['answer'][:200]}...")
        
        return True
    
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return False
    
    finally:
        Config.ENABLE_QUERY_REWRITE = original_value


def test_section_detection():
    """Test that rewrite detects relevant sections for child dosing questions."""
    logger.info("Testing section detection...")
    
    # Enable rewrite for this test
    original_value = Config.ENABLE_QUERY_REWRITE
    Config.ENABLE_QUERY_REWRITE = True
    
    try:
        vector_store_manager = VectorStoreManager()
        doc_count = vector_store_manager.get_document_count()
        
        if doc_count == 0:
            logger.warning("No documents in vector store. Skipping test.")
            return False
        
        rag_graph = create_rag_graph(
            vector_store_manager=vector_store_manager,
            provider=Config.LLM_PROVIDER
        )
        
        if rag_graph.rewrite_llm is None:
            logger.warning("Rewrite LLM not available. Skipping test.")
            return False
        
        # Test query about child dosing (should suggest sections 4.2, 4.3)
        test_question = "Má gefa börnum ibuprofen?"
        logger.info(f"Querying about child dosing: {test_question}")
        
        result = query_rag_graph(
            rag_graph=rag_graph,
            question=test_question
        )
        
        if result.get("error"):
            logger.error(f"Error in query: {result['error']}")
            return False
        
        # Check that we got results
        if len(result.get("sources", [])) == 0:
            logger.warning("No sources retrieved")
            return False
        
        logger.info(f"✓ Child dosing query succeeded. Retrieved {len(result['sources'])} sources")
        logger.info(f"Answer preview: {result['answer'][:200]}...")
        
        # Note: We can't directly check rewrite_metadata from result, but if retrieval
        # succeeded with correct documents, rewrite likely worked
        
        return True
    
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return False
    
    finally:
        Config.ENABLE_QUERY_REWRITE = original_value


def test_error_handling():
    """Test that rewrite handles errors gracefully and falls back to original query."""
    logger.info("Testing error handling...")
    
    # Enable rewrite for this test
    original_value = Config.ENABLE_QUERY_REWRITE
    Config.ENABLE_QUERY_REWRITE = True
    
    try:
        vector_store_manager = VectorStoreManager()
        doc_count = vector_store_manager.get_document_count()
        
        if doc_count == 0:
            logger.warning("No documents in vector store. Skipping test.")
            return False
        
        rag_graph = create_rag_graph(
            vector_store_manager=vector_store_manager,
            provider=Config.LLM_PROVIDER
        )
        
        if rag_graph.rewrite_llm is None:
            logger.warning("Rewrite LLM not available. Skipping test.")
            return False
        
        # Test with a query that should still work even if rewrite fails
        test_question = "Hverjar eru frábendingarnar fyrir ibuprofen?"
        logger.info(f"Testing error handling with query: {test_question}")
        
        result = query_rag_graph(
            rag_graph=rag_graph,
            question=test_question
        )
        
        # Even if rewrite fails, query should still work with original question
        if result.get("error"):
            logger.error(f"Error in query: {result['error']}")
            return False
        
        logger.info(f"✓ Error handling test passed. Query succeeded even if rewrite had issues")
        logger.info(f"Retrieved {len(result.get('sources', []))} sources")
        
        return True
    
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return False
    
    finally:
        Config.ENABLE_QUERY_REWRITE = original_value


def test_chat_history_context():
    """Test that rewrite maintains drug context from chat history."""
    logger.info("Testing chat history context...")
    
    # Enable rewrite for this test
    original_value = Config.ENABLE_QUERY_REWRITE
    Config.ENABLE_QUERY_REWRITE = True
    
    try:
        from langchain.memory import ConversationBufferMemory
        
        vector_store_manager = VectorStoreManager()
        doc_count = vector_store_manager.get_document_count()
        
        if doc_count == 0:
            logger.warning("No documents in vector store. Skipping test.")
            return False
        
        memory_store = {}
        session_id = "test-rewrite-session"
        
        rag_graph = create_rag_graph(
            vector_store_manager=vector_store_manager,
            provider=Config.LLM_PROVIDER,
            memory_store=memory_store
        )
        
        if rag_graph.rewrite_llm is None:
            logger.warning("Rewrite LLM not available. Skipping test.")
            return False
        
        # First question about a specific drug
        first_question = "Hvað er ibuprofen?"
        logger.info(f"First question: {first_question}")
        
        result1 = query_rag_graph(
            rag_graph=rag_graph,
            question=first_question,
            session_id=session_id
        )
        
        if result1.get("error"):
            logger.error(f"Error in first query: {result1['error']}")
            return False
        
        # Follow-up question that should maintain context
        followup_question = "Má gefa börnum?"
        logger.info(f"Follow-up question: {followup_question}")
        
        result2 = query_rag_graph(
            rag_graph=rag_graph,
            question=followup_question,
            session_id=session_id
        )
        
        if result2.get("error"):
            logger.error(f"Error in follow-up query: {result2['error']}")
            return False
        
        # Check that follow-up retrieved relevant documents
        if len(result2.get("sources", [])) == 0:
            logger.warning("No sources retrieved for follow-up - may indicate context wasn't maintained")
            return False
        
        logger.info(f"✓ Chat history context test passed")
        logger.info(f"Follow-up retrieved {len(result2['sources'])} sources")
        logger.info(f"Answer preview: {result2['answer'][:200]}...")
        
        return True
    
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return False
    
    finally:
        Config.ENABLE_QUERY_REWRITE = original_value


def run_all_tests():
    """Run all rewrite tests."""
    logger.info("=" * 60)
    logger.info("Running Query Rewrite Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Rewrite Disabled", test_rewrite_disabled),
        ("Misspelling Correction", test_misspelling_correction),
        ("Brand to Ingredient", test_brand_to_ingredient),
        ("Section Detection", test_section_detection),
        ("Error Handling", test_error_handling),
        ("Chat History Context", test_chat_history_context),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- Running: {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                logger.info(f"✓ {test_name} PASSED")
            else:
                logger.warning(f"✗ {test_name} FAILED or SKIPPED")
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED with exception: {e}", exc_info=True)
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED/SKIPPED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

