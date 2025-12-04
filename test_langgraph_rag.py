"""
Simple test script for LangGraph RAG implementation.

This script demonstrates basic usage and can be used to verify the implementation works.
"""
import logging
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


def test_basic_query():
    """Test basic query without memory."""
    logger.info("Testing basic query...")
    
    try:
        # Initialize vector store
        vector_store_manager = VectorStoreManager()
        doc_count = vector_store_manager.get_document_count()
        logger.info(f"Vector store initialized with {doc_count} documents")
        
        if doc_count == 0:
            logger.warning("No documents in vector store. Please run ingestion first.")
            return False
        
        # Create RAG graph
        logger.info(f"Creating RAG graph with provider: {Config.LLM_PROVIDER}")
        rag_graph = create_rag_graph(
            vector_store_manager=vector_store_manager,
            provider=Config.LLM_PROVIDER
        )
        
        # Test query
        test_question = "Hver er munurinn á innihaldsefnum í Voltaren forte og voltaren hlaupi?"
        logger.info(f"Querying: {test_question}")
        
        result = query_rag_graph(
            rag_graph=rag_graph,
            question=test_question
        )
        
        # Check results
        if result.get("error"):
            logger.error(f"Error in query: {result['error']}")
            return False
        
        logger.info("Query successful!")
        logger.info(f"Answer: {result['answer'][:200]}...")
        logger.info(f"Sources: {len(result['sources'])}")
        
        if result['sources']:
            logger.info(f"First source: {result['sources'][0].get('drug_id', 'Unknown')}")
        
        return True
    
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return False


def test_with_memory():
    """Test query with conversation memory."""
    logger.info("Testing query with memory...")
    
    try:
        from langchain.memory import ConversationBufferMemory
        
        # Initialize vector store
        vector_store_manager = VectorStoreManager()
        doc_count = vector_store_manager.get_document_count()
        
        if doc_count == 0:
            logger.warning("No documents in vector store. Skipping memory test.")
            return False
        
        # Create memory store
        memory_store = {}
        
        # Create RAG graph with memory
        rag_graph = create_rag_graph(
            vector_store_manager=vector_store_manager,
            provider=Config.LLM_PROVIDER,
            memory_store=memory_store
        )
        
        session_id = "test-session-123"
        
        # First question
        logger.info("First question...")
        result1 = query_rag_graph(
            rag_graph=rag_graph,
            question="Hver er munurinn á innihaldsefnum í Voltaren forte og voltaren hlaupi?",
            session_id=session_id
        )
        
        logger.info(f"Answer 1: {result1['answer'][:100]}...")
        
        # Follow-up question
        logger.info("Follow-up question...")
        result2 = query_rag_graph(
            rag_graph=rag_graph,
            question="Hvað þarf kona með barn á brjósti að hafa í huga þegar hún notar Voriconazole?",
            session_id=session_id
        )
        
        logger.info(f"Answer 2: {result2['answer'][:100]}...")
        
        # Check memory was used
        if session_id in memory_store:
            memory = memory_store[session_id]
            history = memory.load_memory_variables({})
            logger.info(f"Memory has {len(history.get('chat_history', []))} messages")
        
        return True
    
    except Exception as e:
        logger.error(f"Memory test failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("LangGraph RAG Test")
    logger.info("=" * 60)
    
    # Validate configuration
    try:
        Config.validate()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    
    # Run tests
    success = True
    
    logger.info("\n" + "=" * 60)
    success &= test_basic_query()
    
    logger.info("\n" + "=" * 60)
    success &= test_with_memory()
    
    logger.info("\n" + "=" * 60)
    if success:
        logger.info("All tests passed! ✓")
    else:
        logger.error("Some tests failed. ✗")
        sys.exit(1)
