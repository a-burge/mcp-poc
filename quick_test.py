#!/usr/bin/env python3
"""
Quick test script to check system readiness and run basic tests.

This script:
1. Validates configuration
2. Checks vector store status
3. Runs a quick RAG test if documents are available
4. Provides clear next steps
"""
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from src.vector_store import VectorStoreManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_configuration() -> bool:
    """Check if configuration is valid."""
    logger.info("Checking configuration...")
    try:
        Config.validate()
        logger.info(f"✓ Configuration valid (Provider: {Config.LLM_PROVIDER})")
        return True
    except ValueError as e:
        logger.error(f"✗ Configuration error: {e}")
        logger.error("Please check your .env file and ensure API keys are set.")
        return False


def check_vector_store() -> tuple[bool, int, list[str]]:
    """Check vector store status."""
    logger.info("Checking vector store...")
    try:
        vector_store = VectorStoreManager()
        doc_count = vector_store.get_document_count()
        medications = vector_store.get_unique_medications()
        
        if doc_count == 0:
            logger.warning("⚠️  Vector store is empty")
            logger.info("   Next step: Run 'python3 ingest_all_smpcs.py' to ingest documents")
            return False, 0, []
        
        logger.info(f"✓ Vector store has {doc_count} documents")
        if medications:
            logger.info(f"   Available medications: {', '.join(medications[:5])}")
            if len(medications) > 5:
                logger.info(f"   ... and {len(medications) - 5} more")
        return True, doc_count, medications
    
    except Exception as e:
        logger.error(f"✗ Error checking vector store: {e}")
        return False, 0, []


def run_quick_test() -> bool:
    """Run a quick RAG test."""
    logger.info("\n" + "=" * 60)
    logger.info("Running quick RAG test...")
    logger.info("=" * 60)
    
    try:
        from src.rag_chain_langgraph import create_rag_graph, query_rag_graph
        
        # Initialize
        vector_store_manager = VectorStoreManager()
        logger.info("Creating RAG graph...")
        rag_graph = create_rag_graph(
            vector_store_manager=vector_store_manager,
            provider=Config.LLM_PROVIDER
        )
        
        # Test query
        test_question = "Hver er munurinn á innihaldsefnum í Voltaren forte og voltaren hlaupi?"
        logger.info(f"Query: {test_question}")
        
        result = query_rag_graph(
            rag_graph=rag_graph,
            question=test_question
        )
        
        if result.get("error"):
            logger.error(f"✗ Query failed: {result['error']}")
            return False
        
        logger.info("✓ Query successful!")
        logger.info(f"   Answer preview: {result['answer'][:150]}...")
        logger.info(f"   Sources: {len(result['sources'])}")
        
        if result['sources']:
            first_source = result['sources'][0]
            logger.info(f"   First source: {first_source.get('drug_id', 'Unknown')} - {first_source.get('section_title', 'Unknown section')}")
        
        return True
    
    except Exception as e:
        logger.error(f"✗ Test failed: {e}", exc_info=True)
        return False


def main():
    """Main test function."""
    logger.info("=" * 60)
    logger.info("SmPC RAG System - Quick Test")
    logger.info("=" * 60)
    logger.info("")
    
    # Step 1: Check configuration
    config_ok = check_configuration()
    if not config_ok:
        logger.error("\n❌ Configuration check failed. Please fix configuration before proceeding.")
        sys.exit(1)
    
    logger.info("")
    
    # Step 2: Check vector store
    store_ok, doc_count, medications = check_vector_store()
    
    logger.info("")
    
    # Step 3: Run test if documents are available
    if store_ok and doc_count > 0:
        test_ok = run_quick_test()
        
        logger.info("")
        logger.info("=" * 60)
        if test_ok:
            logger.info("✅ All checks passed! System is ready to use.")
            logger.info("")
            logger.info("Next steps:")
            logger.info("  1. Run full test suite: python3 test_langgraph_rag.py")
            logger.info("  2. Start MCP server: python3 run_mcp_server.py")
            logger.info("  3. Start Streamlit UI: streamlit run src/streamlit_app.py")
        else:
            logger.error("❌ Quick test failed. Check logs above for details.")
            sys.exit(1)
    else:
        logger.info("=" * 60)
        logger.warning("⚠️  Vector store is empty. Cannot run RAG test.")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Ensure PDF files are in: data/raw_source_docs/")
        logger.info("  2. Run ingestion: python3 ingest_all_smpcs.py")
        logger.info("  3. Then run this script again to test")


if __name__ == "__main__":
    main()
