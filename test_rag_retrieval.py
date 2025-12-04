#!/usr/bin/env python3
"""
Test script for RAG chain retrieval.

This script demonstrates how to:
1. Initialize the vector store manager
2. Create a QA chain
3. Query the RAG chain with test questions
4. Display retrieval results

Usage:
    python test_rag_retrieval.py
    
    # Test with a specific question:
    python test_rag_retrieval.py "Hver er skammturinn fyrir Voltaren?"
    
    # Test with medication filter:
    python test_rag_retrieval.py --medication "Voltaren" "Hver er skammturinn?"
"""
import logging
import sys
import argparse
from typing import Optional

from config import Config
from src.vector_store import VectorStoreManager
from src.rag_chain import create_qa_chain, query_rag

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_vector_store_status(vector_store_manager: VectorStoreManager) -> bool:
    """
    Check if vector store has documents loaded.
    
    Args:
        vector_store_manager: VectorStoreManager instance
        
    Returns:
        True if documents exist, False otherwise
    """
    try:
        doc_count = vector_store_manager.get_document_count()
        logger.info(f"Vector store contains {doc_count} documents")
        
        if doc_count == 0:
            logger.warning(
                "⚠️  Vector store is empty! "
                "You need to ingest documents first using ingest_all_smpcs.py"
            )
            return False
        
        # Show available medications
        medications = vector_store_manager.get_unique_medications()
        if medications:
            logger.info(f"Available medications: {', '.join(medications[:10])}")
            if len(medications) > 10:
                logger.info(f"... and {len(medications) - 10} more")
        
        return True
    except Exception as e:
        logger.error(f"Error checking vector store: {e}")
        return False


def test_retrieval(
    question: str,
    medication_filter: Optional[str] = None,
    provider: Optional[str] = None
) -> None:
    """
    Test RAG chain retrieval with a question.
    
    Args:
        question: Question to ask (in Icelandic)
        medication_filter: Optional medication name to filter retrieval
        provider: LLM provider ("gemini" or "gpt5"), defaults to Config.LLM_PROVIDER
    """
    logger.info("=" * 60)
    logger.info("TESTING RAG CHAIN RETRIEVAL")
    logger.info("=" * 60)
    
    # Step 1: Validate configuration
    logger.info("\n1. Validating configuration...")
    try:
        Config.validate()
        logger.info(f"✓ Configuration valid (Provider: {Config.LLM_PROVIDER})")
    except ValueError as e:
        logger.error(f"✗ Configuration error: {e}")
        logger.error("Please set required API keys in your .env file")
        return
    
    # Step 2: Initialize vector store manager
    logger.info("\n2. Initializing vector store manager...")
    try:
        vector_store_manager = VectorStoreManager()
        logger.info("✓ Vector store manager initialized")
    except Exception as e:
        logger.error(f"✗ Failed to initialize vector store: {e}")
        return
    
    # Step 3: Check if documents are loaded
    logger.info("\n3. Checking vector store status...")
    if not check_vector_store_status(vector_store_manager):
        logger.error(
            "\n❌ Cannot test retrieval - vector store is empty.\n"
            "Please run: python ingest_all_smpcs.py"
        )
        return
    
    # Step 4: Create QA chain
    logger.info("\n4. Creating QA chain...")
    try:
        qa_chain = create_qa_chain(
            vector_store_manager,
            provider=provider,
            medication_filter=medication_filter
        )
        logger.info("✓ QA chain created successfully")
    except Exception as e:
        logger.error(f"✗ Failed to create QA chain: {e}")
        return
    
    # Step 5: Query the RAG chain
    logger.info("\n5. Querying RAG chain...")
    logger.info(f"Question: {question}")
    if medication_filter:
        logger.info(f"Medication filter: {medication_filter}")
    
    try:
        result = query_rag(
            qa_chain,
            question,
            medication_filter=medication_filter
        )
        
        # Step 6: Display results
        logger.info("\n" + "=" * 60)
        logger.info("RETRIEVAL RESULTS")
        logger.info("=" * 60)
        
        print("\n📝 ANSWER:")
        print("-" * 60)
        print(result["answer"])
        print("-" * 60)
        
        print(f"\n📚 SOURCES ({len(result['sources'])} documents retrieved):")
        print("-" * 60)
        for i, source in enumerate(result["sources"], 1):
            print(f"\n{i}. {source.get('medication_name', 'Unknown')}")
            print(f"   Section: {source.get('section_title', 'Unknown')} "
                  f"(#{source.get('section_number', 'Unknown')})")
            print(f"   Source: {source.get('source', 'Unknown')}")
            print(f"   Page: {source.get('page', 'Unknown')}")
            print(f"   Preview: {source.get('text', '')[:150]}...")
        
        logger.info("\n✓ Retrieval test completed successfully")
        
    except Exception as e:
        logger.error(f"✗ Error querying RAG chain: {e}", exc_info=True)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test RAG chain retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with default question
  python test_rag_retrieval.py
  
  # Test with custom question
  python test_rag_retrieval.py "Hver er skammturinn fyrir Voltaren?"
  
  # Test with medication filter
  python test_rag_retrieval.py --medication "Voltaren" "Hver er skammturinn?"
  
  # Test with specific provider
  python test_rag_retrieval.py --provider gemini "Hver er skammturinn?"
        """
    )
    
    parser.add_argument(
        "question",
        nargs="?",
        default="Hver er skammturinn fyrir þessa lyf?",
        help="Question to ask (in Icelandic)"
    )
    
    parser.add_argument(
        "--medication",
        "-m",
        type=str,
        help="Filter retrieval by medication name"
    )
    
    parser.add_argument(
        "--provider",
        "-p",
        choices=["gemini", "gpt5"],
        help="LLM provider to use (defaults to Config.LLM_PROVIDER)"
    )
    
    args = parser.parse_args()
    
    test_retrieval(
        question=args.question,
        medication_filter=args.medication,
        provider=args.provider
    )


if __name__ == "__main__":
    main()
