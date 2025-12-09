"""
Test script to verify follow-up question context is maintained.
Tests the scenario where a second question should maintain medication context from the first.
"""
import os
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

import logging
import sys
import uuid
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from src.vector_store import VectorStoreManager
from src.rag_chain_langgraph import create_rag_graph, query_rag_graph

# Configure logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

def test_followup_question():
    """Test that follow-up questions maintain medication context."""
    print("=" * 80)
    print("Testing Follow-up Question Context")
    print("=" * 80)
    print()
    
    # Initialize
    print("Initializing RAG system...")
    vector_store_manager = VectorStoreManager()
    memory_store = {}
    session_id = str(uuid.uuid4())
    
    rag_graph = create_rag_graph(
        vector_store_manager=vector_store_manager,
        provider=Config.LLM_PROVIDER,
        memory_store=memory_store
    )
    
    print(f"✓ RAG system initialized")
    print(f"✓ Session ID: {session_id}")
    print()
    
    # First question
    print("=" * 80)
    print("QUESTION 1: er í lagi að gefa barni íbúfen?")
    print("=" * 80)
    print()
    
    result1 = query_rag_graph(
        rag_graph=rag_graph,
        question="er í lagi að gefa barni íbúfen?",
        session_id=session_id
    )
    
    print("ANSWER 1:")
    print(result1.get("answer", "")[:200] + "..." if len(result1.get("answer", "")) > 200 else result1.get("answer", ""))
    print()
    print(f"Sources: {len(result1.get('sources', []))}")
    print()
    
    # Check if medication was extracted
    print("Checking memory state...")
    if session_id in memory_store:
        memory = memory_store[session_id]
        memory_vars = memory.load_memory_variables({})
        history = memory_vars.get("chat_history", memory_vars.get("history", []))
        print(f"✓ Memory has {len(history)} messages")
        if history:
            for i, msg in enumerate(history):
                msg_type = type(msg).__name__
                content_preview = str(msg.content)[:50] + "..." if len(str(msg.content)) > 50 else str(msg.content)
                print(f"  Message {i+1}: {msg_type} - {content_preview}")
    else:
        print("✗ Session not found in memory store!")
    print()
    
    # Second question (follow-up)
    print("=" * 80)
    print("QUESTION 2: en fyrir 14 ára ungling?")
    print("=" * 80)
    print()
    print("Expected: Should extract 'íbúfen' from chat history and maintain context")
    print()
    
    result2 = query_rag_graph(
        rag_graph=rag_graph,
        question="en fyrir 14 ára ungling?",
        session_id=session_id
    )
    
    print("ANSWER 2:")
    print(result2.get("answer", "")[:200] + "..." if len(result2.get("answer", "")) > 200 else result2.get("answer", ""))
    print()
    print(f"Sources: {len(result2.get('sources', []))}")
    print()
    
    # Verify the answer is about ibuprofen, not random medications
    answer2 = result2.get("answer", "").lower()
    sources2 = result2.get("sources", [])
    
    # Check if answer mentions ibuprofen-related terms
    ibuprofen_terms = ["íbúfen", "íbúprófen", "ibuprofen"]
    mentions_ibuprofen = any(term in answer2 for term in ibuprofen_terms)
    
    # Check if sources are about ibuprofen
    ibuprofen_sources = []
    for source in sources2:
        drug_id = source.get("drug_id", "").lower()
        if any(term in drug_id for term in ibuprofen_terms):
            ibuprofen_sources.append(source.get("drug_id"))
    
    print("=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    print(f"Answer mentions ibuprofen: {mentions_ibuprofen}")
    print(f"Ibuprofen-related sources: {len(ibuprofen_sources)}")
    if ibuprofen_sources:
        print(f"  Sources: {ibuprofen_sources[:5]}")  # Show first 5
    
    if mentions_ibuprofen or len(ibuprofen_sources) > 0:
        print()
        print("✓ SUCCESS: Follow-up question maintained context!")
    else:
        print()
        print("✗ FAILURE: Follow-up question lost context!")
        print("   The answer should be about ibuprofen for 14-year-olds")
    
    print()
    print("=" * 80)

if __name__ == "__main__":
    try:
        test_followup_question()
    except Exception as e:
        logger.exception("Test failed with error")
        sys.exit(1)


