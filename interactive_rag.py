"""
Interactive RAG testing tool.

This script provides an interactive command-line interface to test the RAG system
with real queries. It supports conversation memory and displays answers with sources.
"""
import os

# Disable ChromaDB telemetry before any imports to prevent PostHog compatibility errors
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

import argparse
import logging
import sys
import uuid
from pathlib import Path
from typing import Optional, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from src.vector_store import VectorStoreManager
from src.rag_chain_langgraph import create_rag_graph, query_rag_graph
from src.rag_chain_and_graph import SmPCRAGGraph

# Logger will be configured after parsing arguments
logger = logging.getLogger(__name__)


class InteractiveRAG:
    """Interactive RAG testing interface."""
    
    def __init__(self, implementation: str = "langgraph"):
        """
        Initialize the RAG system.
        
        Args:
            implementation: Which RAG implementation to use ("langgraph" or "and_graph")
        """
        print("Initializing RAG system...")
        
        # Validate configuration
        try:
            Config.validate()
        except ValueError as e:
            print(f"❌ Configuration error: {e}")
            sys.exit(1)
        
        # Initialize vector store
        self.vector_store_manager = VectorStoreManager()
        doc_count = self.vector_store_manager.get_document_count()
        
        if doc_count == 0:
            print("❌ No documents found in vector store.")
            print("   Please run ingest_all_smpcs.py first.")
            sys.exit(1)
        
        print(f"✓ Vector store initialized with {doc_count} documents")
        
        # Initialize memory store for conversation history
        self.memory_store: dict[str, any] = {}
        # Generate default session_id to ensure memory works automatically
        self.current_session_id: Optional[str] = str(uuid.uuid4())
        self.implementation = implementation
        
        # Create RAG graph based on selected implementation
        if implementation == "langgraph":
            print(f"✓ Creating RAG graph (langgraph) with provider: {Config.LLM_PROVIDER}")
            self.rag_graph = create_rag_graph(
                vector_store_manager=self.vector_store_manager,
                provider=Config.LLM_PROVIDER,
                memory_store=self.memory_store
            )
            self._query_func = self._query_langgraph
        elif implementation == "and_graph":
            print(f"✓ Creating RAG graph (and_graph) with model: gpt-4o-mini")
            self.rag_graph = SmPCRAGGraph(
                vector_store_manager=self.vector_store_manager,
                memory_store=self.memory_store,
                model_name="gpt-4o-mini"
            )
            self._query_func = self._query_and_graph
        else:
            print(f"❌ Unknown implementation: {implementation}")
            print("   Supported implementations: 'langgraph', 'and_graph'")
            sys.exit(1)
        
        print("✓ RAG system ready!\n")
    
    def _query_langgraph(self, question: str, session_id: Optional[str]) -> Dict[str, Any]:
        """Query using langgraph implementation."""
        return query_rag_graph(
            rag_graph=self.rag_graph,
            question=question,
            session_id=session_id
        )
    
    def _query_and_graph(self, question: str, session_id: Optional[str]) -> Dict[str, Any]:
        """Query using and_graph implementation."""
        result = self.rag_graph.process(
            question=question,
            session_id=session_id
        )
        # Normalize return format to match langgraph
        if "error" not in result:
            result["error"] = None
        
        # Convert Document objects to dictionaries for consistent formatting
        if "sources" in result and result["sources"]:
            normalized_sources = []
            for doc in result["sources"]:
                if hasattr(doc, 'metadata'):  # It's a Document object
                    metadata = doc.metadata
                    normalized_sources.append({
                        "drug_id": metadata.get("drug_id", metadata.get("medication_name", "Unknown")),
                        "section_number": metadata.get("section_number", "Unknown"),
                        "section_title": metadata.get("section_title", metadata.get("section", "Unknown")),
                        "text": doc.page_content,
                        "chunk_text": doc.page_content,
                    })
                else:  # Already a dictionary
                    normalized_sources.append(doc)
            result["sources"] = normalized_sources
        
        return result
    
    def print_answer(self, result: dict) -> None:
        """Print the answer and sources in a formatted way."""
        if result.get("error"):
            print(f"\n❌ Error: {result['error']}\n")
            return
        
        answer = result.get("answer", "")
        sources = result.get("sources", [])
        
        print("\n" + "=" * 80)
        print("ANSWER:")
        print("=" * 80)
        print(answer)
        print()
        
        if sources:
            print("=" * 80)
            print(f"SOURCES ({len(sources)}):")
            print("=" * 80)
            for i, source in enumerate(sources, 1):
                drug_id = source.get("drug_id", "Unknown")
                section_number = source.get("section_number", "Unknown")
                section_title = source.get("section_title", "")
                
                print(f"{i}. [{drug_id}] Section {section_number}: {section_title}")
                # The RAG chain returns "text" for chunk content
                chunk_text = source.get("text") or source.get("chunk_text")
                if chunk_text:
                    # Show first 150 chars of chunk
                    chunk_preview = chunk_text[:150]
                    if len(chunk_text) > 150:
                        chunk_preview += "..."
                    print(f"   Preview: {chunk_preview}")
        else:
            print("(No sources found)")
        
        print("=" * 80 + "\n")
    
    def run(self) -> None:
        """Run the interactive loop."""
        print("=" * 80)
        print("Interactive RAG Testing")
        print("=" * 80)
        print("\nCommands:")
        print("  - Type your question and press Enter")
        print("  - Type '/new' to start a new conversation (clear memory)")
        print("  - Type '/session <id>' to switch to a specific session")
        print("  - Type '/quit' or '/exit' to exit")
        print("  - Type '/help' to show this help message")
        print("\n" + "=" * 80 + "\n")
        
        while True:
            try:
                # Get user input
                question = input("❓ Question: ").strip()
                
                if not question:
                    continue
                
                # Handle commands
                if question.lower() in ["/quit", "/exit", "/q"]:
                    print("\n👋 Goodbye!")
                    break
                
                if question.lower() == "/help":
                    print("\nCommands:")
                    print("  /new          - Start a new conversation (clear memory)")
                    print("  /session <id> - Switch to a specific session")
                    print("  /quit, /exit  - Exit the program")
                    print("  /help         - Show this help message\n")
                    continue
                
                if question.lower() == "/new":
                    self.current_session_id = str(uuid.uuid4())
                    print("✓ Started new conversation (new session ID generated)\n")
                    continue
                
                if question.lower().startswith("/session "):
                    session_id = question.split(" ", 1)[1].strip()
                    self.current_session_id = session_id if session_id else str(uuid.uuid4())
                    print(f"✓ Switched to session: {self.current_session_id}\n")
                    continue
                
                # Ensure session_id is set before processing query
                if not self.current_session_id:
                    self.current_session_id = str(uuid.uuid4())
                    logger.info(f"Generated new session_id: {self.current_session_id}")
                
                # Process query
                print("\n🔍 Processing query...")
                
                result = self._query_func(
                    question=question,
                    session_id=self.current_session_id
                )
                
                # Display result
                self.print_answer(result)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                logger.exception("Error in interactive loop")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactive RAG testing tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python interactive_rag.py                    # Use langgraph (default)
  python interactive_rag.py --impl langgraph  # Use langgraph explicitly
  python interactive_rag.py --impl and_graph  # Use and_graph implementation
  python interactive_rag.py --verbose          # Enable verbose logging (INFO level)
  python interactive_rag.py --impl and_graph --verbose  # Combine flags
        """
    )
    parser.add_argument(
        "--impl",
        "--implementation",
        dest="implementation",
        choices=["langgraph", "and_graph"],
        default="langgraph",
        help="RAG implementation to use (default: langgraph)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging (INFO level). Shows memory loading, state transitions, and debug information."
    )
    
    args = parser.parse_args()
    
    # Configure logging based on verbose flag
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(levelname)s: %(message)s'
    )
    
    if args.verbose:
        logger.info("Verbose logging enabled - you will see detailed memory and state information")
    
    try:
        app = InteractiveRAG(implementation=args.implementation)
        app.run()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        logger.exception("Fatal error")
        sys.exit(1)
