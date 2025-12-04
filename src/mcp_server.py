"""
MCP Server implementation for SmPC document querying.

Provides tools and resources for querying structured SmPC documents
via the Model Context Protocol (MCP).
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import Config
from src.vector_store import VectorStoreManager
from src.rag_chain import create_qa_chain, query_rag, create_llm
from langchain.memory import ConversationBufferMemory

# Try to import opik for instrumentation
try:
    import opik
    OPIK_AVAILABLE = True
except ImportError:
    OPIK_AVAILABLE = False

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="SmPC MCP Server",
    description="Model Context Protocol server for Icelandic SmPC documents",
    version="1.0.0"
)

# Global state
vector_store_manager: Optional[VectorStoreManager] = None
memory_store: Dict[str, ConversationBufferMemory] = {}  # Session-based memory


# Pydantic models for request/response
class SearchRequest(BaseModel):
    """Request model for search_smpc_sections tool."""
    query: str
    drug_id: Optional[str] = None


class AskRequest(BaseModel):
    """Request model for ask_smpc tool."""
    question: str
    drug_id: Optional[str] = None
    session_id: Optional[str] = "default"


class GetSectionRequest(BaseModel):
    """Request model for get_section tool."""
    drug_id: str
    section_number: str


# Initialize on startup
@app.on_event("startup")
async def startup_event():
    """Initialize vector store on server startup."""
    global vector_store_manager
    try:
        logger.info("Initializing vector store manager...")
        vector_store_manager = VectorStoreManager()
        doc_count = vector_store_manager.get_document_count()
        logger.info(f"Vector store initialized with {doc_count} documents")
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {e}", exc_info=True)
        raise


def get_memory(session_id: str) -> ConversationBufferMemory:
    """
    Get or create memory for a session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        ConversationBufferMemory instance for the session
    """
    if session_id not in memory_store:
        memory_store[session_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    return memory_store[session_id]


def load_structured_json(drug_id: str) -> Optional[Dict[str, Any]]:
    """
    Load structured JSON for a drug.
    
    Args:
        drug_id: Drug identifier
        
    Returns:
        Structured JSON data or None if not found
    """
    json_path = Config.STRUCTURED_DIR / f"{drug_id}_SmPC.json"
    if not json_path.exists():
        return None
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON for {drug_id}: {e}")
        return None


# MCP Tools
@app.post("/tools/search_smpc_sections")
async def tool_search_smpc_sections(request: SearchRequest) -> Dict[str, Any]:
    """
    Search SmPC sections using vector search.
    
    Args:
        request: Search request with query and optional drug_id filter
        
    Returns:
        Dictionary with chunks and metadata
    """
    if not vector_store_manager:
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    
    logger.info(f"Search request: query='{request.query[:50]}...', drug_id={request.drug_id}")
    
    # Opik instrumentation
    if OPIK_AVAILABLE:
        opik.log_event("mcp_tool_invoke", {
            "tool": "search_smpc_sections",
            "query": request.query,
            "drug_id": request.drug_id,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    try:
        # Get retriever with optional filter
        if request.drug_id:
            retriever = vector_store_manager.get_retriever_with_filter(
                medication_name=request.drug_id
            )
        else:
            retriever = vector_store_manager.get_retriever()
        
        # Perform search (use invoke() for LangChain 0.3.x compatibility)
        try:
            docs = retriever.invoke(request.query)
        except AttributeError:
            # Fallback for older LangChain versions
            docs = retriever.get_relevant_documents(request.query)
        
        # Format results
        results = []
        for doc in docs:
            metadata = doc.metadata
            results.append({
                "text": doc.page_content,
                "drug_id": metadata.get("drug_id", metadata.get("medication_name", "Unknown")),
                "section_number": metadata.get("section_number", "Unknown"),
                "section_title": metadata.get("section_title", metadata.get("section", "Unknown")),
                "canonical_key": metadata.get("canonical_key", "Unknown"),
                "version_hash": metadata.get("version_hash", "Unknown"),
                "similarity_score": getattr(doc, "score", None)
            })
        
        # Opik instrumentation
        if OPIK_AVAILABLE:
            opik.log_event("mcp_tool_complete", {
                "tool": "search_smpc_sections",
                "chunks_retrieved": len(results),
                "timestamp": datetime.utcnow().isoformat()
            })
        
        return {
            "chunks": results,
            "count": len(results)
        }
    
    except Exception as e:
        logger.error(f"Error in search_smpc_sections: {e}", exc_info=True)
        if OPIK_AVAILABLE:
            opik.log_event("mcp_tool_error", {
                "tool": "search_smpc_sections",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/tools/ask_smpc")
async def tool_ask_smpc(request: AskRequest) -> Dict[str, Any]:
    """
    Ask a question about SmPC documents with memory support.
    
    Args:
        request: Question request with optional drug_id and session_id
        
    Returns:
        Dictionary with answer and sources
    """
    if not vector_store_manager:
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    
    logger.info(f"Ask request: question='{request.question[:50]}...', drug_id={request.drug_id}, session={request.session_id}")
    
    # Get memory for session
    memory = get_memory(request.session_id)
    
    # Opik instrumentation: Log memory state before query
    if OPIK_AVAILABLE:
        memory_vars = memory.load_memory_variables({})
        chat_history = memory_vars.get("chat_history", [])
        opik.log_event("mcp_tool_invoke", {
            "tool": "ask_smpc",
            "question": request.question,
            "drug_id": request.drug_id,
            "session_id": request.session_id,
            "memory_history_length": len(chat_history),
            "timestamp": datetime.utcnow().isoformat()
        })
    
    try:
        # Create QA chain with memory
        qa_chain = create_qa_chain(
            vector_store_manager,
            provider=None,  # Use default from config
            medication_filter=request.drug_id,
            memory=memory
        )
        
        # Query RAG
        result = query_rag(
            qa_chain,
            request.question,
            medication_filter=request.drug_id,
            memory=memory
        )
        
        # Update memory
        memory.save_context(
            {"input": request.question},
            {"output": result["answer"]}
        )
        
        # Opik instrumentation: Log memory state after query
        if OPIK_AVAILABLE:
            memory_vars_after = memory.load_memory_variables({})
            chat_history_after = memory_vars_after.get("chat_history", [])
            opik.log_event("mcp_tool_complete", {
                "tool": "ask_smpc",
                "answer_length": len(result["answer"]),
                "sources_count": len(result["sources"]),
                "memory_history_length_after": len(chat_history_after),
                "timestamp": datetime.utcnow().isoformat()
            })
        
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "session_id": request.session_id
        }
    
    except Exception as e:
        logger.error(f"Error in ask_smpc: {e}", exc_info=True)
        if OPIK_AVAILABLE:
            opik.log_event("mcp_tool_error", {
                "tool": "ask_smpc",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/tools/list_drugs")
async def tool_list_drugs() -> Dict[str, Any]:
    """
    List all available drugs in the vector store.
    
    Returns:
        Dictionary with list of drugs and their metadata
    """
    if not vector_store_manager:
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    
    logger.info("Listing all drugs")
    
    try:
        # Get unique documents
        documents = vector_store_manager.get_unique_documents()
        
        # Extract drug metadata
        drugs = []
        seen_drugs = set()
        
        for doc in documents:
            drug_id = doc.get("medication_name", "Unknown")
            if drug_id not in seen_drugs:
                # Try to load structured JSON for version_hash and extracted_at
                smpc_data = load_structured_json(drug_id)
                
                drug_info = {
                    "drug_id": drug_id,
                    "version_hash": smpc_data.get("version_hash", "") if smpc_data else "",
                    "extracted_at": smpc_data.get("extracted_at", "") if smpc_data else "",
                    "chunk_count": doc.get("chunk_count", 0)
                }
                drugs.append(drug_info)
                seen_drugs.add(drug_id)
        
        return {
            "drugs": drugs,
            "count": len(drugs)
        }
    
    except Exception as e:
        logger.error(f"Error in list_drugs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list drugs: {str(e)}")


@app.post("/tools/get_section")
async def tool_get_section(request: GetSectionRequest) -> Dict[str, Any]:
    """
    Get raw text for a specific section from structured JSON.
    
    Args:
        request: Request with drug_id and section_number
        
    Returns:
        Dictionary with section text and metadata
    """
    logger.info(f"Get section request: drug_id={request.drug_id}, section={request.section_number}")
    
    try:
        # Load structured JSON
        smpc_data = load_structured_json(request.drug_id)
        if not smpc_data:
            raise HTTPException(
                status_code=404,
                detail=f"Drug '{request.drug_id}' not found in structured data"
            )
        
        # Get section
        sections = smpc_data.get("sections", {})
        section_data = sections.get(request.section_number)
        
        if not section_data:
            raise HTTPException(
                status_code=404,
                detail=f"Section '{request.section_number}' not found for drug '{request.drug_id}'"
            )
        
        return {
            "drug_id": request.drug_id,
            "section_number": request.section_number,
            "section_title": section_data.get("title", ""),
            "canonical_key": section_data.get("canonical_key", ""),
            "text": section_data.get("text", ""),
            "version_hash": smpc_data.get("version_hash", ""),
            "extracted_at": smpc_data.get("extracted_at", "")
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_section: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get section: {str(e)}")


# MCP Resources
@app.get("/resources/smpc/{drug_id}")
async def resource_smpc_drug(drug_id: str) -> Dict[str, Any]:
    """
    Get metadata for a specific drug.
    
    Args:
        drug_id: Drug identifier
        
    Returns:
        Dictionary with drug metadata
    """
    smpc_data = load_structured_json(drug_id)
    if not smpc_data:
        raise HTTPException(
            status_code=404,
            detail=f"Drug '{drug_id}' not found"
        )
    
    return {
        "drug_id": smpc_data.get("drug_id", drug_id),
        "version_hash": smpc_data.get("version_hash", ""),
        "extracted_at": smpc_data.get("extracted_at", ""),
        "source_pdf": smpc_data.get("source_pdf", ""),
        "sections_count": len(smpc_data.get("sections", {}))
    }


@app.get("/resources/smpc/{drug_id}/{section_number}")
async def resource_smpc_section(drug_id: str, section_number: str) -> Dict[str, Any]:
    """
    Get raw text for a specific section (for sponsor verification).
    
    Args:
        drug_id: Drug identifier
        section_number: Section number (e.g., "4.3")
        
    Returns:
        Dictionary with section text and metadata
    """
    smpc_data = load_structured_json(drug_id)
    if not smpc_data:
        raise HTTPException(
            status_code=404,
            detail=f"Drug '{drug_id}' not found"
        )
    
    sections = smpc_data.get("sections", {})
    section_data = sections.get(section_number)
    
    if not section_data:
        raise HTTPException(
            status_code=404,
            detail=f"Section '{section_number}' not found for drug '{drug_id}'"
        )
    
    return {
        "drug_id": drug_id,
        "section_number": section_number,
        "section_title": section_data.get("title", ""),
        "canonical_key": section_data.get("canonical_key", ""),
        "text": section_data.get("text", ""),
        "version_hash": smpc_data.get("version_hash", ""),
        "extracted_at": smpc_data.get("extracted_at", "")
    }


# Health check
@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "vector_store_initialized": vector_store_manager is not None,
        "documents_count": vector_store_manager.get_document_count() if vector_store_manager else 0
    }


# MCP Protocol endpoints
@app.get("/mcp/info")
async def mcp_info() -> Dict[str, Any]:
    """MCP server information."""
    return {
        "name": "smpc-mcp-server",
        "version": "1.0.0",
        "description": "Model Context Protocol server for Icelandic SmPC documents",
        "tools": [
            "search_smpc_sections",
            "ask_smpc",
            "list_drugs",
            "get_section"
        ],
        "resources": [
            "/smpc/{drug_id}",
            "/smpc/{drug_id}/{section_number}"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=Config.MCP_SERVER_HOST,
        port=Config.MCP_SERVER_PORT
    )
