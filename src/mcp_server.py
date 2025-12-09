"""
MCP Server implementation for SmPC document querying.

Provides tools and resources for querying structured SmPC documents
via the Model Context Protocol (MCP) with password protection.
"""
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import Config
from src.vector_store import VectorStoreManager
from src.rag_chain_langgraph import create_rag_graph, query_rag_graph
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

# CORS middleware for web interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security setup
security = HTTPBasic()

# Global state
vector_store_manager: Optional[VectorStoreManager] = None
rag_graph = None
memory_store: Dict[str, ConversationBufferMemory] = {}  # Session-based memory
_warmed_up: bool = False  # Track warmup status


def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    """
    Verify HTTP Basic Auth credentials.
    
    Args:
        credentials: HTTP Basic Auth credentials
        
    Returns:
        Username if valid
        
    Raises:
        HTTPException: If credentials are invalid
    """
    if Config.MCP_AUTH_PASSWORD:
        # Password is set, require authentication
        if (
            credentials.username != Config.MCP_AUTH_USERNAME or
            credentials.password != Config.MCP_AUTH_PASSWORD
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Basic"},
            )
    return credentials.username


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


class ClearSessionRequest(BaseModel):
    """Request model for clear_session tool."""
    session_id: str


# Initialize on startup
@app.on_event("startup")
async def startup_event():
    """Initialize vector store and RAG graph on server startup."""
    global vector_store_manager, rag_graph
    try:
        logger.info("Initializing vector store manager...")
        vector_store_manager = VectorStoreManager()
        doc_count = vector_store_manager.get_document_count()
        logger.info(f"Vector store initialized with {doc_count} documents")
        
        # Initialize RAG graph with langgraph implementation
        logger.info("Initializing RAG graph (langgraph)...")
        rag_graph = create_rag_graph(
            vector_store_manager=vector_store_manager,
            provider=Config.LLM_PROVIDER,
            memory_store=memory_store
        )
        logger.info("RAG graph initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize: {e}", exc_info=True)
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
            return_messages=True,
            input_key="input",
            output_key="output"
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


# Web interface routes
@app.get("/", response_class=HTMLResponse)
async def web_interface():
    """Serve the web interface."""
    html_path = Path(__file__).parent.parent / "web" / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse("Web interface not found. Please ensure web/index.html exists.")


# Mount static files
web_dir = Path(__file__).parent.parent / "web"
if web_dir.exists():
    app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")


# Warmup endpoint
@app.post("/api/warmup")
async def warmup(username: str = Depends(verify_credentials)) -> Dict[str, Any]:
    """
    Warmup endpoint to initialize components and mitigate cold starts.
    
    Args:
        username: Authenticated username (from dependency)
        
    Returns:
        Dictionary with warmup status
    """
    global _warmed_up
    
    if _warmed_up:
        logger.info("System already warmed up, skipping warmup")
        return {
            "status": "already_warmed_up",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    logger.info("Starting warmup process...")
    
    try:
        if not vector_store_manager:
            raise HTTPException(status_code=500, detail="Vector store not initialized")
        
        # Trigger vector store connection by performing a minimal retrieval
        retriever = vector_store_manager.get_retriever()
        try:
            # Perform minimal query to initialize components
            docs = retriever.invoke("test")
            logger.info(f"Warmup: Retrieved {len(docs)} documents")
        except Exception as e:
            logger.warning(f"Warmup retrieval warning: {e}")
        
        # If RAG graph exists, trigger a minimal query to initialize LLM components
        if rag_graph:
            try:
                # Use a minimal query that won't generate a full answer
                result = query_rag_graph(
                    rag_graph=rag_graph,
                    question="test",
                    session_id="warmup_session"
                )
                logger.info("Warmup: RAG graph initialized")
            except Exception as e:
                logger.warning(f"Warmup RAG graph warning: {e}")
        
        _warmed_up = True
        
        # Opik tracing is handled automatically by the RAG graph via OpikTracer
        logger.info("Warmup completed successfully")
        return {
            "status": "warmed_up",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Warmup failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Warmup failed: {str(e)}")


# MCP Tools
@app.post("/api/tools/search_smpc_sections")
async def tool_search_smpc_sections(
    request: SearchRequest,
    username: str = Depends(verify_credentials)
) -> Dict[str, Any]:
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
        
        return {
            "chunks": results,
            "count": len(results)
        }
    
    except Exception as e:
        logger.error(f"Error in search_smpc_sections: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/api/tools/ask_smpc")
async def tool_ask_smpc(
    request: AskRequest,
    username: str = Depends(verify_credentials)
) -> Dict[str, Any]:
    """
    Ask a question about SmPC documents with memory support.
    
    Args:
        request: Question request with optional drug_id and session_id
        username: Authenticated username (from dependency)
        
    Returns:
        Dictionary with answer and sources
    """
    if not vector_store_manager or not rag_graph:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    logger.info(f"Ask request: question='{request.question[:50]}...', drug_id={request.drug_id}, session={request.session_id}")
    
    try:
        # Parse comma-separated drug names if provided
        # If multiple drugs are specified, we'll let the RAG system handle them
        # through the query analysis node which can extract multiple medications
        medication_filter = request.drug_id.strip() if request.drug_id else None
        
        # Query using langgraph implementation
        result = query_rag_graph(
            rag_graph=rag_graph,
            question=request.question,
            session_id=request.session_id,
            medication_filter=medication_filter
        )
        
        # Opik tracing is handled automatically by the RAG graph via OpikTracer
        
        return {
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "similar_drugs": result.get("similar_drugs", []),
            "session_id": request.session_id,
            "error": result.get("error")
        }
    
    except Exception as e:
        logger.error(f"Error in ask_smpc: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/api/tools/list_drugs")
async def tool_list_drugs(username: str = Depends(verify_credentials)) -> Dict[str, Any]:
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


@app.post("/api/tools/get_section")
async def tool_get_section(
    request: GetSectionRequest,
    username: str = Depends(verify_credentials)
) -> Dict[str, Any]:
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


@app.post("/api/tools/clear_session")
async def tool_clear_session(
    request: ClearSessionRequest,
    username: str = Depends(verify_credentials)
) -> Dict[str, Any]:
    """
    Clear conversation memory for a session.
    
    Args:
        request: Request with session_id to clear
        
    Returns:
        Dictionary with success status
    """
    logger.info(f"Clear session request: session_id={request.session_id}")
    
    try:
        # Clear from global memory_store
        if request.session_id in memory_store:
            del memory_store[request.session_id]
            logger.info(f"Cleared memory for session: {request.session_id}")
        
        # Also clear from rag_graph's memory_store if it exists
        if rag_graph and hasattr(rag_graph, 'memory_store'):
            if request.session_id in rag_graph.memory_store:
                del rag_graph.memory_store[request.session_id]
                logger.info(f"Cleared memory from rag_graph for session: {request.session_id}")
        
        return {
            "status": "success",
            "message": f"Session '{request.session_id}' cleared successfully"
        }
    
    except Exception as e:
        logger.error(f"Error in clear_session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to clear session: {str(e)}")


# MCP Resources
@app.get("/api/resources/smpc/{drug_id}")
async def resource_smpc_drug(
    drug_id: str,
    username: str = Depends(verify_credentials)
) -> Dict[str, Any]:
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


@app.get("/api/resources/smpc/{drug_id}/{section_number}")
async def resource_smpc_section(
    drug_id: str,
    section_number: str,
    username: str = Depends(verify_credentials)
) -> Dict[str, Any]:
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


# Health check (public endpoint)
@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "vector_store_initialized": vector_store_manager is not None,
        "rag_graph_initialized": rag_graph is not None,
        "documents_count": vector_store_manager.get_document_count() if vector_store_manager else 0
    }


# MCP Protocol endpoints (protected)
@app.get("/api/mcp/info")
async def mcp_info(username: str = Depends(verify_credentials)) -> Dict[str, Any]:
    """MCP server information."""
    return {
        "name": "smpc-mcp-server",
        "version": "0.0.1",
        "description": "Model Context Protocol vefþjónn fyrir íslenska lyfjatexta (SmPC)",
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
