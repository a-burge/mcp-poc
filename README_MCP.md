# MCP SmPC Server - Implementation Guide

## Overview

This is a complete MCP (Model Context Protocol) server for Icelandic SmPC (Summary of Product Characteristics) documents with full observability via Opik.

## Architecture

```
PDFs (data/raw_source_docs/)
  ↓
[ingest_all_smpcs.py] → Parse & Validate SmPCs
  ↓
Structured JSON (data/structured/)
  ↓
Chunk & Embed → ChromaDB (data/vector_store/)
  ↓
[MCP Server] → FastAPI + Tools/Resources
  ↓
ChatGPT Developer Mode / Clients
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Ingest SmPC Documents

Place PDF files in `data/raw_source_docs/` and run:

```bash
python ingest_all_smpcs.py
```

This will:
- Scan for PDF files
- Validate they are actual SmPC documents (skips reminder cards, leaflets, etc.)
- Parse into structured JSON
- Chunk sections
- Embed and store in ChromaDB

### 4. Start MCP Server

```bash
python run_mcp_server.py
```

Server will be available at `http://localhost:8000`

## MCP Tools

### 1. `search_smpc_sections`
Vector search for SmPC sections.

**Request:**
```json
{
  "query": "frábendingar",
  "drug_id": "Heparin"  // optional
}
```

### 2. `ask_smpc`
Ask questions with RAG + memory support.

**Request:**
```json
{
  "question": "Hverjar eru frábendingar fyrir Tegretol?",
  "drug_id": "Tegretol",  // optional
  "session_id": "session-123"  // optional, for conversation memory
}
```

### 3. `list_drugs`
List all available drugs in the vector store.

**Request:**
```json
{}
```

### 4. `get_section`
Get raw section text for verification.

**Request:**
```json
{
  "drug_id": "Heparin",
  "section_number": "4.3"
}
```

## MCP Resources

### `/smpc/{drug_id}`
Get drug metadata and version hash.

### `/smpc/{drug_id}/{section_number}`
Get raw section text for sponsor verification.

## Observability

Opik instrumentation is built-in and logs:
- Tool invocations
- Retrieval results (chunks, similarity scores)
- Prompt construction
- LLM calls
- Memory state (before/after queries)
- Final responses

Set `OPIK_API_KEY` in `.env` to enable.

## Testing

### Test Ingestion
```bash
python ingest_all_smpcs.py
```

### Test MCP Server
```bash
# Start server
python run_mcp_server.py

# In another terminal, test endpoints
curl http://localhost:8000/health
curl -X POST http://localhost:8000/tools/list_drugs
```

### Test with ChatGPT Developer Mode

1. Deploy server to Vercel/Replit
2. Update `mcp_manifest.json` with your server URL
3. Add MCP server in ChatGPT Developer Mode
4. Test query: "Hverjar eru frábendingar fyrir Tegretol?"

## Deployment

### Vercel

1. Push to GitHub
2. Import project in Vercel
3. Set environment variables
4. Deploy

The `vercel.json` is configured for Python 3.11.

### Replit

1. Import repository
2. Set environment variables in Secrets
3. Run `python run_mcp_server.py`

## File Structure

```
.
├── ingest_all_smpcs.py      # Main ingestion script
├── run_mcp_server.py        # MCP server runner
├── mcp_manifest.json        # MCP server manifest
├── vercel.json              # Vercel deployment config
├── .env.example             # Environment template
├── config.py                # Configuration
├── requirements.txt         # Dependencies
└── src/
    ├── mcp_server.py        # MCP server implementation
    ├── smpc_parser.py       # SmPC parsing & validation
    ├── chunker.py           # Chunking (includes chunk_smpc_json)
    ├── vector_store.py      # ChromaDB management
    └── rag_chain.py         # RAG chain with memory & Opik
```

## Key Features

✅ Deterministic SmPC parsing with canonical keys
✅ Automatic filtering (skips non-SmPC documents)
✅ Structured JSON storage
✅ Section-based chunking
✅ Persistent vector store (ChromaDB)
✅ MCP tools & resources
✅ Conversational memory (session-based)
✅ Citation enforcement
✅ Full Opik observability
✅ Icelandic language optimized

## Next Steps

1. Run ingestion on your PDF files
2. Test MCP server locally
3. Deploy to Vercel/Replit
4. Connect to ChatGPT Developer Mode
5. Verify Opik traces in dashboard
