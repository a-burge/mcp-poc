# Testing Guide - SmPC RAG System

This guide walks you through testing the SmPC RAG system step by step.

## Prerequisites Check

### 1. Verify Python Version
```bash
python3 --version  # Should be 3.11 or higher
```

### 2. Activate Virtual Environment
```bash
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Environment Configuration
Check that your `.env` file has the required API keys:
```bash
# Check if .env exists and has required keys
cat .env | grep -E "(LLM_PROVIDER|GOOGLE_API_KEY|OPENAI_API_KEY)"
```

Required variables:
- `LLM_PROVIDER`: Either `"gemini"` or `"gpt5"`
- `GOOGLE_API_KEY`: Required if using Gemini
- `OPENAI_API_KEY`: Required if using GPT-5

## Step 1: Check Current State

### Check if Documents are Already Ingested
```bash
python3 -c "from src.vector_store import VectorStoreManager; vs = VectorStoreManager(); print(f'Documents: {vs.get_document_count()}'); print(f'Medications: {vs.get_unique_medications()}')"
```

**What to look for:**
- If `Documents: 0`, you need to ingest documents (Step 2)
- If documents exist, you can skip to Step 3

### Check Available PDF Files
```bash
ls -la data/raw_source_docs/*.pdf | grep -i smpc
```

**What to look for:**
- PDF files with "SMPC" in the filename are ready for ingestion
- Files without "SMPC" are typically leaflets/reminders and will be skipped

## Step 2: Ingest Documents (If Needed)

### Option A: Ingest All SmPC PDFs
```bash
python3 ingest_all_smpcs.py
```

### Option B: Dry Run (Validate Without Indexing)
```bash
python3 ingest_all_smpcs.py --dry-run
```

### Option C: Ingest Specific Number of Files
```bash
python3 ingest_all_smpcs.py --max-files 5
```

### Option D: Clear and Re-ingest Everything
```bash
python3 ingest_all_smpcs.py --clear-existing
```

**What happens:**
1. Scans `data/raw_source_docs/` for PDF files with "SMPC" in filename
2. Parses each PDF using `smpc_parser.py`
3. Validates SmPC structure
4. Saves structured JSON to `data/structured/`
5. Chunks and indexes documents in vector store

**Expected output:**
```
Ingestion Summary:
  Successful: X
  Skipped (not SmPC): Y
  Failed: Z
Total documents in vector store: X
```

## Step 3: Test the RAG Chain

### Quick Test (Basic Query)
```bash
python3 test_langgraph_rag.py
```

This runs two tests:
1. **Basic query** - Single question without memory
2. **Memory test** - Conversation with follow-up questions

**What to look for:**
- ✅ "Query successful!" message
- ✅ Answer preview (first 200 chars)
- ✅ Number of sources retrieved
- ✅ "All tests passed! ✓"

### Test with Custom Questions

You can modify `test_questions.md` or create your own test script:

```python
from src.vector_store import VectorStoreManager
from src.rag_chain_langgraph import create_rag_graph, query_rag_graph
from config import Config

# Initialize
vector_store_manager = VectorStoreManager()
rag_graph = create_rag_graph(
    vector_store_manager=vector_store_manager,
    provider=Config.LLM_PROVIDER
)

# Query
result = query_rag_graph(
    rag_graph=rag_graph,
    question="Hver er munurinn á innihaldsefnum í Voltaren forte og voltaren hlaupi?"
)

print(result["answer"])
print(f"Sources: {len(result['sources'])}")
```

### Test with Medication Filtering

```python
# Filter to specific medication
rag_graph = create_rag_graph(
    vector_store_manager=vector_store_manager,
    provider=Config.LLM_PROVIDER,
    medication_filter="Voltaren"
)

result = query_rag_graph(
    rag_graph=rag_graph,
    question="Hverjar eru frábendingarnar?"
)
```

## Step 4: Test the MCP Server (API)

### Start the Server
```bash
python3 run_mcp_server.py
```

Or with custom host/port:
```bash
python3 run_mcp_server.py --host 0.0.0.0 --port 8000
```

**Expected output:**
```
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Test API Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Ask a Question
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Hver er munurinn á innihaldsefnum í Voltaren forte og voltaren hlaupi?",
    "session_id": "test-123"
  }'
```

#### Ask with Medication Filter
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Hverjar eru frábendingarnar?",
    "drug_id": "Voltaren",
    "session_id": "test-123"
  }'
```

#### List Available Medications
```bash
curl http://localhost:8000/medications
```

#### Get Document Statistics
```bash
curl http://localhost:8000/stats
```

## Step 5: Test Streamlit UI (Optional)

### Start Streamlit App
```bash
streamlit run src/streamlit_app.py
```

The app will open at `http://localhost:8501`

**Features:**
- Process PDFs from URL
- Ask questions interactively
- View source citations
- See conversation history

## Troubleshooting

### Issue: "No documents in vector store"
**Solution:** Run ingestion (Step 2)

### Issue: "Configuration error: API key required"
**Solution:** Check your `.env` file has the correct API key for your chosen provider

### Issue: "ModuleNotFoundError"
**Solution:** 
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: "Invalid SmPC structure" during ingestion
**Explanation:** The file is not a valid SmPC document (might be a leaflet or reminder card)
**Solution:** This is expected - only files with "SMPC" in filename are processed

### Issue: Slow first query
**Explanation:** First query loads the embedding model (can take 10-30 seconds)
**Solution:** This is normal - subsequent queries are faster

## Quick Test Checklist

- [ ] Python 3.11+ installed
- [ ] Virtual environment activated
- [ ] Dependencies installed
- [ ] `.env` file configured with API keys
- [ ] Documents ingested (check with `get_document_count()`)
- [ ] Basic RAG test passes (`test_langgraph_rag.py`)
- [ ] MCP server starts successfully
- [ ] API endpoints respond correctly

## Next Steps After Testing

1. **Compare Providers:** Test with both Gemini and GPT-5 to compare results
2. **Add More Documents:** Ingest additional SmPC PDFs
3. **Customize Prompts:** Modify prompts in `rag_chain_langgraph.py` for your use case
4. **Enable Re-ranking:** Set `ENABLE_RERANKING=true` in `.env` for better accuracy (slower)
5. **Set up Opik Tracing:** Add `OPIK_API_KEY` to `.env` for observability

## Example Test Questions

From `test_questions.md`:
1. "Hver er munurinn á innihaldsefnum í Voltaren forte og voltaren hlaupi?"
2. "Hvað þarf kona með barn á brjósti að hafa í huga þegar hún notar Voriconazole?"

Additional examples:
- "Hverjar eru frábendingar fyrir Tegretol?"
- "Hver er skammturinn fyrir Voltaren?"
- "Hver eru aukaverkanirnar fyrir Voriconazole?"
- "Hvaða lyf geta haft áhrif á Voltaren?"
