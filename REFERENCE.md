# MCP Server for Up-to-Date PDF Documents with LangChain and Gemini/GPT-4.1
## Reference Document

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture Overview](#architecture-overview)
3. [Component Specifications](#component-specifications)
4. [POC Implementation Plan](#poc-implementation-plan)
5. [Production Considerations](#production-considerations)
6. [Security & Accuracy Requirements](#security--accuracy-requirements)
7. [Key Design Decisions](#key-design-decisions)

---

## Project Overview

### Goal
Build a system that:
- Fetches the latest PDF documents (SmPC files from serlyfjaskra.is) when updated
- Breaks PDFs into manageable, semantically coherent segments
- Provides a conversational MCP server using LangChain with Google's Gemini or OpenAI's GPT-4.1 (optimized for Icelandic)
- Answers questions based on PDF content using Retrieval-Augmented Generation (RAG)

### Domain Context
- **Documents**: SmPC (Summary of Product Characteristics) files
- **Source**: serlyfjaskra.is
- **Language**: Icelandic (all documents and user queries)
- **Users**: Healthcare professionals (doctors, pharmacists)
- **Critical Requirements**: 
  - Accuracy and source attribution
  - Section integrity
  - **Icelandic language understanding and error-free generation** (non-negotiable)
  - No reasoning capabilities required, but superior language quality is essential

---

## Architecture Overview

### RAG Pipeline Components

```
┌─────────────────────┐
│  PDF Update Event   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  PDF Fetch Pipeline │
│  - Download PDF     │
│  - Extract Text     │
│  - Preprocess       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Document Segmentation│
│  - Section Detection │
│  - Chunking          │
│  - Metadata Tagging  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Vector Store        │
│  - Embeddings        │
│  - Indexing          │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  MCP Server          │
│  - Query Processing  │
│  - Retrieval         │
│  - Gemini/GPT-4.1    │
│    Generation         │
└─────────────────────┘
```

---

## Component Specifications

### 1. PDF Update Pipeline

#### Purpose
Automatically retrieve new or updated PDF files and extract text content.

#### Recommended Implementation
- **Trigger**: Event-driven (webhook/message when PDF published)
- **Download**: HTTP request using `requests` or `httpx`
- **Text Extraction**: PyMuPDF (fitz) or PDFPlumber
- **LangChain Integration**: `PyPDFLoader` for convenience
- **Preprocessing**: Remove headers/footers, clean characters, normalize spacing

#### Key Requirements
- Process PDFs in near real-time when updates occur
- Handle PDF encoding and layout variations
- Preserve document structure for section detection

#### Trade-offs & Alternatives

| Aspect | POC Choice | Production Alternative | Rationale |
|--------|-----------|----------------------|-----------|
| **Trigger** | Event-driven | Polling (periodic checks) | Event-driven = real-time; Polling = simpler but delayed |
| **PDF Parser** | PyMuPDF/PDFPlumber | Unstructured.io, AWS Textract | POC: Fast & simple; Production: Better for complex layouts |
| **Orchestration** | Python script | Airflow, Prefect, CI/CD | POC: Minimal; Production: Robust error handling & logging |

---

### 2. Document Segmentation (Chunking)

#### Purpose
Break PDF text into logical segments that preserve semantic meaning and enable efficient retrieval.

#### Critical Requirements
1. **Section Integrity**: Never split chapters or sections across chunks
2. **Context Preservation**: Maintain full context within each section
3. **Metadata Tagging**: Track source document, section title, page number
4. **No Overlap Dependency**: Each section should be self-contained (no reliance on overlap)

#### Recommended Implementation

**Primary Strategy: Section-Based Chunking**
- Detect section boundaries using headings/patterns
- Split at section boundaries, not arbitrary character counts
- If a section exceeds max size, subdivide intelligently (preserve subsections)
- Tag each chunk with:
  - Source document name
  - Section title (e.g., "Indications", "Contraindications")
  - Page number(s)
  - Drug name

**Secondary Strategy: Hybrid Approach**
- Split by section where possible
- If section > threshold (e.g., 2000 tokens), subdivide by subsections
- Maintain section metadata at all levels

#### Chunk Size Guidelines
- **Target**: 200-500 tokens per chunk (balance precision vs. context)
- **Minimum**: Large enough to contain complete semantic units
- **Maximum**: Within LLM context limits (consider retrieval of 3-5 chunks)

#### Tools
- **Primary**: LangChain `RecursiveCharacterTextSplitter` (with custom separators)
- **Alternative**: LlamaIndex (hierarchical indices) for production
- **Custom Logic**: Regex/pattern matching for section detection

#### Trade-offs & Alternatives

| Aspect | Recommendation | Alternative | Rationale |
|--------|---------------|-------------|-----------|
| **Chunking Method** | Section-based | Fixed-size chunks | Section-based preserves semantic integrity |
| **Chunk Size** | 200-500 tokens | 150-250 or 500+ | Balance between precision and context |
| **Overlap** | Minimal/None | 50-token overlap | Sections should be self-contained |
| **Table Handling** | Text extraction | Structured extraction (CSV/JSON) | POC: Simple; Production: May need special handling |

#### Update Strategy
- When PDF updates: Regenerate all chunks for that document
- Drop old chunks, create new ones (ensures consistency)
- Future optimization: Diff-based updates (only re-index changed sections)

---

### 3. MCP Server Implementation

#### Purpose
Provide conversational interface that answers questions using PDF content via RAG.

#### Architecture: Retrieval-Augmented Generation

**Components:**
1. **Embedding Model**
   - **Requirement**: Multilingual (Icelandic support)
   - **Options**:
     - `paraphrase-multilingual-MiniLM` (SentenceTransformers)
     - OpenAI `text-embedding-ada-002`
     - LangChain: `HuggingFaceEmbeddings` or `OpenAIEmbeddings`

2. **Vector Store**
   - **POC**: FAISS (in-memory) or Chroma (local)
   - **Production**: Milvus, Weaviate, Qdrant, or Pinecone
   - **Features Needed**: Metadata filtering, similarity search

3. **Retrieval**
   - Embed user query
   - Similarity search in vector store
   - Retrieve top k chunks (typically 3-5)
   - Filter by metadata if needed (e.g., specific section or drug)

4. **Generation (Gemini or GPT-4.1)**
   - **Model**: Google Gemini or OpenAI GPT-4.1 (top performers for Icelandic)
   - **Rationale**: Superior Icelandic language understanding and error-free generation
   - **Integration**: 
     - LangChain `ChatGoogleGenerativeAI` (for Gemini)
     - LangChain `ChatOpenAI` (for GPT-4.1)
   - **Context Windows**:
     - Gemini: Large context support (varies by model version)
     - GPT-4.1: Up to 128k tokens
   - **Prompt Structure**:
     ```
     System: "You are a helpful assistant answering questions about medication 
     information in Icelandic. Use only the provided document excerpts to answer, 
     and quote them if necessary. Include section references. If the answer is not 
     in the excerpts, say you don't know. Respond in Icelandic with accurate, 
     error-free language."
     
     User: [Question in Icelandic]
     
     Context: [Retrieved chunks with source/section labels]
     
     Assistant: [Model's response in Icelandic]
     ```

5. **Server Interface**
   - **POC**: Streamlit app or CLI
   - **Production**: FastAPI/Flask with REST API

#### Trade-offs & Alternatives

| Component | Choice | Alternative | Rationale |
|-----------|--------|-------------|-----------|
| **Framework** | LangChain | LlamaIndex, Haystack | LangChain: One-stop, familiar, quick POC |
| **LLM** | Gemini or GPT-4.1 | Claude, Llama 2 (open-source) | Gemini/GPT-4.1: Top Icelandic performance, strong language understanding; Claude: Large context but weaker Icelandic; Open models: Lower cost but more setup |
| **Retrieval vs Direct** | RAG (retrieval) | Direct LLM (whole PDF) | RAG: Faster, scalable, transparent; Direct: Slower, costly, opaque |
| **Vector DB** | FAISS/Chroma (POC) | Qdrant/Weaviate/Pinecone (Prod) | POC: Simple & free; Prod: Scalable & persistent |

---

## POC Implementation Plan

### Scope: Minimal Viable Prototype

**Goal**: Demonstrate working system with one PDF, basic section-based chunking, and Gemini/GPT-4.1 integration for Icelandic language support.

### Shortcuts for POC

1. **Single Hard-Coded PDF**
   - Use one known SmPC PDF URL
   - No automated pipeline or multi-document handling
   - Manual trigger for demonstration

2. **Simple Section Detection**
   - Pattern matching for common section headers
   - No advanced NLP or document structure parsing
   - Basic regex/string matching for section boundaries

3. **In-Memory Vector Store**
   - FAISS or Chroma running locally
   - No external database setup
   - Persist index to disk for demo consistency

4. **Lightweight Interface**
   - Streamlit app or CLI
   - Hard-coded example questions for demo
   - No full web application

### Implementation Steps

#### Step 1: Environment Setup
```bash
# Core dependencies
pip install langchain
pip install pymupdf  # or pdfplumber
# Choose one LLM provider:
pip install google-generativeai  # For Gemini
# OR
pip install openai  # For GPT-4.1
pip install chromadb  # or faiss-cpu
pip install sentence-transformers  # for embeddings
pip install streamlit  # for UI (optional)
```

#### Step 2: PDF Fetcher
- Function to download PDF from hard-coded URL
- Extract text using PyMuPDF or LangChain `PyPDFLoader`
- Return text with basic structure preserved

#### Step 3: Section-Based Chunking
- Identify section headers (common patterns: "1. INDICATIONS", "2. DOSAGE", etc.)
- Split text at section boundaries
- Tag chunks with:
  - Section name
  - Source document
  - Page number
- Ensure no section is split across chunks

#### Step 4: Vector Store Setup
- Initialize Chroma or FAISS
- Generate embeddings for all chunks (multilingual model)
- Store with metadata (section, document, page)
- Save index to disk

#### Step 5: LangChain QA Chain
- Configure `RetrievalQA` chain:
  - Vector store retriever (with metadata filtering support)
  - Gemini via `ChatGoogleGenerativeAI` OR GPT-4.1 via `ChatOpenAI`
  - Custom prompt with source citation requirements and Icelandic language instructions
- Test with sample queries in Icelandic

#### Step 6: Demo Interface
- Streamlit app or CLI
- Input: User question
- Output: Answer with section references
- Display retrieved chunks for transparency

### Success Criteria
- ✅ Can fetch and parse one SmPC PDF
- ✅ Chunks preserve section integrity (no splits within sections)
- ✅ Retrieval finds relevant sections
- ✅ Gemini/GPT-4.1 answers in Icelandic with citations to specific sections
- ✅ Demo shows non-hallucination (answers traceable to source)

---

## Production Considerations

### Full Tech Stack (Post-POC)

#### 1. Automated Pipeline
- **Event Listener**: Webhook receiver or message queue consumer
- **Orchestration**: Apache Airflow, Prefect, or serverless functions
- **Error Handling**: Retry logic, logging, alerting
- **Versioning**: Track PDF versions (date, hash) for audit trail

#### 2. Advanced Chunking
- **NLP-Based Section Detection**: Use models to identify section boundaries
- **Document Structure Parser**: Unstructured.io for complex layouts
- **Table Extraction**: Special handling for structured data (CSV/JSON)
- **Incremental Updates**: Diff-based re-indexing (only update changed sections)

#### 3. Persistent Vector Database
- **Options**: Weaviate, Milvus, Qdrant, or Pinecone
- **Features**: 
  - Persistence across restarts
  - Replication for availability
  - Hybrid search (semantic + keyword)
  - Metadata filtering at scale

#### 4. Robust UI/API
- **API**: FastAPI with OpenAPI documentation
- **UI**: Full web application with search, filters, history
- **Authentication**: Access control for authorized users
- **Monitoring**: Query logging, performance metrics, error tracking

---

## Security & Accuracy Requirements

### Data Privacy & Access Control

**Requirements:**
- Process PDFs on own infrastructure when possible
- Review API data usage policies (Google, OpenAI)
- Secure API keys (environment variables, secret manager)
- Implement authentication for external access

**Trade-off**: Local processing (open-source models) vs. Cloud APIs (Gemini/GPT-4.1)
- Local: Maximum privacy, no API costs, but requires hardware and may have weaker Icelandic support
- Cloud: Faster setup, superior Icelandic language quality, but data sent externally

### Prompt Injection Mitigation

**Threats:**
1. **User Prompt Injection**: Malicious user queries
2. **Indirect Prompt Injection**: Malicious content in PDFs

**Mitigations:**
- **Filtering**: Remove instruction-like patterns from extracted text
- **Strong System Prompts**: Explicitly forbid following instructions in documents
- **Monitoring**: Test with malicious inputs, implement content scanning
- **Source Control**: Only ingest from trusted sources (verify HTTPS, checksums)

### Retrieval Accuracy & Answer Correctness

**Requirements:**
1. **Grounding & Citations**
   - Answers must include quotes or section references
   - Prompt model to cite sources explicitly
   - Display source document and section in UI

2. **Limit Hallucination**
   - Prompt: "If answer not in documents, say 'I don't know'"
   - Ground all answers in retrieved chunks
   - No creative freedom beyond source material

3. **Multiple Chunk Retrieval**
   - Retrieve 3-5 top chunks per query
   - Consider reranking for precision (optional)
   - Ensure chunks are relevant (monitor retrieval quality)

4. **Evaluation & Feedback**
   - Test with known questions/answers
   - Identify retrieval vs. generation failures
   - Iterate on chunk size, embeddings, prompts

5. **Error Handling**
   - Graceful API failures (Gemini/GPT-4.1, embeddings)
   - Out-of-scope query handling
   - Restrict to medication-related questions only

### Compliance & Safety (Medical Domain)

**Requirements:**
- **Verbatim Sourcing**: Critical info (doses, contraindications) should use direct quotes
- **Version Tracking**: Display document date/version with answers
- **Disclaimer**: "This assistant provides information from official SmPC documents. Always verify critical decisions against the source document."
- **Scope Limitation**: Only answer questions within document scope; refuse medical advice beyond SmPC content

---

## Key Design Decisions

### 1. Section-Based Chunking (Non-Negotiable)

**Rationale**: 
- SmPC documents have specific chapters (Indications, Dosage, Contraindications, etc.)
- Users need complete section context (e.g., all ingredients together)
- Overlap is not acceptable—sections must be self-contained

**Implementation**: 
- Detect section boundaries (headings, numbering)
- Never split within a section
- Tag chunks with section metadata for filtered retrieval

### 2. Source Attribution (Critical)

**Rationale**: 
- Healthcare professionals need to verify answers
- Prevents hallucination concerns
- Regulatory compliance (traceability)

**Implementation**:
- Include section references in every answer
- Display source document and section in UI
- Quote directly from source for critical information

### 3. LangChain + Gemini/GPT-4.1 (POC Choice)

**Rationale**:
- LangChain: One-stop framework, familiar, quick development
- Gemini/GPT-4.1: Top performers for Icelandic language understanding and generation
- Critical for medical domain: Error-free Icelandic generation is essential
- Strong language understanding ensures accurate interpretation of user queries in Icelandic
- Trade-off: API costs vs. superior language quality (critical for this use case)

**Model Selection**:
- **Primary**: Gemini (currently top of leaderboard for Icelandic)
- **Alternative**: GPT-4.1 (close second, strong alternative)
- Both provide excellent Icelandic support without requiring fine-tuning

**Alternatives for Production**:
- LlamaIndex for advanced indexing
- Haystack for modular pipelines
- Open-source LLMs for cost control (requires infrastructure, but may have weaker Icelandic support)

### 4. Event-Driven Updates

**Rationale**:
- Real-time processing of new PDFs
- Avoids polling overhead
- Critical for up-to-date information

**POC Shortcut**: Manual trigger (single PDF)
**Production**: Full event-driven pipeline

---

## Technical Specifications Summary

### POC Stack
- **PDF Processing**: PyMuPDF or LangChain `PyPDFLoader`
- **Chunking**: LangChain `RecursiveCharacterTextSplitter` (section-aware)
- **Embeddings**: SentenceTransformers (multilingual) or OpenAI
- **Vector Store**: FAISS or Chroma (local)
- **LLM**: Google Gemini or OpenAI GPT-4.1 (via LangChain, optimized for Icelandic)
- **Interface**: Streamlit or CLI

### Production Stack
- **Orchestration**: Airflow/Prefect
- **PDF Processing**: Unstructured.io (if complex layouts)
- **Vector DB**: Weaviate, Milvus, Qdrant, or Pinecone
- **API**: FastAPI
- **Monitoring**: Logging, metrics, error tracking

---

## Next Steps

1. **POC Development**:
   - Set up environment
   - Implement PDF fetcher (single PDF)
   - Build section-based chunker
   - Create vector store and QA chain
   - Build demo interface

2. **Testing**:
   - Test with known questions/answers
   - Verify section integrity
   - Check citation accuracy
   - Test edge cases (ambiguous queries, missing info)

3. **Production Planning**:
   - Design event-driven pipeline
   - Select production vector DB
   - Plan authentication/authorization
   - Set up monitoring and logging

---

## References & Resources

- **LangChain Documentation**: https://python.langchain.com/
- **Google Gemini API**: https://ai.google.dev/docs
- **OpenAI GPT-4.1 API**: https://platform.openai.com/docs
- **RAG Best Practices**: Various research papers and guides
- **OWASP LLM Security**: Guidelines for LLM applications
- **SmPC Source**: serlyfjaskra.is

---

*Document Version: 1.0*  
*Last Updated: [Current Date]*

