# MCP Server for Up-to-Date PDF Documents

A system that fetches the latest PDF documents (SmPC files from serlyfjaskra.is), processes them using section-based chunking, and provides a conversational MCP server using LangChain with Google's Gemini or OpenAI's GPT-4.1, optimized for Icelandic language.

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) system that:
- Fetches the latest PDF documents when updated
- Breaks PDFs into manageable, semantically coherent segments
- Provides a conversational MCP server using LangChain
- Answers questions based on PDF content with source attribution

### Domain Context
- **Documents**: SmPC (Summary of Product Characteristics) files
- **Source**: serlyfjaskra.is
- **Language**: Icelandic (all documents and user queries)
- **Users**: Healthcare professionals (doctors, pharmacists)

## Key Features

- **Section-Based Chunking**: Preserves semantic integrity by never splitting sections across chunks
- **Source Attribution**: All answers include citations to specific document sections
- **Icelandic Language Support**: Optimized for accurate Icelandic language understanding and generation
- **RAG Architecture**: Efficient retrieval and generation using vector embeddings
- **Dual LLM Support**: Configurable support for both Google Gemini and OpenAI GPT-4.1

## Architecture

The system follows a RAG pipeline:
1. PDF Update Event → PDF Fetch Pipeline
2. Document Segmentation (section-based chunking)
3. Vector Store (embeddings and indexing)
4. MCP Server (query processing, retrieval, and generation)

## Quick Start

### Prerequisites

- Python 3.11 or higher (required for httpx>=0.27 and Opik compatibility)
- API key for either Google Gemini or OpenAI GPT-4.1

### Installation

1. **Clone the repository** (if applicable) or navigate to the project directory

2. **Ensure Python 3.11+ is installed**:
   ```bash
   python3 --version  # Should show 3.11 or higher
   ```
   
   If you need to install Python 3.11+:
   - **macOS (using Homebrew)**: `brew install python@3.11`
   - **Using pyenv**: `pyenv install 3.11.9 && pyenv local 3.11.9`
   - **Linux**: Use your distribution's package manager
   - **Windows**: Download from [python.org](https://www.python.org/downloads/)

3. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure environment variables**:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your API keys:
   ```env
   # Choose one LLM provider
   LLM_PROVIDER=gemini  # or "gpt4"
   
   # For Gemini
   GOOGLE_API_KEY=your_google_api_key_here
   
   # For GPT-4.1
   OPENAI_API_KEY=your_openai_api_key_here
   
   # PDF URL (update with actual SmPC PDF URL)
   PDF_URL=https://serlyfjaskra.is/example/smpc.pdf
   ```

### Running the Application

**Start the Streamlit interface**:
```bash
streamlit run src/streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`.

### Usage

1. **Process a PDF**:
   - Enter the PDF URL in the sidebar
   - Select your LLM provider (Gemini or GPT-4.1)
   - Click "Vinna úr PDF" (Process PDF) button
   - Wait for the document to be downloaded, chunked, and indexed

2. **Ask Questions**:
   - Enter your question in Icelandic in the query box
   - Click "Leita" (Search) button
   - View the answer with source citations

3. **View Sources**:
   - Expand the source sections to see the exact text used to generate the answer
   - Each source includes section name, document name, and page number

## Project Structure

```
fwh/
├── README.md                 # This file
├── REFERENCE.md              # Detailed technical documentation
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variable template
├── config.py                 # Configuration management
├── src/
│   ├── __init__.py
│   ├── pdf_fetcher.py        # PDF download and text extraction
│   ├── chunker.py            # Section-based chunking
│   ├── vector_store.py       # Vector store management
│   ├── rag_chain.py          # RAG chain setup
│   └── streamlit_app.py      # Streamlit UI
└── data/
    ├── pdfs/                 # Downloaded PDFs
    └── vector_store/         # Persisted vector store
```

## Configuration

Configuration is managed through environment variables in `.env` file:

- `LLM_PROVIDER`: "gemini" or "gpt4"
- `GOOGLE_API_KEY`: Google Gemini API key (required if using Gemini)
- `OPENAI_API_KEY`: OpenAI API key (required if using GPT-4.1)
- `PDF_URL`: Default PDF URL to process
- `CHUNK_SIZE`: Target chunk size in characters (default: 300)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 0)
- `EMBEDDING_MODEL`: Embedding model name (default: paraphrase-multilingual-MiniLM-L6-v2)
- `RETRIEVAL_TOP_K`: Number of chunks to retrieve (default: 5)

## API Keys

### Google Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Copy the key to your `.env` file

### OpenAI API Key

1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Sign in or create an account
3. Create a new API key
4. Copy the key to your `.env` file

## Technical Details

### Section-Based Chunking

The system uses intelligent section detection to preserve semantic integrity:
- Detects section headers using pattern matching
- Never splits content within a section
- Tags each chunk with metadata (section, page, source document)
- Subdivides large sections only when necessary

### Vector Store

- Uses Chroma for persistent vector storage
- Employs multilingual embeddings (paraphrase-multilingual-MiniLM-L6-v2) for Icelandic support
- Stores metadata for filtering and source attribution

### RAG Chain

- Retrieves top-k relevant chunks based on semantic similarity
- Uses custom Icelandic prompts emphasizing accuracy and source citation
- Supports both Gemini and GPT-4.1 with configurable selection
- Returns answers with source references

## Troubleshooting

### Common Issues

1. **API Key Errors**:
   - Ensure your API key is correctly set in `.env`
   - Verify the key is valid and has sufficient credits/quota

2. **PDF Download Failures**:
   - Check that the PDF URL is accessible
   - Verify network connectivity
   - Ensure the URL points to a valid PDF file

3. **Import Errors**:
   - Make sure all dependencies are installed: `pip install -r requirements.txt`
   - Verify you're using the correct Python version (3.11+)

4. **Vector Store Issues**:
   - Clear the vector store by deleting `data/vector_store/` directory
   - Re-process the PDF to rebuild the index

## Development

### Running Tests

(To be implemented)

### Code Style

- Follows PEP 8 standards
- Uses type hints throughout
- Google-style docstrings for public functions

## Documentation

See [REFERENCE.md](REFERENCE.md) for detailed architecture, implementation plans, and technical specifications.

## Status

✅ **POC Complete** - Ready for testing and evaluation

## License

[Add your license here]
