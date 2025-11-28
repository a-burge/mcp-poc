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

## Architecture

The system follows a RAG pipeline:
1. PDF Update Event → PDF Fetch Pipeline
2. Document Segmentation (section-based chunking)
3. Vector Store (embeddings and indexing)
4. MCP Server (query processing, retrieval, and generation)

## Documentation

See [REFERENCE.md](REFERENCE.md) for detailed architecture, implementation plans, and technical specifications.

## Status

🚧 **In Development** - POC Implementation Phase

## License

[Add your license here]
