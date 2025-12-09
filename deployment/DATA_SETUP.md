# Data Setup Instructions

The `data/` directory (2.7GB) is excluded from Git due to GitHub's size limits.

## After Cloning to Replit

You have two options:

### Option 1: Upload Pre-built Data (Recommended)

1. **From your local machine**, compress the data directory:
   ```bash
   cd /path/to/mcp-poc
   tar -czf data.tar.gz data/
   ```

2. **Upload to Replit:**
   - Use Replit's file upload feature
   - Upload `data.tar.gz`
   - Extract in Replit Shell:
     ```bash
     tar -xzf data.tar.gz
     ```

3. **Verify:**
   ```bash
   ls -lh data/
   du -sh data/vector_store/
   ```

### Option 2: Rebuild Vector Store (Slower)

If you only upload `data/structured/` (JSON files):

1. Upload `data/structured/` directory
2. The server will rebuild the vector store on first run
3. This may take 10-30 minutes depending on file count

## Data Directory Structure

```
data/
├── vector_store/     # ChromaDB vector store (~2.5GB)
├── structured/       # Structured JSON files (~200MB)
├── atc/             # ATC index files (~few MB)
└── ingredients/     # Ingredients index files (~few MB)
```

## Minimum Required

For the server to work, you need at least:
- `data/vector_store/` - Pre-built ChromaDB (fastest)
- OR `data/structured/` - JSON files (server will rebuild vector store)

## File Size Breakdown

- `vector_store/`: ~2.5GB (largest)
- `structured/`: ~200MB
- `atc/`: ~few MB
- `ingredients/`: ~few MB

**Total:** ~2.7GB
