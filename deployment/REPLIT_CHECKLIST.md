# Replit Deployment Quick Checklist

Use this checklist to ensure everything is configured correctly before deploying.

## Pre-Deployment Checklist

### Code Files
- [ ] `run_mcp_server.py` exists and is executable
- [ ] `config.py` exists
- [ ] `requirements.txt` exists and is up-to-date
- [ ] `src/` directory with all Python modules uploaded
- [ ] `web/` directory with `index.html`, `app.js`, `style.css` uploaded
- [ ] `.replit` file configured correctly

### Environment Variables (Replit Secrets)
- [ ] `MCP_AUTH_USERNAME` - Demo username
- [ ] `MCP_AUTH_PASSWORD` - Demo password (strong!)
- [ ] `LLM_PROVIDER` - Set to `gemini` or `gpt5`
- [ ] `GOOGLE_API_KEY` - If using Gemini
- [ ] `OPENAI_API_KEY` - If using GPT-5
- [ ] `OPIK_API_KEY` - Optional, for observability
- [ ] `OPIK_PROJECT_NAME` - Optional, defaults to `mcp-poc`

### Data Files
- [ ] `data/vector_store/` directory exists (or will be created on first run)
- [ ] Any pre-built vector stores uploaded (if applicable)
- [ ] `data/structured/` directory with SmPC JSON files (if needed)

### Testing
- [ ] Run `pip install -r requirements.txt` successfully
- [ ] Server starts without errors
- [ ] Login page loads at root URL (`/`)
- [ ] Can authenticate with test credentials
- [ ] Query interface appears after login
- [ ] Can submit a test query successfully

## Post-Deployment Checklist

- [ ] Public URL is accessible
- [ ] Login form displays correctly
- [ ] CSS and JavaScript load properly
- [ ] Authentication works with demo credentials
- [ ] Query submission works
- [ ] Results display correctly
- [ ] Citations and sources show properly

## Quick Reference

### Required Secrets (Minimum)
```
MCP_AUTH_USERNAME=your_username
MCP_AUTH_PASSWORD=your_password
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_key
```

### .replit File Content
```
run = "python run_mcp_server.py"
entrypoint = "run_mcp_server.py"
```

### Common Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Run server manually (if needed)
python run_mcp_server.py

# Check if server is running
curl http://localhost:8000/
```

### Public URL Format
```
https://your-repl-name.your-username.repl.co
```
