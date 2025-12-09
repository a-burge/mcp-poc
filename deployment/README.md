# MCP Server Deployment Package

This is a clean deployment package containing only the files needed to run the MCP server.

## Structure

- `run_mcp_server.py` - Server entry point
- `config.py` - Configuration management
- `src/` - Core source modules
- `web/` - Web frontend
- `data/` - Vector store and structured data
- `requirements.txt` - Python dependencies

## Deployment

See `REPLIT_DEPLOYMENT.md` for detailed deployment instructions.

## Environment Variables

Set these in Replit Secrets:
- `MCP_AUTH_USERNAME` - Demo username
- `MCP_AUTH_PASSWORD` - Demo password
- `LLM_PROVIDER` - `gemini` or `gpt5`
- `GOOGLE_API_KEY` - If using Gemini
- `OPENAI_API_KEY` - If using GPT-5
