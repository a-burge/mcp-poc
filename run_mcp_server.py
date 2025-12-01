#!/usr/bin/env python3
"""
Run the MCP server.

Usage:
    python run_mcp_server.py [--host HOST] [--port PORT]
"""
import argparse
import logging
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from src.mcp_server import app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SmPC MCP Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload
    )
