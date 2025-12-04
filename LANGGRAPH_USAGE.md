# LangGraph RAG Chain Usage Guide

This document explains how to use the new LangGraph-based RAG implementation (`rag_chain_langgraph.py`) which provides better observability and control compared to the original RetrievalQA chain.

## Key Features

- **Explicit State Management**: All intermediate state is visible and debuggable
- **Graph-Aware Tracing**: Opik tracing with full graph visualization
- **Domain-Specific Features**: Icelandic language, rich metadata, citations, conversation memory
- **Multi-Provider Support**: Works with both Gemini and GPT-5

## Basic Usage

### Simple Query (No Memory)

```python
from src.vector_store import VectorStoreManager
from src.rag_chain_langgraph import create_rag_graph, query_rag_graph

# Initialize vector store
vector_store_manager = VectorStoreManager()

# Create RAG graph
rag_graph = create_rag_graph(
    vector_store_manager=vector_store_manager,
    provider="gpt5"  # or "gemini"
)

# Query
result = query_rag_graph(
    rag_graph=rag_graph,
    question="Hverjar eru frábendingar fyrir Tegretol?"
)

print(result["answer"])
print(f"Sources: {len(result['sources'])}")
```

### With Conversation Memory

```python
from langchain.memory import ConversationBufferMemory

# Create memory store (shared across sessions)
memory_store = {}

# Create RAG graph with memory store
rag_graph = create_rag_graph(
    vector_store_manager=vector_store_manager,
    provider="gpt5",
    memory_store=memory_store
)

# First question
result1 = query_rag_graph(
    rag_graph=rag_graph,
    question="Hverjar eru frábendingar fyrir Tegretol?",
    session_id="user-123"
)

# Follow-up question (uses conversation history)
result2 = query_rag_graph(
    rag_graph=rag_graph,
    question="Hver er skammturinn?",
    session_id="user-123"  # Same session ID
)
```

### With Medication Filtering

```python
# Create graph with medication filter
rag_graph = create_rag_graph(
    vector_store_manager=vector_store_manager,
    provider="gpt5",
    medication_filter="Tegretol"  # Filter to specific medication
)

# Query (automatically filtered)
result = query_rag_graph(
    rag_graph=rag_graph,
    question="Hverjar eru frábendingarnar?"
)
```

## Graph Architecture

The LangGraph implementation uses a 4-node workflow:

```
START → memory → retrieval → generation → citation → END
```

### Node Responsibilities

1. **memory**: Loads conversation history from memory store
2. **retrieval**: Retrieves relevant documents from vector store
3. **generation**: Generates answer using LLM with retrieved context
4. **citation**: Ensures proper citations and extracts source metadata

### State Structure

```python
class DocumentRAGState(TypedDict):
    question: str
    medication_filter: Optional[str]
    session_id: Optional[str]
    retrieved_docs: List[Document]
    formatted_context: str
    chat_history: str
    answer: str
    sources: List[Dict[str, Any]]
    error: Optional[str]
```

## Migration from RetrievalQA Chain

### Before (RetrievalQA)

```python
from src.rag_chain import create_qa_chain, query_rag

qa_chain = create_qa_chain(
    vector_store_manager=vector_store_manager,
    provider="gpt5"
)

result = query_rag(
    qa_chain=qa_chain,
    question="Hverjar eru frábendingarnar?"
)
```

### After (LangGraph)

```python
from src.rag_chain_langgraph import create_rag_graph, query_rag_graph

rag_graph = create_rag_graph(
    vector_store_manager=vector_store_manager,
    provider="gpt5"
)

result = query_rag_graph(
    rag_graph=rag_graph,
    question="Hverjar eru frábendingarnar?"
)
```

The interface is nearly identical! The main difference is:
- `create_qa_chain()` → `create_rag_graph()`
- `query_rag()` → `query_rag_graph()`
- Returns same structure: `{"answer": ..., "sources": ..., "error": ...}`

## Advantages Over RetrievalQA

1. **Observability**: You can inspect state at each node
2. **Debugging**: See exactly what documents were retrieved before generation
3. **Extensibility**: Easy to add nodes (re-ranking, validation, etc.)
4. **Graph Tracing**: Opik can visualize the entire graph execution
5. **Error Handling**: Errors propagate through state, making debugging easier

## Example: Inspecting Intermediate State

```python
# You can access the graph directly to inspect state
result = rag_graph.graph.invoke({
    "question": "Hverjar eru frábendingarnar?",
    "medication_filter": None,
    "session_id": None,
    "retrieved_docs": [],
    "formatted_context": "",
    "chat_history": "",
    "answer": "",
    "sources": [],
    "error": None
})

# Check what was retrieved
print(f"Retrieved {len(result['retrieved_docs'])} documents")
print(f"Context length: {len(result['formatted_context'])}")
print(f"Answer: {result['answer']}")
```

## Integration with FastAPI/MCP Server

You can use the LangGraph implementation in your MCP server:

```python
from src.rag_chain_langgraph import create_rag_graph, query_rag_graph

# In your FastAPI endpoint
@app.post("/ask")
async def ask_question(request: AskRequest):
    global rag_graph, memory_store
    
    if not rag_graph:
        rag_graph = create_rag_graph(
            vector_store_manager=vector_store_manager,
            provider=Config.LLM_PROVIDER,
            memory_store=memory_store
        )
    
    result = query_rag_graph(
        rag_graph=rag_graph,
        question=request.question,
        session_id=request.session_id,
        medication_filter=request.drug_id
    )
    
    return {
        "answer": result["answer"],
        "sources": result["sources"]
    }
```

## Error Handling

The LangGraph implementation includes comprehensive error handling:

```python
result = query_rag_graph(rag_graph, question="...")

if result.get("error"):
    print(f"Error occurred: {result['error']}")
else:
    print(result["answer"])
```

Errors are captured in the state and propagated through the graph, making debugging easier.

## Opik Tracing

With Opik configured, you get graph-aware tracing:

- See execution flow through each node
- Inspect state at each step
- Visualize the graph structure
- Track performance metrics per node

The tracer is automatically configured if `OPIK_API_KEY` is set in your environment.

## Performance Considerations

- **Graph Compilation**: Happens once during initialization
- **State Management**: Minimal overhead, state is lightweight
- **Memory**: Conversation memory is stored per session
- **Retrieval**: Same performance as RetrievalQA (uses same VectorStoreManager)

## Next Steps

1. **Try it out**: Use the examples above to test the LangGraph implementation
2. **Compare**: Run the same queries with both implementations and compare results
3. **Migrate**: Gradually migrate your codebase to use LangGraph
4. **Extend**: Add custom nodes for your specific needs (re-ranking, validation, etc.)
