"""RAG chain setup with Gemini/GPT-5 Mini support for Icelandic."""
import logging
from typing import Dict, Any, List, Optional

from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
import opik
from opik.integrations.langchain import OpikTracer

from config import Config
from src.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


# Icelandic system prompt emphasizing accuracy and source citation
ICELANDIC_SYSTEM_PROMPT = """Þú ert aðstoðarmaður sem svarar spurningum um lyfjaupplýsingar á íslensku. 

Mikilvægar leiðbeiningar:
- Notaðu EINUNGIS upplýsingarnar úr gefnum skjölum til að svara
- Ef svarið er ekki í skjölunum, segðu að þú vitir það ekki
- Vitnaðu ALLTAF í tilheyrandi kafla (section) þegar þú svarar með sniðinu: [drug_id, kafli section_number: section_title]
- Notaðu nákvæmar tilvitnanir úr skjölunum fyrir mikilvægar upplýsingar (t.d. skammtar, viðvörun)
- Svaraðu á nákvæmri og villulausri íslensku
- Ekki búa til upplýsingar sem ekki eru í skjölunum
- Ekki búa til kafla sem ekki eru til
- Fyrir lista, notaðu punktalista (bullet points)
- Fyrir samanburð, notaðu töflu (table) ef viðeigandi
- Svörin skulu vera stutt og skýr, fagleg íslenska

{history}

Kontext úr skjölum: {context}

Spurning: {question}

Svar með vitnunum (MUST include citations in format [drug_id, kafli section_number: section_title]):"""


def create_llm(provider: str = None) -> Any:
    """
    Create LLM instance based on provider.
    
    Args:
        provider: "gemini" or "gpt5" (defaults to Config.LLM_PROVIDER)
        
    Returns:
        LLM instance (ChatGoogleGenerativeAI or ChatOpenAI)
        
    Raises:
        ValueError: If provider is invalid or API key is missing
    """
    if provider is None:
        provider = Config.LLM_PROVIDER
    
    provider = provider.lower()
    
    # Validate provider exists in configuration
    if provider not in Config.LLM_MODELS:
        valid_providers = ", ".join(Config.LLM_MODELS.keys())
        raise ValueError(
            f"Invalid provider: {provider}. Must be one of: {valid_providers}"
        )
    
    # Get model name from centralized configuration
    model_name = Config.LLM_MODELS[provider]
    
    if provider == "gemini":
        if not Config.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is required for Gemini")
        
        logger.info(f"Creating Gemini LLM with model: {model_name}")
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=Config.GOOGLE_API_KEY,
            temperature=0.1,  # Low temperature for accuracy
        )
        return llm
    
    elif provider == "gpt5":
        if not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for GPT-5 Mini")
        
        logger.info(f"Creating GPT-5 Mini LLM with model: {model_name}")
        llm = ChatOpenAI(
            model=model_name,
            openai_api_key=Config.OPENAI_API_KEY,
            temperature=1, 
            reasoning_effort="minimal"
        )
        return llm
    
    else:
        # This should not be reached due to validation above, but kept for safety
        valid_providers = ", ".join(Config.LLM_MODELS.keys())
        raise ValueError(
            f"Invalid provider: {provider}. Must be one of: {valid_providers}"
        )


def _configure_opik() -> Optional["OpikTracer"]:
    """
    Configure Opik and create OpikTracer if API key is set.
    
    Returns:
        OpikTracer instance if Opik is configured, None otherwise
    """
    if not Config.OPIK_API_KEY:
        logger.warning("OPIK_API_KEY not set. Opik tracing will be disabled.")
        return None
    
    try:
        # Skip opik.configure() due to httpx version compatibility issue
        # Instead, set environment variables and create OpikTracer directly
        # Opik will read configuration from environment variables automatically
        import os
        
        # Set environment variables for Opik to read
        os.environ["OPIK_API_KEY"] = Config.OPIK_API_KEY
        if Config.OPIK_PROJECT_NAME:
            os.environ["OPIK_PROJECT_NAME"] = Config.OPIK_PROJECT_NAME
        
        # Create OpikTracer directly - it will read from environment variables
        # This avoids the httpx.Client proxy parameter compatibility issue
        tracer = OpikTracer(project_name=Config.OPIK_PROJECT_NAME)
        logger.info(f"Opik tracing enabled (project: {Config.OPIK_PROJECT_NAME})")
        return tracer
    except Exception as e:
        logger.warning(f"Failed to configure Opik: {e}", exc_info=True)
        return None


def create_qa_chain(
    vector_store_manager: VectorStoreManager,
    provider: str = None,
    medication_filter: Optional[str] = None,
    memory: Optional[ConversationBufferMemory] = None
) -> RetrievalQA:
    """
    Create RetrievalQA chain with custom prompt for Icelandic.
    
    Args:
        vector_store_manager: VectorStoreManager instance
        provider: LLM provider ("gemini" or "gpt5")
        medication_filter: Optional medication name to filter retrieval
        memory: Optional ConversationBufferMemory for multi-turn conversations
        
    Returns:
        RetrievalQA chain instance
    """
    logger.info(f"Creating QA chain with provider: {provider or Config.LLM_PROVIDER}")
    
    # Create LLM
    llm = create_llm(provider)
    
    # Create prompt template (with or without memory)
    if memory:
        prompt = PromptTemplate(
            template=ICELANDIC_SYSTEM_PROMPT,
            input_variables=["context", "question", "history"]
        )
    else:
        # Simplified prompt without history
        prompt_template = ICELANDIC_SYSTEM_PROMPT.replace("{history}\n\n", "")
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
    
    # Get retriever (with optional medication filter)
    if medication_filter:
        retriever = vector_store_manager.get_retriever_with_filter(medication_name=medication_filter)
    else:
        retriever = vector_store_manager.get_retriever()
    
    # Configure Opik tracer
    opik_tracer = _configure_opik()
    
    # Prepare callbacks
    callbacks = []
    if opik_tracer:
        callbacks.append(opik_tracer)
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
        callbacks=callbacks if callbacks else None,
    )
    
    logger.info("QA chain created successfully")
    return qa_chain


def query_rag(
    qa_chain: RetrievalQA,
    question: str,
    medication_filter: Optional[str] = None,
    memory: Optional[ConversationBufferMemory] = None
) -> Dict[str, Any]:
    """
    Query RAG chain with a question.
    
    Args:
        qa_chain: RetrievalQA chain instance
        question: Question to ask (in Icelandic)
        medication_filter: Optional medication name filter (for logging)
        
    Returns:
        Dictionary with "answer" and "sources" keys
    """
    logger.info(f"Querying RAG: {question[:50]}...")
    
    # Configure Opik tracer for this invocation
    # Callbacks must be passed at invocation time, not just at chain creation
    opik_tracer = _configure_opik()
    callbacks = []
    if opik_tracer:
        callbacks.append(opik_tracer)
        logger.info("Opik tracer configured for this query")
    else:
        logger.debug("Opik tracer not available (API key may be missing)")
    
    try:
        # Prepare input with memory if available
        chain_input = {"query": question}
        if memory:
            memory_vars = memory.load_memory_variables({})
            history = memory_vars.get("chat_history", [])
            if history:
                # Format history for prompt
                history_str = "\n".join([
                    f"Spurning: {msg.content}" if hasattr(msg, 'type') and msg.type == 'human' 
                    else f"Svar: {msg.content}"
                    for msg in history[-4:]  # Last 4 exchanges
                ])
                chain_input["history"] = history_str
            else:
                chain_input["history"] = ""
        
        # Opik instrumentation: Log prompt construction
        if opik_tracer:
            # Get context from retriever (if available)
            context_blocks = []
            try:
                # Try to get context preview
                retriever = qa_chain.retriever
                preview_docs = retriever.get_relevant_documents(question)
                context_blocks = [
                    {
                        "text": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in preview_docs[:3]  # Preview first 3
                ]
            except:
                pass
            
            opik.log_event("prompt_construction", {
                "question": question,
                "has_memory": memory is not None,
                "memory_history_length": len(chain_input.get("history", "")),
                "context_blocks_preview": context_blocks
            })
        
        # Use invoke() method with explicit callbacks parameter
        # This ensures callbacks are properly propagated through the chain
        if callbacks:
            result = qa_chain.invoke(
                chain_input,
                config={"callbacks": callbacks}
            )
        else:
            # Fallback to dict-style invocation if no callbacks
            result = qa_chain(chain_input)
        
        answer = result.get("result", "Ég get ekki svarað þessari spurningu.")
        source_docs = result.get("source_documents", [])
        
        # Opik instrumentation: Log model output and citations
        if opik_tracer:
            opik.log_event("model_invocation", {
                "question": question,
                "answer_length": len(answer),
                "sources_count": len(source_docs),
                "answer_preview": answer[:200] + "..." if len(answer) > 200 else answer
            })
        
        # Extract source information with full metadata
        sources = []
        for doc in source_docs:
            metadata = doc.metadata
            sources.append({
                "drug_id": metadata.get("drug_id", metadata.get("medication_name", "Unknown")),
                "section_number": metadata.get("section_number", "Unknown"),
                "section_title": metadata.get("section_title", metadata.get("section", "Unknown")),
                "canonical_key": metadata.get("canonical_key", "Unknown"),
                "source": metadata.get("source", "Unknown"),
                "page": metadata.get("page", "Unknown"),
                "medication_name": metadata.get("medication_name", "Unknown"),
                "text": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            })
        
        # Validate citations are present in answer
        answer_with_citations = _ensure_citations(answer, sources)
        
        return {
            "answer": answer_with_citations,
            "sources": sources,
        }
    
    except Exception as e:
        logger.error(f"Error querying RAG: {e}", exc_info=True)
        return {
            "answer": f"Villa kom upp við að svara spurningu: {str(e)}",
            "sources": [],
        }


def _ensure_citations(answer: str, sources: List[Dict[str, Any]]) -> str:
    """
    Ensure answer includes citations in the format [drug_id, kafli section_number: section_title].
    
    Args:
        answer: LLM-generated answer
        sources: List of source documents with metadata
        
    Returns:
        Answer with citations added if missing
    """
    # Check if citations are already present
    if "[" in answer and "kafli" in answer.lower():
        return answer
    
    # Add citations at the end
    citations = []
    seen = set()
    for source in sources:
        drug_id = source.get("drug_id", source.get("medication_name", "Unknown"))
        section_num = source.get("section_number", "Unknown")
        section_title = source.get("section_title", "Unknown")
        
        citation_key = f"{drug_id}_{section_num}"
        if citation_key not in seen:
            citations.append(f"[{drug_id}, kafli {section_num}: {section_title}]")
            seen.add(citation_key)
    
    if citations:
        citation_text = " " + " ".join(citations)
        return answer + citation_text
    
    return answer
