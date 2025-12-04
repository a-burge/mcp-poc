"""
LangGraph-based RAG chain with explicit state management.

This implementation combines LangGraph architecture (better observability and control)
with domain-specific features (Icelandic language, rich metadata, citations, conversation memory).

Main advantages over RetrievalQA chain:
- Explicit state management (TypedDict) for better debugging
- Observable intermediate steps (retrieval results visible)
- Graph-aware tracing with Opik
- Easier to extend with additional nodes (re-ranking, validation, etc.)
"""
import json
import logging
import re
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from operator import add
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, START, END
from opik.integrations.langchain import OpikTracer

from config import Config
from src.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


# Pydantic models for structured output
class BinaryDecision(str, Enum):
    """Binary decision values for re-ranking."""
    YES = "YES"
    NO = "NO"


class RerankingDecision(BaseModel):
    """Structured output for re-ranking decision."""
    decision: BinaryDecision = Field(
        description="Whether re-ranking is needed. YES for comparison queries, complex multi-part questions, or multiple medications. NO for simple factual queries."
    )


# Icelandic system prompt emphasizing accuracy and source citation
ICELANDIC_SYSTEM_PROMPT = """Þú ert heilbrigðissérfræðingur sem svarar spurningum um lyfjaupplýsingar á íslensku. 

Mikilvægar leiðbeiningar:
- Notaðu EINUNGIS upplýsingarnar úr gefnum skjölum til að svara
- Ef svarið er ekki í skjölunum, segðu að þú vitir ekki svarið
- Vitnaðu ALLTAF í tilheyrandi kafla (section) þegar þú svarar með sniðinu: [drug_id, kafli section_number: section_title]
- Notaðu nákvæmar tilvitnanir úr skjölunum fyrir mikilvægar upplýsingar (t.d. skammtar, viðvörun)
- Svaraðu á nákvæmri og villulausri íslensku
- Ekki búa til upplýsingar sem ekki eru í skjölunum
- Ekki búa til kafla sem ekki eru til
- Fyrir lista, notaðu punktalista (bullet points)
- Fyrir samanburð, notaðu töflu (table) ef viðeigandi
- Svörin skulu vera stutt og skýr, á faglegri íslensku

{history}

Upplýsingar úr skjölum: {context}

Spurning: {question}

Svar með tilvísunum (MUST include citations in format [drug_id, kafli section_number: section_title]):"""


# Define the LangGraph state (shared between nodes)
class DocumentRAGState(TypedDict):
    """State shared between LangGraph nodes."""
    question: str
    medication_filter: Optional[str]  # Optional medication name filter
    extracted_medication: Optional[str]  # Single medication extracted from query
    extracted_medications: List[str]  # Multiple medications extracted from query
    session_id: Optional[str]  # Session ID for conversation memory
    retrieved_docs: Annotated[List[Document], add]  # List of retrieved documents
    formatted_context: str  # Formatted context string for LLM
    chat_history: str  # Formatted conversation history
    answer: str  # Final answer
    sources: Annotated[List[Dict[str, Any]], add]  # Source metadata
    retrieval_sufficient: bool  # Flag indicating if retrieval found sufficient results
    reranking_needed: Optional[bool]  # Flag indicating if re-ranking is needed (from structured decision)
    error: Optional[str]  # Error message if something goes wrong


class DocumentRAGGraph:
    """
    LangGraph-based Document RAG system with explicit state management.
    
    Combines LangGraph architecture with domain-specific features:
    - Icelandic language support
    - Rich metadata handling (section numbers, drug IDs)
    - Citation enforcement
    - Conversation memory
    - Multi-provider LLM support
    """
    
    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        provider: str = None,
        medication_filter: Optional[str] = None,
        memory_store: Optional[Dict[str, ConversationBufferMemory]] = None
    ):
        """
        Initialize DocumentRAGGraph.
        
        Args:
            vector_store_manager: VectorStoreManager instance
            provider: LLM provider ("gemini" or "gpt5")
            medication_filter: Optional medication name to filter retrieval
            memory_store: Optional dictionary mapping session_id to ConversationBufferMemory
        """
        self.vector_store_manager = vector_store_manager
        self.provider = provider or Config.LLM_PROVIDER
        self.medication_filter = medication_filter
        self.memory_store = memory_store or {}
        self.llm = None
        self.reranking_llm = None  # Cheaper model for ranking
        self.graph = None
        self.tracer = None
        self.atc_manager = ATCManager()  # ATC data manager
        self._initialize()
    
    def _initialize(self) -> None:
        """
        Initialize all core components of the RAG pipeline.
        
        Includes:
        - LLM initialization
        - Graph construction and compilation
        - Opik tracing setup
        """
        logger.info(f"Initializing LangGraph RAG system with provider: {self.provider}")
        
        # Initialize LLM
        self.llm = self._create_llm()
        
        # Initialize cheaper LLM for ranking if re-ranking is enabled
        if Config.ENABLE_RERANKING:
            self.reranking_llm = self._create_reranking_llm()
        
        # Build LangGraph
        graph = StateGraph(DocumentRAGState)
        
        # Add nodes
        graph.add_node("memory", self._create_memory_node)
        graph.add_node("query_analysis", self._create_query_analysis_node)
        graph.add_node("retrieval", self._create_retrieval_node)
        graph.add_node("reranking_decision", self._create_reranking_decision_node)
        graph.add_node("reranking", self._create_reranking_node)
        graph.add_node("fallback", self._create_fallback_node)
        graph.add_node("generation", self._create_generation_node)
        graph.add_node("citation", self._create_citation_node)
        
        # Define edges
        graph.add_edge(START, "memory")
        graph.add_edge("memory", "query_analysis")
        graph.add_edge("query_analysis", "retrieval")
        
        # Conditional edge: check if retrieval was sufficient
        def route_after_retrieval(state: DocumentRAGState) -> str:
            """Route after retrieval based on sufficiency and re-ranking config."""
            if state.get("error"):
                return "fallback"
            if not state.get("retrieval_sufficient", True):
                return "fallback"
            # If re-ranking is enabled, check if it's needed
            if Config.ENABLE_RERANKING:
                return "reranking_decision"
            # Skip re-ranking, go directly to generation
            return "generation"
        
        graph.add_conditional_edges("retrieval", route_after_retrieval)
        
        # Conditional edge: route based on re-ranking decision
        def route_after_decision(state: DocumentRAGState) -> str:
            """Route based on re-ranking decision."""
            if state.get("reranking_needed", False):
                return "reranking"
            return "generation"
        
        graph.add_conditional_edges("reranking_decision", route_after_decision)
        graph.add_edge("reranking", "generation")
        graph.add_edge("fallback", "citation")
        graph.add_edge("generation", "citation")
        graph.add_edge("citation", END)
        
        # Compile graph
        self.graph = graph.compile()
        
        # Configure Opik tracing
        self.tracer = self._configure_opik()
        
        logger.info("LangGraph RAG system initialized successfully")
    
    def _create_llm(self) -> Any:
        """
        Create LLM instance based on provider.
        
        Returns:
            LLM instance (ChatGoogleGenerativeAI or ChatOpenAI)
            
        Raises:
            ValueError: If provider is invalid or API key is missing
        """
        provider = self.provider.lower()
        
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
            valid_providers = ", ".join(Config.LLM_MODELS.keys())
            raise ValueError(
                f"Invalid provider: {provider}. Must be one of: {valid_providers}"
            )
    
    def _create_reranking_llm(self) -> Any:
        """
        Create cheaper LLM instance for re-ranking decisions and document scoring.
        
        Uses a faster/cheaper model (default: gemini-2.5-flash) for ranking tasks.
        
        Returns:
            LLM instance for ranking
            
        Raises:
            ValueError: If API key is missing
        """
        model_name = Config.RERANKING_MODEL
        
        # Determine provider from model name
        if "gemini" in model_name.lower():
            if not Config.GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY is required for Gemini re-ranking model")
            logger.info(f"Creating Gemini re-ranking LLM with model: {model_name}")
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=Config.GOOGLE_API_KEY,
                temperature=0.1,  # Low temperature for consistent scoring
            )
        elif "gpt" in model_name.lower() or "o1" in model_name.lower():
            if not Config.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is required for OpenAI re-ranking model")
            logger.info(f"Creating OpenAI re-ranking LLM with model: {model_name}")
            return ChatOpenAI(
                model=model_name,
                openai_api_key=Config.OPENAI_API_KEY,
                temperature=0.1,  # Low temperature for consistent scoring
            )
        else:
            # Default to Gemini if model name doesn't match known patterns
            logger.warning(f"Unknown re-ranking model: {model_name}, defaulting to Gemini")
            if not Config.GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY is required for default Gemini re-ranking model")
            return ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=Config.GOOGLE_API_KEY,
                temperature=0.1,
            )
    
    def _configure_opik(self) -> Optional[OpikTracer]:
        """
        Configure Opik and create OpikTracer if API key is set.
        
        Returns:
            OpikTracer instance if Opik is configured, None otherwise
        """
        if not Config.OPIK_API_KEY:
            logger.warning("OPIK_API_KEY not set. Opik tracing will be disabled.")
            return None
        
        try:
            import os
            
            # Set environment variables for Opik to read
            os.environ["OPIK_API_KEY"] = Config.OPIK_API_KEY
            if Config.OPIK_PROJECT_NAME:
                os.environ["OPIK_PROJECT_NAME"] = Config.OPIK_PROJECT_NAME
            
            # Create OpikTracer with graph-aware tracing
            tracer = OpikTracer(
                graph=self.graph.get_graph(xray=True),
                project_name=Config.OPIK_PROJECT_NAME
            )
            logger.info(f"Opik tracing enabled (project: {Config.OPIK_PROJECT_NAME})")
            return tracer
        except Exception as e:
            logger.warning(f"Failed to configure Opik: {e}", exc_info=True)
            return None
    
    # Define memory node
    def _create_memory_node(self, state: DocumentRAGState) -> Dict[str, Any]:
        """
        Load conversation history from memory if session_id is provided.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with chat_history
        """
        session_id = state.get("session_id")
        chat_history = ""
        
        if session_id and session_id in self.memory_store:
            memory = self.memory_store[session_id]
            logger.debug(f"Loading memory for session: {session_id}")
            
            # Load memory variables
            memory_vars = memory.load_memory_variables({})
            history = memory_vars.get("chat_history", [])
            
            if history:
                # Format history for prompt (last 4 exchanges)
                history_str = "\n".join([
                    f"Spurning: {msg.content}" if hasattr(msg, 'type') and msg.type == 'human' 
                    else f"Svar: {msg.content}"
                    for msg in history[-4:]
                ])
                chat_history = history_str
                logger.debug(f"Loaded {len(history)} messages from memory")
        
        return {"chat_history": chat_history}
    
    # Define query analysis node
    def _create_query_analysis_node(self, state: DocumentRAGState) -> Dict[str, Any]:
        """
        Extract medication name(s) from query if mentioned.
        
        If no medication is found in the current question, checks chat history
        to maintain context for follow-up questions.
        
        Supports both single medication and multiple medications for comparison queries.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with extracted medication information
        """
        question = state["question"]
        medication_filter = state.get("medication_filter") or self.medication_filter
        chat_history = state.get("chat_history", "")
        
        # Only extract if not already provided
        if not medication_filter:
            from src.query_disambiguation import find_matching_medications
            
            # First, try to extract from current question
            matches = find_matching_medications(question, self.vector_store_manager)
            
            # If no matches in current question, check chat history
            if not matches and chat_history:
                logger.debug("No medication found in current question, checking chat history")
                matches = find_matching_medications(chat_history, self.vector_store_manager)
                if matches:
                    logger.info(f"Found medication(s) in chat history: {matches}")
            
            if len(matches) == 1:
                # Single medication
                medication_filter = matches[0]
                logger.info(f"Auto-extracted single medication: {medication_filter}")
                return {
                    "medication_filter": medication_filter,
                    "extracted_medication": medication_filter,
                    "extracted_medications": []
                }
            elif len(matches) >= 2:
                # Multiple medications (comparison query)
                logger.info(f"Auto-extracted multiple medications: {matches}")
                return {
                    "medication_filter": None,  # Clear single filter
                    "extracted_medication": None,
                    "extracted_medications": matches  # List of medications
                }
        
        # Medication filter already provided or no matches found
        return {
            "medication_filter": medication_filter,
            "extracted_medication": medication_filter if medication_filter else None,
            "extracted_medications": []
        }
    
    # Define retrieval node
    def _create_retrieval_node(self, state: DocumentRAGState) -> Dict[str, Any]:
        """
        Retrieve relevant documents based on the user's question.
        
        Handles single medication, multiple medications, and no medication queries.
        Retrieves more documents initially for re-ranking.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with retrieved_docs, formatted_context, and retrieval_sufficient flag
        """
        try:
            question = state["question"]
            medication_filter = state.get("medication_filter") or self.medication_filter
            extracted_medications = state.get("extracted_medications", [])
            
            logger.info(f"Retrieving documents for question: {question[:50]}...")
            
            # Multi-medication query (comparison queries) - PARALLELIZED
            if extracted_medications:
                all_docs = []
                
                def retrieve_for_medication(med: str) -> List[Document]:
                    """Retrieve documents for a single medication."""
                    logger.info(f"Retrieving from medication: {med}")
                    retriever = self.vector_store_manager.get_retriever_with_filter(
                        medication_name=med,
                        top_k=Config.RETRIEVAL_MULTI_MED_K
                    )
                    try:
                        docs = retriever.invoke(question)
                    except AttributeError:
                        # Fallback for older LangChain versions
                        docs = retriever.get_relevant_documents(question)
                    logger.info(f"Retrieved {len(docs)} chunks for {med}")
                    return docs
                
                # Parallelize retrieval across medications
                with ThreadPoolExecutor(max_workers=min(len(extracted_medications), 5)) as executor:
                    future_to_med = {
                        executor.submit(retrieve_for_medication, med): med 
                        for med in extracted_medications
                    }
                    for future in as_completed(future_to_med):
                        med = future_to_med[future]
                        try:
                            docs = future.result()
                            all_docs.extend(docs)
                        except Exception as e:
                            logger.error(f"Error retrieving for {med}: {e}", exc_info=True)
                
                # Check if retrieval is sufficient
                if len(all_docs) < Config.RETRIEVAL_MIN_DOCS:
                    logger.warning(f"Insufficient documents retrieved: {len(all_docs)} < {Config.RETRIEVAL_MIN_DOCS}")
                    return {
                        "retrieved_docs": [],
                        "formatted_context": "",
                        "retrieval_sufficient": False
                    }
                
                formatted_context = self._format_context(all_docs)
                return {
                    "retrieved_docs": all_docs,
                    "formatted_context": formatted_context,
                    "retrieval_sufficient": True
                }
            
            # Single medication or no medication query
            if medication_filter:
                logger.info(f"Single medication filter: {medication_filter}")
                retriever = self.vector_store_manager.get_retriever_with_filter(
                    medication_name=medication_filter,
                    top_k=Config.RETRIEVAL_INITIAL_K
                )
            else:
                logger.info("No medication filter - retrieving from all documents")
                retriever = self.vector_store_manager.get_retriever(
                    top_k=Config.RETRIEVAL_INITIAL_K
                )
            
            # Perform retrieval
            try:
                docs = retriever.invoke(question)
            except AttributeError:
                # Fallback for older LangChain versions
                docs = retriever.get_relevant_documents(question)
            
            logger.info(f"Retrieved {len(docs)} matching chunks")
            
            # Check if retrieval is sufficient
            if len(docs) < Config.RETRIEVAL_MIN_DOCS:
                logger.warning(f"Insufficient documents retrieved: {len(docs)} < {Config.RETRIEVAL_MIN_DOCS}")
                return {
                    "retrieved_docs": [],
                    "formatted_context": "",
                    "retrieval_sufficient": False
                }
            
            # Format context for LLM prompt
            formatted_context = self._format_context(docs)
            
            # Opik tracing is handled automatically by OpikTracer callback
            
            return {
                "retrieved_docs": docs,
                "formatted_context": formatted_context,
                "retrieval_sufficient": True
            }
        
        except Exception as e:
            logger.error(f"Error in retrieval node: {e}", exc_info=True)
            return {
                "error": f"Villa kom upp við að sækja skjöl: {str(e)}",
                "retrieved_docs": [],
                "formatted_context": "",
                "retrieval_sufficient": False
            }
    
    def _format_context(self, docs: List[Document], include_atc: bool = True) -> str:
        """
        Format documents into context string for LLM prompt.
        
        Args:
            docs: List of Document objects
            include_atc: If True, include ATC context for drugs
            
        Returns:
            Formatted context string
        """
        formatted_context_parts = []
        seen_drugs = set()  # Track drugs we've added ATC info for
        
        for idx, doc in enumerate(docs, 1):
            metadata = doc.metadata
            drug_id = metadata.get("drug_id", metadata.get("medication_name", "Unknown"))
            section_num = metadata.get("section_number", "Unknown")
            section_title = metadata.get("section_title", metadata.get("section", "Unknown"))
            
            context_block = f"""
Kafli {section_num}: {section_title}
Lyf: {drug_id}
Innihald:
{doc.page_content}
"""
            formatted_context_parts.append(context_block)
            
            # Add ATC context for first occurrence of each drug
            if include_atc and drug_id not in seen_drugs and drug_id != "Unknown":
                atc_context = self.atc_manager.format_atc_context_for_rag(
                    drug_id,
                    include_alternatives=False  # Don't include alternatives in main context
                )
                if atc_context:
                    formatted_context_parts.append(f"ATC upplýsingar fyrir {drug_id}:\n{atc_context}")
                seen_drugs.add(drug_id)
        
        return "\n---\n".join(formatted_context_parts)
    
    # Define re-ranking decision node
    def _create_reranking_decision_node(self, state: DocumentRAGState) -> Dict[str, Any]:
        """
        Decide if re-ranking is needed using structured output (YES/NO).
        
        Uses a single LLM call with structured output to determine if re-ranking
        would improve results. Considers query complexity, number of documents,
        and multi-medication queries.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with reranking_needed flag
        """
        # Check if re-ranking is enabled
        if not Config.ENABLE_RERANKING:
            return {"reranking_needed": False}
        
        # Check for errors or insufficient retrieval
        if state.get("error") or not state.get("retrieval_sufficient", True):
            return {"reranking_needed": False}
        
        docs = state.get("retrieved_docs", [])
        question = state["question"]
        extracted_medications = state.get("extracted_medications", [])
        
        # Quick checks: if too few docs, skip re-ranking
        if len(docs) <= Config.RETRIEVAL_TOP_K:
            logger.info(f"Only {len(docs)} documents, skipping re-ranking decision")
            return {"reranking_needed": False}
        
        # Quick check: multi-medication queries always benefit from re-ranking
        if extracted_medications and len(extracted_medications) >= 2:
            logger.info("Multi-medication query detected, re-ranking needed")
            return {"reranking_needed": True}
        
        # Use LLM with structured output to decide
        try:
            # Create parser for structured output
            parser = PydanticOutputParser(pydantic_object=RerankingDecision)
            format_instructions = parser.get_format_instructions()
            
            # Create decision prompt
            decision_prompt = f"""Ákvarða hvort endurröðun (re-ranking) sé nauðsynleg fyrir eftirfarandi spurningu.

Leiðbeiningar:
- JÁ (YES) ef spurningin er samanburður (t.d. "munur", "samanburður", "og", "bæði")
- JÁ (YES) ef spurningin er flókin eða margþætt (t.d. "af hverju", "hvernig", "útskýrðu")
- JÁ (YES) ef fleiri en {Config.RERANKING_DECISION_THRESHOLD} skjöl voru sótt
- NEI (NO) ef spurningin er einföld staðreyndaspurning með fáum skjölum

Spurning: {question}
Fjöldi sóttra skjala: {len(docs)}

{format_instructions}

Svaraðu með JSON sniði sem passar við Pydantic líkanið."""
            
            # Use cheaper LLM for decision if available, otherwise use main LLM
            decision_llm = self.reranking_llm if self.reranking_llm else self.llm
            
            # Create chain: prompt -> LLM -> parser
            prompt_template = PromptTemplate(
                template=decision_prompt,
                input_variables=[],
            )
            chain = prompt_template | decision_llm | parser
            
            # Get structured decision
            decision_result = chain.invoke({})
            reranking_needed = decision_result.decision == BinaryDecision.YES
            
            logger.info(f"Re-ranking decision: {decision_result.decision.value} (needed: {reranking_needed})")
            
            return {"reranking_needed": reranking_needed}
            
        except Exception as e:
            logger.warning(f"Error in re-ranking decision: {e}, defaulting to NO", exc_info=True)
            # Default to no re-ranking on error
            return {"reranking_needed": False}
    
    # Define re-ranking node
    def _create_reranking_node(self, state: DocumentRAGState) -> Dict[str, Any]:
        """
        Re-rank retrieved documents by relevance to the query.
        
        Uses LLM to score each document's relevance, then selects top-k most relevant.
        For multi-medication queries, ensures balanced representation.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with re-ranked documents and formatted context
        """
        # Check for errors or insufficient retrieval
        if state.get("error") or not state.get("retrieval_sufficient", True):
            return {}
        
        docs = state.get("retrieved_docs", [])
        question = state["question"]
        extracted_medications = state.get("extracted_medications", [])
        
        # Determine target number of documents after re-ranking
        # For multi-medication queries, use more documents to ensure representation
        target_k = 10 if extracted_medications else Config.RETRIEVAL_TOP_K
        
        # No need to re-rank if we already have fewer docs than final target
        if len(docs) <= target_k:
            logger.info(f"Only {len(docs)} documents, skipping re-ranking")
            return {}
        
        logger.info(f"Re-ranking {len(docs)} documents to select top-{target_k}")
        
        # Use cheaper LLM for ranking if available
        ranking_llm = self.reranking_llm if self.reranking_llm else self.llm
        
        # Score each document using LLM
        scored_docs = []
        for doc in docs:
            try:
                score = self._score_relevance(question, doc, ranking_llm)
                scored_docs.append((score, doc))
            except Exception as e:
                logger.warning(f"Error scoring document: {e}")
                # Default score of 0 if scoring fails
                scored_docs.append((0.0, doc))
        
        # Sort by score (descending)
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # For multi-medication queries, ensure balanced representation
        if extracted_medications and len(extracted_medications) >= 2:
            # Group by medication
            med_groups: Dict[str, List[tuple]] = {med: [] for med in extracted_medications}
            other_docs = []
            
            for score, doc in scored_docs:
                doc_med = doc.metadata.get("drug_id") or doc.metadata.get("medication_name", "")
                found = False
                for med in extracted_medications:
                    if med.lower() in doc_med.lower() or doc_med.lower() in med.lower():
                        med_groups[med].append((score, doc))
                        found = True
                        break
                if not found:
                    other_docs.append((score, doc))
            
            # Take top-k per medication, then fill remaining slots
            top_docs = []
            per_med_k = max(1, target_k // len(extracted_medications))
            
            for med in extracted_medications:
                med_docs = med_groups[med][:per_med_k]
                top_docs.extend(med_docs)
            
            # Fill remaining slots with highest scoring docs
            remaining = target_k - len(top_docs)
            if remaining > 0:
                all_remaining = []
                for med_docs in med_groups.values():
                    all_remaining.extend(med_docs[per_med_k:])
                all_remaining.extend(other_docs)
                all_remaining.sort(key=lambda x: x[0], reverse=True)
                top_docs.extend(all_remaining[:remaining])
            
            # Final sort by score
            top_docs.sort(key=lambda x: x[0], reverse=True)
            top_docs = top_docs[:target_k]
        else:
            # Single medication or no medication - just take top-k
            top_docs = scored_docs[:target_k]
        
        # Extract documents from scored tuples
        ranked_docs = [doc for _, doc in top_docs]
        
        # Re-format context with re-ranked documents
        formatted_context = self._format_context(ranked_docs)
        
        logger.info(f"Re-ranking complete. Selected {len(ranked_docs)} most relevant documents")
        
        return {
            "retrieved_docs": ranked_docs,
            "formatted_context": formatted_context
        }
    
    def _score_relevance(self, question: str, doc: Document, ranking_llm: Optional[Any] = None) -> float:
        """
        Score a document's relevance to the question using LLM with anchored rubric.
        
        Uses a detailed rubric with clear anchor points for each relevance level,
        improving inter-rater reliability and reducing subjectivity compared to
        unanchored numerical scales. The scale deliberately omits 3 to filter out
        average answers that are irrelevant.
        
        Args:
            question: User question
            doc: Document to score
            ranking_llm: Optional LLM to use for ranking (defaults to self.llm)
            
        Returns:
            Relevance score (0.0 to 5.0)
        """
        # Anchored rubric for relevance scoring
        # Scale: 0,1,2,4,5 (deliberately omitting 3 to filter out average/irrelevant)
        rubric = """Notaðu eftirfarandi skilgreiningar til að meta mikilvægi textans:

5 - Fullkomið: Textinn inniheldur ALLAR upplýsingarnar og svarar spurningunni algjörlega með öllum nauðsynlegum upplýsingum.

4 - Rétt en ófullkomið: Textinn er réttur og svarar spurningunni, en getur vantað nokkrar upplýsingar eða smáatriði.

2 - Lauslega tengt: Textinn er lauslega tengdur spurningunni en svarar ekki beint eða er ekki nógu nákvæmur.

1 - Fyrir lyfið en ekki tengt spurningunni: Textinn er fyrir rétt lyf/lyfjameðal, en inniheldur ekki upplýsingar sem tengjast spurningunni.

0 - Ekki tengt lyfinu: Textinn er ekki tengdur þessu lyfi/lyfjameðali eða inniheldur engar gagnlegar upplýsingar fyrir spurninguna.

ATHUGIÐ: Ekki nota tölustafinn 3. Ef textinn er "meðal" í mikilvægi, veldu annað hvort 2 (lauslega tengt) eða 4 (rétt en ófullkomið) eftir því sem við á."""

        score_prompt = f"""{rubric}

Spurning: {question}

Texti:
{doc.page_content[:500]}...

Svaraðu EINUNGIS með tölustaf (0, 1, 2, 4, eða 5) sem passar best við skilgreiningarnar hér að ofan. Ekki nota tölustafinn 3 eða aðra tölustafi."""

        # Use provided ranking LLM or default to main LLM
        llm_to_use = ranking_llm if ranking_llm else self.llm
        
        try:
            response = llm_to_use.invoke(score_prompt)
            if hasattr(response, "content"):
                score_text = response.content.strip()
            else:
                score_text = str(response).strip()
            
            # Extract numeric score - only accept valid rubric scores
            match = re.search(r'\b(0|1|2|4|5)\b', score_text)
            if match:
                score = float(match.group())
                return min(5.0, max(0.0, score))
            # Fallback: try to extract any number and map to nearest rubric value
            match = re.search(r'\d+', score_text)
            if match:
                raw_score = float(match.group())
                # Map to nearest rubric value (excluding 3)
                rubric_values = [0, 1, 2, 4, 5]
                score = min(rubric_values, key=lambda x: abs(x - raw_score))
                return float(score)
            return 2.0  # Default to "loosely related" if parsing fails
        except Exception as e:
            logger.warning(f"Error scoring document: {e}")
            return 2.0  # Default score on error
    
    # Define fallback node
    def _create_fallback_node(self, state: DocumentRAGState) -> Dict[str, Any]:
        """
        Generate helpful fallback response when retrieval fails.
        
        Provides guidance to users suggesting they check serlyfjaskra.is for official information.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with fallback answer and empty sources
        """
        question = state["question"]
        extracted_medication = state.get("extracted_medication")
        extracted_medications = state.get("extracted_medications", [])
        
        # Build list of medications mentioned
        medications = []
        if extracted_medication:
            medications = [extracted_medication]
        elif extracted_medications:
            medications = extracted_medications
        
        # Build medication text for message
        medication_text = ""
        if medications:
            medication_text = f" fyrir {', '.join(medications)}"
        
        # Generate helpful fallback message in Icelandic
        fallback_message = f"""Ég fann ekki nægilega upplýsingar{medication_text} í tiltækum skjölum.

Vinsamlegast athugaðu opinberar lyfjaupplýsingar (SmPC) á serlyfjaskra.is fyrir nákvæmari og tæmandi upplýsingar.

Ef þú leitar að upplýsingum um tiltekið lyf, getur þú notað leitina á serlyfjaskra.is með lyfjanafninu."""
        
        logger.info("Generating fallback response - insufficient documents retrieved")
        
        return {
            "answer": fallback_message,
            "sources": [],
            "retrieval_sufficient": False
        }
    
    # Define generation node
    def _create_generation_node(self, state: DocumentRAGState) -> Dict[str, Any]:
        """
        Generate answer using retrieved documents and conversation history.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with answer
        """
        # Check for errors from previous nodes
        if state.get("error"):
            return {"answer": state["error"]}
        
        # Check if retrieval was sufficient
        if not state.get("retrieval_sufficient", True):
            # Retrieval failed - fallback node should have handled this
            # But if we reach here, return empty answer
            return {"answer": ""}
        
        try:
            question = state["question"]
            context = state.get("formatted_context", "")
            chat_history = state.get("chat_history", "")
            
            # Check if query asks for alternatives
            question_lower = question.lower()
            asks_for_alternatives = any(
                phrase in question_lower
                for phrase in ["valkostir", "aðrar lyf", "sambærileg lyf", "alternatives", "similar drugs"]
            )
            
            # Add ATC-based alternatives if requested
            if asks_for_alternatives:
                extracted_medication = state.get("extracted_medication")
                if extracted_medication:
                    alternatives = self.atc_manager.get_alternatives(extracted_medication, max_results=5)
                    if alternatives:
                        alt_context = f"\n\nValkostir með sama ATC flokk:\n"
                        for alt_drug in alternatives:
                            alt_atc = self.atc_manager.get_atc_codes_for_drug(alt_drug)
                            alt_context += f"- {alt_drug} (ATC: {', '.join(alt_atc) if alt_atc else 'Ekki þekkt'})\n"
                        context += alt_context
            
            # Create prompt template
            if chat_history:
                prompt_template = ICELANDIC_SYSTEM_PROMPT
                input_variables = ["context", "question", "history"]
            else:
                # Simplified prompt without history
                prompt_template = ICELANDIC_SYSTEM_PROMPT.replace("{history}\n\n", "")
                input_variables = ["context", "question"]
                chat_history = ""
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=input_variables
            )
            
            # Prepare prompt input
            prompt_input = {
                "context": context,
                "question": question
            }
            if chat_history:
                prompt_input["history"] = chat_history
            
            # Generate answer
            logger.info(f"Generating answer for question: {question[:50]}...")
            
            chain = prompt | self.llm
            response = chain.invoke(prompt_input)
            
            # Extract answer text
            if hasattr(response, "content"):
                answer = response.content
            else:
                answer = str(response)
            
            logger.info(f"Generated answer (length: {len(answer)})")
            
            # Opik tracing is handled automatically by OpikTracer callback
            
            return {"answer": answer}
        
        except Exception as e:
            logger.error(f"Error in generation node: {e}", exc_info=True)
            return {
                "error": f"Villa kom upp við að búa til svar: {str(e)}",
                "answer": "Ég get ekki svarað þessari spurningu vegna villa."
            }
    
    # Define citation node
    def _create_citation_node(self, state: DocumentRAGState) -> Dict[str, Any]:
        """
        Ensure answer includes proper citations and extract source metadata.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with answer (with citations) and sources
        """
        # Check for errors from previous nodes
        if state.get("error"):
            return {"answer": state["error"], "sources": []}
        
        try:
            answer = state.get("answer", "")
            retrieved_docs = state.get("retrieved_docs", [])
            
            # Extract source information with full metadata
            sources = []
            for doc in retrieved_docs:
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
            
            # Ensure citations are present in answer
            answer_with_citations = self._ensure_citations(answer, sources)
            
            logger.info(f"Citation processing complete. Sources: {len(sources)}")
            
            return {
                "answer": answer_with_citations,
                "sources": sources
            }
        
        except Exception as e:
            logger.error(f"Error in citation node: {e}", exc_info=True)
            return {
                "answer": state.get("answer", ""),
                "sources": [],
                "error": f"Villa kom upp við að bæta við tilvísunum: {str(e)}"
            }
    
    def _ensure_citations(self, answer: str, sources: List[Dict[str, Any]]) -> str:
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
    
    def process_message(
        self,
        question: str,
        session_id: Optional[str] = None,
        medication_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute the full RAG flow with explicit state management.
        
        Args:
            question: Question to ask (in Icelandic)
            session_id: Optional session ID for conversation memory
            medication_filter: Optional medication name to filter retrieval
            
        Returns:
            Dictionary with "answer", "sources", and "error" keys
        """
        try:
            # Prepare initial state
            initial_state: DocumentRAGState = {
                "question": question,
                "medication_filter": medication_filter or self.medication_filter,
                "extracted_medication": None,
                "extracted_medications": [],
                "session_id": session_id,
                "retrieved_docs": [],
                "formatted_context": "",
                "chat_history": "",
                "answer": "",
                "sources": [],
                "retrieval_sufficient": True,
                "reranking_needed": None,
                "error": None
            }
            
            # Prepare callbacks
            callbacks = []
            if self.tracer:
                callbacks.append(self.tracer)
            
            # Invoke graph
            logger.info(f"Processing message: {question[:50]}...")
            result = self.graph.invoke(
                initial_state,
                config={"callbacks": callbacks} if callbacks else None
            )
            
            # Extract results
            answer = result.get("answer", "Ég get ekki svarað þessari spurningu.")
            sources = result.get("sources", [])
            error = result.get("error")
            
            # Save to memory if session_id provided
            if session_id and not error:
                if session_id not in self.memory_store:
                    self.memory_store[session_id] = ConversationBufferMemory(
                        return_messages=True
                    )
                
                memory = self.memory_store[session_id]
                memory.save_context(
                    {"input": question},
                    {"output": answer}
                )
                logger.debug(f"Saved conversation to memory for session: {session_id}")
            
            if error:
                logger.warning(f"Error in RAG flow: {error}")
            
            return {
                "answer": answer,
                "sources": sources,
                "error": error
            }
        
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            return {
                "answer": f"Villa kom upp við að svara spurningu: {str(e)}",
                "sources": [],
                "error": str(e)
            }


def create_rag_graph(
    vector_store_manager: VectorStoreManager,
    provider: str = None,
    medication_filter: Optional[str] = None,
    memory_store: Optional[Dict[str, ConversationBufferMemory]] = None
) -> DocumentRAGGraph:
    """
    Factory function to create DocumentRAGGraph instance.
    
    Args:
        vector_store_manager: VectorStoreManager instance
        provider: LLM provider ("gemini" or "gpt5")
        medication_filter: Optional medication name to filter retrieval
        memory_store: Optional dictionary mapping session_id to ConversationBufferMemory
        
    Returns:
        DocumentRAGGraph instance
    """
    return DocumentRAGGraph(
        vector_store_manager=vector_store_manager,
        provider=provider,
        medication_filter=medication_filter,
        memory_store=memory_store
    )


def query_rag_graph(
    rag_graph: DocumentRAGGraph,
    question: str,
    session_id: Optional[str] = None,
    medication_filter: Optional[str] = None
) -> Dict[str, Any]:
    """
    Query RAG graph with a question.
    
    Args:
        rag_graph: DocumentRAGGraph instance
        question: Question to ask (in Icelandic)
        session_id: Optional session ID for conversation memory
        medication_filter: Optional medication name filter (for logging)
        
    Returns:
        Dictionary with "answer", "sources", and "error" keys
    """
    return rag_graph.process_message(
        question=question,
        session_id=session_id,
        medication_filter=medication_filter
    )
