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
from typing import Dict, Any, List, Optional, Annotated
from typing_extensions import TypedDict
from operator import add
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, START, END
from opik.integrations.langchain import OpikTracer
from src.query_disambiguation import normalize_icelandic, strip_diacritics

from config import Config
from src.vector_store import VectorStoreManager
from src.atc_manager import ATCManager
from src.ingredients_manager import IngredientsManager

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


class RewriteState(BaseModel):
    """Structured output for query rewrite."""
    original_query: str = Field(description="Original query as asked by the user")
    rewritten_query: str = Field(description="Rewritten query in correct Icelandic medical language")
    detected_ingredients: List[str] = Field(default_factory=list, description="List of detected active ingredient names (INN names)")
    detected_drugs: List[str] = Field(default_factory=list, description="List of detected drug/brand names")
    suggested_sections: List[str] = Field(default_factory=list, description="List of relevant SmPC section numbers (e.g., '4.1', '4.2', '4.3')")

def norm_for_matching(s: str) -> str:
    return strip_diacritics(normalize_icelandic(s))

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

# NB
- Hafðu í huga að sum lyf eru til í mismunandi styrk og á mismunandi formi. Stundum er tiltekinn styrkur og form ætlað mismunandi sjúklingum. Því þarf að tilgreina hvaða lyf er átt við í svarinu.

# OUTPUT REQUIREMENTS
- Byrjaðu alltaf á tveimur setningum með því mikilvægasta sem þarf að segja. Merktu þetta sem "AÐALATRIÐI:"
- Bættu svo við bili áður en þú bætir við næstu upplýsingum.
- Eftir fyrstu tvær setningarnar, bættu við nauðsynlegum upplýsingum sem þarf að segja. Merktu þetta sem "FREKARI UPPLÝSINGAR:"
- Svörin skulu vera skýr, á faglegri íslensku
- Endaðu öll svör með því að vísa lesandanum til SmPC texta. 

{history}

Upplýsingar úr skjölum: {context}

Spurning: {question}

Svar með tilvísunum (MUST include citations in format [drug_id, kafli section_number: section_title]):"""


# Query rewrite prompt for improving retrieval accuracy
QUERY_REWRITE_PROMPT = """Þú ert klínískur lyfjafræðingur sem endursemur íslenskar lyfjaspurningar
í skýrar, formlegar SmPC leitaspurningar.

Markmið:
- Normalisera spurninguna í réttri íslenskri lyfjatungumál
- Greina tilgang spyrjanda
- Veldu að hámarki 2 viðeigandi SmPC kafla sem á að nota til að svara spurningunni
- Skila í JSON sem passar við Pydantic líkanið

Mikilvægt:
- Ekki svara spurningunni
- Ekki veita ráðleggingar
- Ekki gefa klínískar leiðbeiningar
- Ekki bæta við nýjum læknisfræðilegum upplýsingum sem eru ekki í spurningunni
- Ekki finna upp nýja kafla
- Ef enginn kafli á við → skilaðu [] fyrir lista af köflum
- Ef spurningin er óljós → notaðu innihald spurningar til að átta sig á tilgangi

Notaðu einungis eftirfarandi lýsingar til að meta hvaða kaflar eiga við.
Ekki draga ályktanir út fyrir þessar skilgreiningar.
SmPC kaflar og merking þeirra (samantekt):

1. NAFN LYFS
    - Opinbert nafn vörunnar
    - Ekki klínískt efni fyrir spurningar

2. INNIHALDSEFNALÝSING
    - Virk innihaldsefni og styrkur
    - Hjálparefni
    - Samsvörun við INN heiti

3. LYFJAFORM
    - Lyfjaform (t.d. tafla, mixtúra, lausn)
    - Útlit og eðli forms

4. KLÍNÍSKAR UPPLÝSINGAR

  4.1 Ábendingar
      - Fyrir hvaða sjúkdóma/ástand lyfið er notað
      - Aldurshópar sem eru ábendingar

  4.2 Skammtar og lyfjagjöf
      - Ráðlagðir skammtar
      - Börn, fullorðnir, aldraðir
      - Leið lyfjagjafar (PO, IV, o.s.frv.)
      - Sértækar breytingar (nýrna-/lifrarstarfsemi)

  4.3 Frábendingar
      - Hvenær lyfið má ekki nota
      - Áhætta þar sem notkun er bönnuð

  4.4 Sérstök varúð og viðvaranir
      - Áhættustjórnun
      - Eftirlit
      - Varúðarráðstafanir
      - Áhættuþættir
      - Notkun við sértækum sjúklingahópum

  4.5 Lyfjamilliverkanir
      - Milliverkanir við önnur lyf, fæðu, áfengi
      - Lyfhrif/lyfjahvarfa samskipti
      - CYP áhrif ef við á

  4.6 Meðganga, brjóstagjöf og frjósemi
      - Notkun á meðgöngu
      - Brjóstagjöf
      - Áhrif á frjósemi

  4.7 Áhrif á akstur og notkun véla
      - Áhrif á einbeitingu og hreyfifærni

  4.8 Aukaverkanir
      - Aukaverkanir og tíðni
      - Klínískar rannsóknir
      - Eftirmarkaðs gögn

  4.9 Ofskömmtun
      - Eitrunarmerki
      - Meðferð við ofskömmtun

5. LYFJAFRÆÐILEGAR UPPLÝSINGAR
  5.1 Lyfhrif (pharmacodynamics)
      - Virknimekanismi
      - Klínískar niðurstöður

  5.2 Lyfjahvörf (pharmacokinetics)
      - Frásog, dreifing, umbrot, útskilnaður
      - Sérstakir sjúklingahópar

  5.3 Forklínískar upplýsingar
      - Dýratilraunir
      - Eiturhrif

6. LYFJAGERÐAUPPLÝSINGAR
    - Hjálparefni
    - Geymsluskilyrði
    - Pakkningar
    - Úrgangsmeðhöndlun

7. MARKAÐSLEYFISHAFI
    - Nafn, heimilisfang
    - Ekki viðeigandi fyrir klínískar spurningar

8. MARKAÐSLEYFISNÚMER
    - Ekki viðeigandi fyrir spurningar

9. DAGSETNING FYRSTU ÚTGÁFU / ENDURNÝJUNAR
    - Ekki viðeigandi fyrir spurningar


⭐ Notkun:
- Ef spurning snýst um "hvað er lyfið fyrir?" → 4.1
- Ef spurning snýst um "skammtar? börn? aldraðir?" → 4.2
- Ef spurning snýst um "má nota við X sjúkdómi?" → 4.3 eða 4.4 (eftir merkingu)
- Ef spurning snýst um meðgöngu/brjóstagjöf → 4.6
- Ef spurning snýst um aukaverkanir → 4.8
- Ef spurning snýst um frásog/helmingunartíma → 5.2
- Ef spurning snýst um virknimekanisma → 5.1
- Ef spurning snýst um milliverkanir → 4.5
- Ef spurning er aðeins um heiti lyfs → enginn kafli (skila [])

Notaðu fyrirspurnarsögu ef um er að ræða framhalsspurningu:
{chat_history_context}

Upprunaleg spurning:
{question}

Svaraðu með JSON sem passar við Pydantic líkanið:
{format_instructions}"""



# Define drug entity structure for per-drug tracking
class DrugEntity(TypedDict, total=False):
    """Represents a single drug entity with medication name and/or active ingredients.
    
    All fields are optional (total=False) to allow flexible creation.
    Code should use .get() with defaults when accessing fields.
    """
    medication_name: Optional[str]  # Brand name with format/strength if specific
    active_ingredients: List[str]  # Active ingredient names (INN names)
    is_generic: bool  # True if brand name is generic (no format/strength specified)
    brand_variants: Optional[List[str]]  # All related brand names sharing the same active ingredients (optional)


# Define the LangGraph state (shared between nodes)
class DocumentRAGState(TypedDict):
    """State shared between LangGraph nodes."""
    question: str
    medication_filter: Optional[str]  # Optional medication name filter (legacy, kept for backward compatibility)
    extracted_medication: Optional[str]  # Single medication extracted from query (legacy)
    extracted_medications: List[str]  # Multiple medications extracted from query (legacy)
    drug_entities: List[DrugEntity]  # Per-drug tracking with medication_name and active_ingredients
    session_id: Optional[str]  # Session ID for conversation memory
    retrieved_docs: Annotated[List[Document], add]  # List of retrieved documents
    formatted_context: str  # Formatted context string for LLM
    chat_history: str  # Formatted conversation history
    answer: str  # Final answer
    sources: Annotated[List[Dict[str, Any]], add]  # Source metadata
    retrieval_sufficient: bool  # Flag indicating if retrieval found sufficient results
    reranking_needed: Optional[bool]  # Flag indicating if re-ranking is needed (from structured decision)
    rewritten_query: Optional[str]  # Rewritten query from rewrite node (used internally for retrieval)
    rewrite_metadata: Optional[Dict[str, Any]]  # Metadata from rewrite (detected_ingredients, detected_drugs, suggested_sections)
    error: Optional[str]  # Error message if something goes wrong
    similar_drugs: List[str]  # List of similar drugs (brand variants) sharing the same active ingredients


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
        self.rewrite_llm = None  # LLM for query rewriting
        self.graph = None
        self.tracer = None
        self.atc_manager = ATCManager()  # ATC data manager
        self.ingredients_manager = IngredientsManager()  # Ingredients data manager
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
        
        # Initialize LLM for query rewriting if rewrite is enabled
        if Config.ENABLE_QUERY_REWRITE:
            self.rewrite_llm = self._create_rewrite_llm()
        
        # Build LangGraph
        graph = StateGraph(DocumentRAGState)
        
        # Add nodes
        graph.add_node("memory", self._create_memory_node)
        graph.add_node("query_analysis", self._create_query_analysis_node)
        graph.add_node("query_rewrite", self._create_rewrite_node)
        graph.add_node("retrieval", self._create_retrieval_node)
        graph.add_node("reranking_decision", self._create_reranking_decision_node)
        graph.add_node("reranking", self._create_reranking_node)
        graph.add_node("fallback", self._create_fallback_node)
        graph.add_node("generation", self._create_generation_node)
        graph.add_node("citation", self._create_citation_node)
        graph.add_node("extract_similar_drugs", self._create_similar_drugs_node)
        
        # Define edges
        graph.add_edge(START, "memory")
        graph.add_edge("memory", "query_analysis")
        graph.add_edge("query_analysis", "query_rewrite")
        graph.add_edge("query_rewrite", "retrieval")
        
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
        graph.add_edge("citation", "extract_similar_drugs")
        graph.add_edge("extract_similar_drugs", END)
        
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
    
    def _create_rewrite_llm(self) -> Any:
        """
        Create LLM instance for query rewriting.
        
        Uses a faster/cheaper model (default: gemini-2.5-flash) for query rewrite tasks.
        
        Returns:
            LLM instance for rewriting
            
        Raises:
            ValueError: If API key is missing
        """
        model_name = Config.REWRITE_MODEL
        
        # Determine provider from model name
        if "gemini" in model_name.lower():
            if not Config.GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY is required for Gemini rewrite model")
            logger.info(f"Creating Gemini rewrite LLM with model: {model_name}")
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=Config.GOOGLE_API_KEY,
                temperature=0.1,  # Low temperature for consistency
            )
        elif "gpt" in model_name.lower() or "o1" in model_name.lower():
            if not Config.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is required for OpenAI rewrite model")
            logger.info(f"Creating OpenAI rewrite LLM with model: {model_name}")
            return ChatOpenAI(
                model=model_name,
                openai_api_key=Config.OPENAI_API_KEY,
                temperature=0.1,  # Low temperature for consistency
            )
        else:
            # Default to Gemini if model name doesn't match known patterns
            logger.warning(f"Unknown rewrite model: {model_name}, defaulting to Gemini")
            if not Config.GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY is required for default Gemini rewrite model")
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
        
        Preserves drug_entities from previous state to maintain context across follow-up questions.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with chat_history and preserved drug_entities
        """
        session_id = state.get("session_id")
        chat_history = ""
        existing_drug_entities = state.get("drug_entities", [])
        
        logger.info(f"[MEMORY NODE] Processing state with session_id: {session_id}")
        logger.info(f"[MEMORY NODE] Memory store has {len(self.memory_store)} sessions: {list(self.memory_store.keys())}")
        logger.info(f"[MEMORY NODE] Existing drug_entities: {len(existing_drug_entities)}")
        
        if not session_id:
            logger.info("[MEMORY NODE] No session_id provided, skipping memory loading")
            # Preserve existing drug_entities even without session
            return {
                "chat_history": chat_history,
                "drug_entities": existing_drug_entities
            }
        
        if session_id not in self.memory_store:
            logger.info(f"[MEMORY NODE] Session {session_id} not found in memory_store")
            # Preserve existing drug_entities
            return {
                "chat_history": chat_history,
                "drug_entities": existing_drug_entities
            }
        
        try:
            memory = self.memory_store[session_id]
            logger.info(f"[MEMORY NODE] Loading memory for session: {session_id}")
            
            # Load memory variables
            memory_vars = memory.load_memory_variables({})
            logger.info(f"[MEMORY NODE] Memory variables keys: {list(memory_vars.keys())}")
            
            # Try both possible keys (chat_history is default, but some configs use history)
            history = memory_vars.get("chat_history", memory_vars.get("history", []))
            logger.info(f"[MEMORY NODE] Found {len(history)} messages in memory")
            
            if history:
                # Format history for prompt (last 4 exchanges)
                history_lines = []
                for msg in history[-4:]:
                    logger.debug(f"[MEMORY NODE] Processing message type: {type(msg).__name__}")
                    if isinstance(msg, HumanMessage):
                        history_lines.append(f"Spurning: {msg.content}")
                        logger.debug(f"[MEMORY NODE] Added human message: {msg.content[:50]}...")
                    elif isinstance(msg, AIMessage):
                        history_lines.append(f"Svar: {msg.content}")
                        logger.debug(f"[MEMORY NODE] Added AI message: {msg.content[:50]}...")
                    else:
                        # Fallback for unexpected message types
                        logger.warning(f"[MEMORY NODE] Unexpected message type: {type(msg)}, treating as AI message")
                        history_lines.append(f"Svar: {msg.content}")
                
                chat_history = "\n".join(history_lines)
                logger.info(f"[MEMORY NODE] Formatted chat_history length: {len(chat_history)}")
            else:
                logger.info(f"[MEMORY NODE] No messages found in memory for session {session_id}")
        
        except Exception as e:
            logger.error(f"[MEMORY NODE] Error loading memory: {e}", exc_info=True)
        
        # Preserve existing drug_entities through memory node
        # Query analysis node will merge/update them based on current question
        return {
            "chat_history": chat_history,
            "drug_entities": existing_drug_entities
        }
    
    # Helper: Determine if brand name is generic (no format/strength)
    def _is_generic_brand_name(self, brand_name: str) -> bool:
        """
        Determine if brand name is generic (no format/strength).
        
        Checks for indicators like:
        - No underscores with numbers (e.g., "_200mg", "_50ml")
        - No format descriptors (e.g., "_filmuhúðaðar_töflur")
        - Matches base brand name pattern
        
        Args:
            brand_name: Brand name to check
            
        Returns:
            True if brand name appears generic (no format/strength), False otherwise
        """
        if not brand_name:
            return False
        
        brand_norm = normalize_icelandic(brand_name)
        brand_norm_noacc = strip_diacritics(brand_norm)
        
        # Check for format/strength indicators
        # Pattern: numbers followed by units (mg, ml, g, etc.)
        import re
        strength_pattern = r'_\d+\s*(mg|ml|g|µg|mcg|iu|me|%|mg/ml|mg/g)'
        if re.search(strength_pattern, brand_norm_noacc):
            return False
        
        # Check for format descriptors (common Icelandic pharmaceutical terms)
        format_indicators = [
            '_filmuhúðað', '_filmuhúðaðar', '_töflur', '_tafla',
            '_kapsúlur', '_kapsúla', '_mjúkt', '_hylki',
            '_innrennslislyf', '_innrennslis', '_lausn', '_spræja',
            '_drep', '_drepi', '_mixtúra', '_sýrup', '_sýrupi',
            '_púður', '_púðri', '_gel', '_krem', '_salvi',
            '_smpc', '_smPC', '_SmPC', '_SMPC'  # Document type suffixes
        ]
        
        for indicator in format_indicators:
            if indicator in brand_norm_noacc:
                return False
        
        # Check for distributor names (indicates specific formulation)
        # Common distributors: Alvogen, Lyfjaver, Abacus, Mylan, STADA, Krka, Teva, Heilsa
        distributor_pattern = r'_(alvogen|lyfjaver|abacus|mylan|stada|krka|teva|heilsa|bluefish)'
        if re.search(distributor_pattern, brand_norm_noacc):
            return False
        
        # If none of the above patterns match, likely generic
        return True
    
    # Helper: Normalize brand name by removing format/strength suffixes
    def _normalize_brand_name(self, brand_name: str) -> str:
        """
        Normalize brand name by removing format/strength suffixes.
        
        Returns base brand name for matching.
        
        Args:
            brand_name: Brand name to normalize
            
        Returns:
            Normalized brand name (base name without format/strength)
        """
        if not brand_name:
            return brand_name
        
        normalized = normalize_icelandic(brand_name)
        
        # Remove document type suffixes
        for suffix in ["_SmPC", "_Smpc", "_smPC", "_SMPC", "_SmPC_SmPC"]:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)]
        
        # Remove strength patterns (e.g., "_200mg", "_50ml")
        import re
        normalized = re.sub(r'_\d+\s*(mg|ml|g|µg|mcg|iu|me|%|mg/ml|mg/g).*$', '', normalized, flags=re.IGNORECASE)
        
        # Remove format descriptors
        format_patterns = [
            r'_filmuhúðað.*$', r'_töflur.*$', r'_tafla.*$',
            r'_kapsúlur.*$', r'_kapsúla.*$', r'_mjúkt.*$', r'_hylki.*$',
            r'_innrennslislyf.*$', r'_innrennslis.*$', r'_lausn.*$', r'_spræja.*$',
            r'_drep.*$', r'_drepi.*$', r'_mixtúra.*$', r'_sýrup.*$', r'_sýrupi.*$',
            r'_púður.*$', r'_púðri.*$', r'_gel.*$', r'_krem.*$', r'_salvi.*$'
        ]
        
        for pattern in format_patterns:
            normalized = re.sub(pattern, '', normalized, flags=re.IGNORECASE)
        
        # Remove distributor suffixes
        normalized = re.sub(r'_(alvogen|lyfjaver|abacus|mylan|stada|krka|teva|heilsa|bluefish).*$', 
                          '', normalized, flags=re.IGNORECASE)
        
        return normalized.strip()
    
    def expand_to_all_related_drugs(self, brands: List[str]) -> List[DrugEntity]:
        """
        Expand detected brands to ALL drugs containing the same active ingredient(s).
        Returns a single consolidated DrugEntity with ingredients and brand_variants.
        
        This creates one conceptual drug entity representing all brand variants
        sharing the same active ingredients, rather than separate entities per brand.
        """
        if not brands:
            return []
        
        # Collect all ingredients from all brands
        all_ingredients = set()
        all_related_drugs = set()
        
        for brand in brands:
            ings = self.ingredients_manager.get_ingredients_for_drug(brand) or []
            all_ingredients.update(ings)
            
            # Find all drugs with these ingredients
            for ing in ings:
                related_drugs = self.ingredients_manager.get_drugs_by_ingredient(ing)
                all_related_drugs.update(related_drugs)
        
        # If we found ingredients, create a single consolidated entity
        if all_ingredients:
            # Use the first brand as the base name (or None if we want pure ingredient-based)
            base_brand = brands[0] if brands else None
            
            # Create single consolidated DrugEntity
            consolidated_entity: DrugEntity = {
                "medication_name": None,  # None indicates this is ingredient-based
                "active_ingredients": sorted(list(all_ingredients)),  # Sort for consistency
                "is_generic": True,  # True since it represents a conceptual drug
                "brand_variants": sorted(list(all_related_drugs))  # All related brand names
            }
            
            logger.info(f"Consolidated {len(brands)} brands into single entity with {len(all_ingredients)} ingredients and {len(all_related_drugs)} brand variants")
            return [consolidated_entity]
        
        # If no ingredients found, treat as failed detection and return empty list
        # This prevents the system from locking onto unmapped brands with no ingredient context
        logger.warning(f"No ingredients found for brands {brands}, ignoring them for now")
        return []
    
    # Define query analysis node
    def _create_query_analysis_node(self, state: DocumentRAGState) -> Dict[str, Any]:
        """
        Extract medication name(s) and active ingredient(s) from query.
        
        Creates DrugEntity objects for each detected drug, distinguishing between:
        - Generic brand names (e.g., "íbúfen") - no format/strength
        - Specific brand names (e.g., "Íbúfen_200mg_SmPC") - with format/strength
        - Active ingredients only (e.g., "Ibuprofen") - no brand name
        
        If no medication is found in the current question, checks chat history
        to maintain context for follow-up questions.
        
        Supports multiple drugs, each tracked individually with their own
        medication_name and active_ingredients.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with drug_entities and legacy fields (for backward compatibility)
        """
        question = state["question"]
        medication_filter = state.get("medication_filter") or self.medication_filter
        chat_history = state.get("chat_history", "")
        existing_drug_entities = state.get("drug_entities", [])
        
        logger.info(f"Query analysis - question: '{question}', medication_filter: {medication_filter}, chat_history length: {len(chat_history)}")
        
        from src.query_disambiguation import find_matching_medications, detect_active_ingredients_in_query
        
        # Detect active ingredients directly from query
        detected_ingredients = detect_active_ingredients_in_query(question, self.ingredients_manager)
        logger.info(f"Detected active ingredients in query: {detected_ingredients}")
        
        # Detect brand names from query
        brand_matches = []
        if not medication_filter:
            # Try to extract from current question
            brand_matches = find_matching_medications(question, self.vector_store_manager, use_ingredients=False)
            if brand_matches:
                # Expand brands into full ingredient-based context
                expanded = self.expand_to_all_related_drugs(brand_matches)
                logger.info(f"Expanded to {len(expanded)} related drugs via ingredients")
                # store into memory for follow-ups
                state["drug_entities"] = expanded
                return {"drug_entities": expanded}
            else:
                logger.info(f"No brand names found in current question: '{question}'")
            
            # If no matches in current question, check chat history
            if not brand_matches and chat_history:
                logger.info(f"No brand names found in current question, checking chat history")
                history_lines = chat_history.split("\n")
                previous_questions = []
                for line in history_lines:
                    if line.startswith("Spurning:"):
                        previous_question = line.replace("Spurning:", "").strip()
                        if previous_question:
                            previous_questions.append(previous_question)
                
                # Search previous questions for medications (most recent first)
                for prev_q in reversed(previous_questions):
                    brand_matches = find_matching_medications(prev_q, self.vector_store_manager, use_ingredients=False)
                    if brand_matches:
                        logger.info(f"Found brand name(s) in previous question: {brand_matches}")
                        break
        
        # If medication_filter provided, parse comma-separated values and add to brand_matches
        if medication_filter:
            # Split by comma and strip whitespace to support multiple drug names
            # e.g., "Tegretol, Panodil" -> ["Tegretol", "Panodil"]
            drug_names = [name.strip() for name in medication_filter.split(",") if name.strip()]
            for drug_name in drug_names:
                if drug_name not in brand_matches:
                    brand_matches.append(drug_name)
            logger.info(f"Parsed {len(drug_names)} drug name(s) from medication_filter: {drug_names}")
        
        # Create DrugEntity objects for each detected drug/ingredient
        new_drug_entities: List[DrugEntity] = []
        
        # Process each brand name individually
        for brand_name in brand_matches:
            # Determine if this is a generic brand name (no format/strength)
            is_generic = self._is_generic_brand_name(brand_name)
            
            # Get active ingredients for this brand
            ingredients = self.ingredients_manager.get_ingredients_for_drug(brand_name)
            
            # Create DrugEntity
            drug_entity: DrugEntity = {
                "medication_name": brand_name,
                "active_ingredients": ingredients if ingredients else [],
                "is_generic": is_generic
            }
            new_drug_entities.append(drug_entity)
            logger.info(f"Created DrugEntity: medication_name={brand_name}, is_generic={is_generic}, ingredients={ingredients}")
        
        # Process active ingredients that weren't already associated with a brand
        for ingredient in detected_ingredients:
            # Check if this ingredient is already covered by a brand we found
            ingredient_covered = False
            for entity in new_drug_entities:
                if ingredient in entity.get("active_ingredients", []):
                    ingredient_covered = True
                    break
            
            if not ingredient_covered:
                # Create DrugEntity for ingredient-only reference
                new_drug_entities.append({
                    "medication_name": None,
                    "active_ingredients": [ingredient],
                    "is_generic": True
                })
                logger.info(f"Created DrugEntity for ingredient-only: ingredient={ingredient}")
        
        # Merge with existing drug entities from previous questions
        final_drug_entities: List[DrugEntity] = []
        if existing_drug_entities and new_drug_entities:
            logger.info(f"Merging {len(existing_drug_entities)} existing and {len(new_drug_entities)} new drug entities")
            seen_keys = set()
            for entity in existing_drug_entities + new_drug_entities:
                key = (
                    entity.get("medication_name"),
                    tuple(entity.get("active_ingredients", []))
                )
                if key not in seen_keys:
                    seen_keys.add(key)
                    final_drug_entities.append(entity)
        elif new_drug_entities:
            logger.info(f"Using {len(new_drug_entities)} newly detected drug entities (no previous context)")
            final_drug_entities = new_drug_entities
        elif existing_drug_entities:
            logger.info(f"Using {len(existing_drug_entities)} stored drug entities from previous context (no new detection)")
            final_drug_entities = existing_drug_entities
        else:
            logger.info("No drug entities detected in query or history")
            final_drug_entities = []
        
        # Build legacy fields for backward compatibility
        # Extract medication names for legacy fields
        medication_names = [e["medication_name"] for e in final_drug_entities if e.get("medication_name")]
        
        # Build return state
        result: Dict[str, Any] = {
            "drug_entities": final_drug_entities
        }
        
        # Legacy fields for backward compatibility
        if len(medication_names) == 1:
            result["medication_filter"] = medication_names[0]
            result["extracted_medication"] = medication_names[0]
            result["extracted_medications"] = []
        elif len(medication_names) >= 2:
            result["medication_filter"] = None
            result["extracted_medication"] = None
            result["extracted_medications"] = medication_names
        else:
            # No medication names, but might have ingredient-only entities
            result["medication_filter"] = medication_filter
            result["extracted_medication"] = None
            result["extracted_medications"] = []
        
        return result
    
    # Define query rewrite node
    def _create_rewrite_node(self, state: DocumentRAGState) -> Dict[str, Any]:
        """
        Rewrite user query to improve retrieval accuracy.
        
        Uses LLM to:
        - Correct typos and normalize medical terminology
        - Resolve "lay language" into medical terms
        - Extract structured metadata (ingredients, drugs, sections)
        - Maintain context from chat history for follow-up questions
        
        Rewrite detections are hints only - they augment but don't replace query_analysis detections.
        Uses existing expand_to_all_related_drugs() infrastructure for consistency.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with rewritten_query, rewrite_metadata, and merged drug_entities
        """
        # Early return if disabled or LLM unavailable
        if not Config.ENABLE_QUERY_REWRITE or self.rewrite_llm is None:
            if not Config.ENABLE_QUERY_REWRITE:
                logger.debug("Query rewrite disabled (ENABLE_QUERY_REWRITE=False), skipping rewrite")
            else:
                logger.debug("Query rewrite LLM unavailable, skipping rewrite")
            return {}
        
        try:
            question = state["question"]
            existing_drug_entities = state.get("drug_entities", [])
            chat_history = state.get("chat_history", "")
            
            logger.info(f"Rewriting query: '{question[:50]}...'")
            
            # Build chat history context
            chat_history_context = ""
            if chat_history:
                # Extract last Q/A pair from chat_history
                history_lines = chat_history.split("\n")
                last_question = None
                last_answer = None
                
                # Find last question and answer
                for i in range(len(history_lines) - 1, -1, -1):
                    if history_lines[i].startswith("Svar:") and last_answer is None:
                        last_answer = history_lines[i].replace("Svar:", "").strip()
                    elif history_lines[i].startswith("Spurning:") and last_question is None:
                        last_question = history_lines[i].replace("Spurning:", "").strip()
                        if last_answer is not None:
                            break
                
                if last_question and last_answer:
                    # Truncate to ~200 chars each
                    truncated_q = last_question[:200] + "..." if len(last_question) > 200 else last_question
                    truncated_a = last_answer[:200] + "..." if len(last_answer) > 200 else last_answer
                    
                    chat_history_context = f"""Fyrri spurning: {truncated_q}
Fyrri svar: {truncated_a}

Ef fyrri spurningar vísa til ákveðins lyfs eða efnis, haltu áfram að miða við sama lyf nema annað sé tekið sérstaklega fram.

"""
            
            # Create parser for structured output
            parser = PydanticOutputParser(pydantic_object=RewriteState)
            format_instructions = parser.get_format_instructions()
            
            # Build prompt
            prompt = QUERY_REWRITE_PROMPT.format(
                chat_history_context=chat_history_context,
                question=question,
                format_instructions=format_instructions
            )
            
            # Invoke rewrite LLM
            response = self.rewrite_llm.invoke(prompt)
            
            # Extract response content
            if hasattr(response, "content"):
                response_text = response.content
            else:
                response_text = str(response)
            
            # Parse structured output
            try:
                rewrite_result = parser.parse(response_text)
            except Exception as parse_error:
                logger.warning(f"Failed to parse rewrite output: {parse_error}. Response: {response_text[:200]}")
                return {}
            
            logger.info(f"Rewrite successful. Rewritten query: '{rewrite_result.rewritten_query[:50]}...'")
            logger.info(f"Detected ingredients: {rewrite_result.detected_ingredients}")
            logger.info(f"Detected drugs: {rewrite_result.detected_drugs}")
            logger.info(f"Suggested sections: {rewrite_result.suggested_sections}")
            
            # Merge rewrite detections with existing drug_entities
            merged_drug_entities = list(existing_drug_entities)  # Start with existing
            
            # Process detected_ingredients
            for ingredient in rewrite_result.detected_ingredients:
                # Check if ingredient is already covered by existing entities
                ingredient_covered = False
                for entity in existing_drug_entities:
                    if ingredient in entity.get("active_ingredients", []):
                        ingredient_covered = True
                        break
                
                if not ingredient_covered:
                    # Create new DrugEntity for ingredient-only
                    new_entity: DrugEntity = {
                        "medication_name": None,
                        "active_ingredients": [ingredient],
                        "is_generic": True
                    }
                    merged_drug_entities.append(new_entity)
                    logger.info(f"Added ingredient-only entity from rewrite: {ingredient}")
            
            # Process detected_drugs
            if rewrite_result.detected_drugs:
                # Use existing expand_to_all_related_drugs() infrastructure
                expanded_entities = self.expand_to_all_related_drugs(rewrite_result.detected_drugs)
                
                # Deduplicate with existing entities
                for new_entity in expanded_entities:
                    # Check if this entity duplicates an existing one
                    # Duplicate rule: same set of active_ingredients and same medication_name (or both None)
                    new_ingredients_set = set(new_entity.get("active_ingredients", []))
                    new_med_name = new_entity.get("medication_name")
                    
                    is_duplicate = False
                    for existing_entity in merged_drug_entities:
                        existing_ingredients_set = set(existing_entity.get("active_ingredients", []))
                        existing_med_name = existing_entity.get("medication_name")
                        
                        # Check if duplicate: same ingredients set and same medication_name (or both None)
                        if (new_ingredients_set == existing_ingredients_set and 
                            new_med_name == existing_med_name):
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        merged_drug_entities.append(new_entity)
                        logger.info(f"Added drug entity from rewrite: {new_entity}")
            
            # Build rewrite metadata
            rewrite_metadata = {
                "detected_ingredients": rewrite_result.detected_ingredients,
                "detected_drugs": rewrite_result.detected_drugs,
                "suggested_sections": rewrite_result.suggested_sections
            }
            
            return {
                "rewritten_query": rewrite_result.rewritten_query,
                "rewrite_metadata": rewrite_metadata,
                "drug_entities": merged_drug_entities
            }
            
        except Exception as e:
            logger.warning(f"Error in rewrite node: {e}, using original query", exc_info=True)
            # Return empty state on error - original query will be used
            return {}
    
    def _boost_suggested_sections(
        self, 
        docs: List[Document], 
        suggested_sections: List[str]
    ) -> List[Document]:
        """
        Boost documents from suggested sections to the front of the list.
        
        This ensures that documents from sections identified as relevant by the
        query rewriter are prioritized during the top-K selection, while still
        allowing documents from other sections to be included.
        
        Args:
            docs: List of Document objects
            suggested_sections: List of section numbers to boost (e.g., ['4.8', '4.2'])
            
        Returns:
            Reordered list with boosted sections first, preserving relative order within groups
        """
        if not suggested_sections or not docs:
            return docs
        
        # Normalize suggested sections for matching (handle variations like "4.8" vs "4.8.")
        normalized_suggested = set()
        for section in suggested_sections:
            # Add both with and without trailing dot
            normalized_suggested.add(section.strip().rstrip('.'))
            normalized_suggested.add(section.strip())
        
        boosted = []
        non_boosted = []
        
        for doc in docs:
            section_num = doc.metadata.get("section_number", "")
            # Normalize the document's section number
            normalized_section = str(section_num).strip().rstrip('.')
            
            if normalized_section in normalized_suggested:
                boosted.append(doc)
            else:
                non_boosted.append(doc)
        
        if boosted:
            logger.info(f"Section boosting: {len(boosted)} docs from suggested sections {suggested_sections} moved to front")
        
        # Return boosted docs first, then non-boosted (preserving original relevance order within each group)
        return boosted + non_boosted
    
    def _apply_global_top_k(self, docs: List[Document], top_k: int) -> List[Document]:
        """
        Apply global top-K limit to documents with deduplication.
        
        Deduplicates documents by creating a unique key based on drug_id, section_number,
        and content hash. Preserves retriever order (relevance order) and takes top-K.
        
        Args:
            docs: List of Document objects
            top_k: Maximum number of documents to return
            
        Returns:
            Deduplicated and limited list of documents
        """
        if len(docs) <= top_k:
            return docs
        
        # Deduplicate by creating unique keys
        seen_keys = set()
        deduplicated = []
        
        for doc in docs:
            metadata = doc.metadata
            drug_id = metadata.get("drug_id", metadata.get("medication_name", "Unknown"))
            section_num = metadata.get("section_number", "Unknown")
            # Use first 100 chars of content as part of key to catch similar but not identical chunks
            content_preview = doc.page_content[:100] if doc.page_content else ""
            
            # Create unique key
            unique_key = f"{drug_id}_{section_num}_{hash(content_preview)}"
            
            if unique_key not in seen_keys:
                seen_keys.add(unique_key)
                deduplicated.append(doc)
                
                # Stop once we have enough
                if len(deduplicated) >= top_k:
                    break
        
        logger.info(f"Applied global top-K: {len(docs)} -> {len(deduplicated)} documents")
        return deduplicated
    
    # Define retrieval node
    def _create_retrieval_node(self, state: DocumentRAGState) -> Dict[str, Any]:
        """
        Retrieve relevant documents based on the user's question and drug entities.
        
        Processes each drug entity individually:
        - If medication_name is set and is_generic=False → filter by that specific brand
        - If medication_name is set and is_generic=True → expand to all brands with same active ingredients
        - If only active_ingredients → expand to all brands with those ingredients
        
        Uses parallel retrieval for multiple drug entities.
        Enhances query with context from chat history for better retrieval relevance.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with retrieved_docs, formatted_context, and retrieval_sufficient flag
        """
        try:
            question = state["question"]
            chat_history = state.get("chat_history", "")
            drug_entities = state.get("drug_entities", [])
            logger.info(f"Retrieval using {len(drug_entities)} drug entities from memory or detection")
            
            # Extract suggested sections from rewrite metadata for boosting
            rewrite_metadata = state.get("rewrite_metadata", {}) or {}
            suggested_sections = rewrite_metadata.get("suggested_sections", [])
            if suggested_sections:
                logger.info(f"Will boost documents from suggested sections: {suggested_sections}")
            
            # Fallback to legacy fields if drug_entities is empty
            medication_filter = state.get("medication_filter") or self.medication_filter
            extracted_medications = state.get("extracted_medications", [])
            
            # Determine which query to use for retrieval
            # Use rewritten query if available, otherwise use original
            rewritten_query = state.get("rewritten_query")
            retrieval_query = rewritten_query if rewritten_query else question
            
            # Always enhance with previous question from chat history when available
            if chat_history:
                # Extract the last question from history (format: "Spurning: ...")
                history_lines = chat_history.split("\n")
                previous_question = None
                for line in reversed(history_lines):
                    if line.startswith("Spurning:"):
                        previous_question = line.replace("Spurning:", "").strip()
                        break
                
                # Enhance query with previous question to maintain context for follow-ups
                if previous_question and previous_question != question:
                    retrieval_query = f"{question} {previous_question}"
                    logger.debug(f"Enhanced query with previous question for better context")
            
            logger.info(f"Retrieving documents for question: {question[:50]}...")
            
            # Process drug entities if available, otherwise fall back to legacy logic
            if drug_entities:
                logger.info(f"Processing {len(drug_entities)} drug entity/entities")
                all_docs = []
                
                def retrieve_for_drug_entity(entity: DrugEntity) -> List[Document]:
                    """Retrieve documents for a single drug entity."""
                    medication_name = entity.get("medication_name")
                    active_ingredients = entity.get("active_ingredients", [])
                    is_generic = entity.get("is_generic", False)
                    brand_variants = entity.get("brand_variants")
                    
                    logger.info(f"Retrieving for drug entity: medication_name={medication_name}, is_generic={is_generic}, ingredients={active_ingredients}, brand_variants={len(brand_variants) if brand_variants else 0}")
                    
                    # Case 1: Consolidated entity with brand_variants (from expand_to_all_related_drugs)
                    # Use ingredient-based retrieval to cover all variants
                    if brand_variants and active_ingredients:
                        logger.info(f"Using ingredient-based retrieval for consolidated entity with {len(brand_variants)} brand variants and ingredients: {active_ingredients}")
                        retriever = self.vector_store_manager.get_retriever_by_ingredients(
                            ingredient_names=active_ingredients,
                            top_k=Config.RETRIEVAL_MULTI_MED_K
                        )
                    
                    # Case 2: Specific brand name (is_generic=False)
                    elif medication_name and not is_generic:
                        logger.info(f"Using specific brand filter: {medication_name}")
                        retriever = self.vector_store_manager.get_retriever_with_filter(
                            medication_name=medication_name,
                            top_k=Config.RETRIEVAL_MULTI_MED_K
                        )
                    
                    # Case 3: Generic brand name (is_generic=True) or ingredient-only
                    elif medication_name and is_generic:
                        # Expand to all brands with same active ingredients
                        if active_ingredients:
                            logger.info(f"Expanding generic brand {medication_name} to all brands with ingredients: {active_ingredients}")
                            retriever = self.vector_store_manager.get_retriever_by_ingredients(
                                ingredient_names=active_ingredients,
                                top_k=Config.RETRIEVAL_MULTI_MED_K
                            )
                        else:
                            # No ingredients found, try to get them
                            ingredients = self.ingredients_manager.get_ingredients_for_drug(medication_name)
                            if ingredients:
                                logger.info(f"Found ingredients for {medication_name}: {ingredients}")
                                retriever = self.vector_store_manager.get_retriever_by_ingredients(
                                    ingredient_names=ingredients,
                                    top_k=Config.RETRIEVAL_MULTI_MED_K
                                )
                            else:
                                # Fallback to brand name filter
                                logger.warning(f"No ingredients found for {medication_name}, using brand filter")
                                retriever = self.vector_store_manager.get_retriever_with_filter(
                                    medication_name=medication_name,
                                    top_k=Config.RETRIEVAL_MULTI_MED_K
                                )
                    
                    # Case 4: Ingredient-only (no medication_name)
                    elif active_ingredients:
                        logger.info(f"Using ingredient-only filter: {active_ingredients}")
                        retriever = self.vector_store_manager.get_retriever_by_ingredients(
                            ingredient_names=active_ingredients,
                            top_k=Config.RETRIEVAL_MULTI_MED_K
                        )
                    
                    # Case 5: No medication_name or ingredients (shouldn't happen, but handle gracefully)
                    else:
                        logger.warning(f"Drug entity has no medication_name or active_ingredients, skipping")
                        return []
                    
                    try:
                        # Use enhanced query (with chat history context) for better relevance
                        docs = retriever.invoke(retrieval_query)
                    except AttributeError:
                        # Fallback for older LangChain versions
                        docs = retriever.get_relevant_documents(retrieval_query)
                    
                    logger.info(f"Retrieved {len(docs)} chunks for drug entity")
                    return docs
                
                # Parallelize retrieval across drug entities
                with ThreadPoolExecutor(max_workers=min(len(drug_entities), 5)) as executor:
                    future_to_entity = {
                        executor.submit(retrieve_for_drug_entity, entity): entity 
                        for entity in drug_entities
                    }
                    for future in as_completed(future_to_entity):
                        entity = future_to_entity[future]
                        try:
                            docs = future.result()
                            all_docs.extend(docs)
                        except Exception as e:
                            logger.error(f"Error retrieving for drug entity {entity}: {e}", exc_info=True)
                
                # Boost documents from suggested sections before applying top-K limit
                if suggested_sections:
                    all_docs = self._boost_suggested_sections(all_docs, suggested_sections)
                
                # Apply global top-K to limit total documents
                all_docs = self._apply_global_top_k(all_docs, Config.RETRIEVAL_TOP_K)
                
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
            
            # Legacy fallback: use old logic if no drug_entities
            logger.info("No drug_entities found, using legacy retrieval logic")
            
            # If we have a single medication_filter but also expanded medications, use the expanded list
            if medication_filter and extracted_medications and len(extracted_medications) > 1:
                logger.info(f"Using expanded medications from single filter {medication_filter}: {len(extracted_medications)} drugs")
                extracted_medications = extracted_medications
                medication_filter = None
            
            if medication_filter:
                logger.info(f"Using medication filter: {medication_filter}")
            elif extracted_medications:
                logger.info(f"Using extracted medications: {len(extracted_medications)} drugs")
            
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
                        docs = retriever.invoke(retrieval_query)
                    except AttributeError:
                        docs = retriever.get_relevant_documents(retrieval_query)
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
                
                # Boost documents from suggested sections before applying top-K limit
                if suggested_sections:
                    all_docs = self._boost_suggested_sections(all_docs, suggested_sections)
                
                # Apply global top-K to limit total documents
                all_docs = self._apply_global_top_k(all_docs, Config.RETRIEVAL_TOP_K)
                
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
                docs = retriever.invoke(retrieval_query)
            except AttributeError:
                docs = retriever.get_relevant_documents(retrieval_query)
            
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
    
    def _format_context(self, docs: List[Document], include_atc: bool = True, include_ingredients: bool = True) -> str:
        """
        Format documents into context string for LLM prompt.
        
        Args:
            docs: List of Document objects
            include_atc: If True, include ATC context for drugs
            include_ingredients: If True, include ingredients context for drugs
            
        Returns:
            Formatted context string
        """
        formatted_context_parts = []
        seen_drugs = set()  # Track drugs we've added metadata for
        
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
            
            # Add ATC and ingredients context for first occurrence of each drug
            if drug_id not in seen_drugs and drug_id != "Unknown":
                # Add ATC context
                if include_atc:
                    atc_context = self.atc_manager.format_atc_context_for_rag(
                        drug_id,
                        include_alternatives=False  # Don't include alternatives in main context
                    )
                    if atc_context:
                        formatted_context_parts.append(f"ATC upplýsingar fyrir {drug_id}:\n{atc_context}")
                
                # Add ingredients context
                if include_ingredients:
                    ingredients_context = self.ingredients_manager.format_ingredients_context_for_rag(
                        drug_id,
                        include_alternatives=False  # Don't include alternatives in main context
                    )
                    if ingredients_context:
                        formatted_context_parts.append(f"Innihaldsefni fyrir {drug_id}:\n{ingredients_context}")
                
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
            
            # Add ATC-based and ingredient-based alternatives if requested
            if asks_for_alternatives:
                extracted_medication = state.get("extracted_medication")
                if extracted_medication:
                    # ATC-based alternatives
                    atc_alternatives = self.atc_manager.get_alternatives(extracted_medication, max_results=5)
                    # Ingredient-based alternatives (generic alternatives)
                    ingredient_alternatives = self.ingredients_manager.get_generic_alternatives(
                        extracted_medication, max_results=5
                    )
                    
                    if atc_alternatives or ingredient_alternatives:
                        alt_context = f"\n\nValkostir:\n"
                        
                        if atc_alternatives:
                            alt_context += "Valkostir með sama ATC flokk:\n"
                            for alt_drug in atc_alternatives:
                                alt_atc = self.atc_manager.get_atc_codes_for_drug(alt_drug)
                                alt_context += f"- {alt_drug} (ATC: {', '.join(alt_atc) if alt_atc else 'Ekki þekkt'})\n"
                        
                        if ingredient_alternatives:
                            alt_context += "\nValkostir með sama virka innihaldsefni (generískir valkostir):\n"
                            for alt_drug in ingredient_alternatives:
                                alt_ingredients = self.ingredients_manager.get_ingredients_for_drug(alt_drug)
                                alt_context += f"- {alt_drug} (Innihaldsefni: {', '.join(alt_ingredients) if alt_ingredients else 'Ekki þekkt'})\n"
                        
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
                    "text": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
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
    

    def _create_similar_drugs_node(self, state: DocumentRAGState) -> Dict[str, Any]:
        """
        Extract similar drugs (brand variants) from drug entities.
        
        This node collects all brand_variants from drug_entities that were created
        by expand_to_all_related_drugs(), which finds all drugs sharing the same
        active ingredients as the queried drug.
        
        Args:
            state: Current graph state containing drug_entities
            
        Returns:
            Updated state with similar_drugs list
        """
        try:
            drug_entities = state.get("drug_entities", [])
            similar_drugs: List[str] = []
            
            # Extract brand_variants from all drug entities
            for entity in drug_entities:
                brand_variants = entity.get("brand_variants")
                if brand_variants:
                    # Add all brand variants to the list
                    similar_drugs.extend(brand_variants)
            
            # Remove duplicates while preserving order
            # Using dict.fromkeys() is a common Python trick to deduplicate while keeping order
            similar_drugs = list(dict.fromkeys(similar_drugs))
            
            logger.info(f"Found {len(similar_drugs)} similar drugs (brand variants)")
            
            return {
                "similar_drugs": similar_drugs
            }
        
        except Exception as e:
            logger.error(f"Error in similar_drugs node: {e}", exc_info=True)
            return {
                "similar_drugs": []
            }
    
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
                "drug_entities": [],  # Per-drug tracking with medication_name and active_ingredients
                "session_id": session_id,
                "retrieved_docs": [],
                "formatted_context": "",
                "chat_history": "",
                "answer": "",
                "sources": [],
                "retrieval_sufficient": True,
                "reranking_needed": None,
                "rewritten_query": None,
                "rewrite_metadata": None,
                "error": None,
                "similar_drugs": []  # Initialize similar_drugs list
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
            similar_drugs = result.get("similar_drugs", [])
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
                "similar_drugs": similar_drugs,
                "error": error
            }
        
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            return {
                "answer": f"Villa kom upp við að svara spurningu: {str(e)}",
                "sources": [],
                "similar_drugs": [],
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
