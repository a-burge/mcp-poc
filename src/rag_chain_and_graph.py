# ADD DEPRECATION NOTICE HERE - point to rag_chain_langgraph.py
"""
This file is deprecated. Please use rag_chain_langgraph.py instead.
"""

"""
Minimal, clean, course‑style RAG implementation using LangGraph.
This is the **baseline v2** rebuild — intentionally simple and easy to reason about.
Once this version is stable, performant, and reliable, we will extend it step‑by‑step.

Key design goals:
- Only two nodes: retrieval → generation
- No re‑ranking, no query‑analysis LLM, no parallelisation, no fallback logic
- Fast execution: only ONE LLM call per query
- Clean prompt structure in Icelandic
- Clean state definition, easy to extend later
- Supports optional conversation memory (last N exchanges)
"""

import logging
import uuid
from typing import Dict, Any, Optional, List, TypedDict, TYPE_CHECKING

from langchain_core.documents import Document
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage

from config import Config
from opik.integrations.langchain import OpikTracer
from src.drug_utils import detect_medications

if TYPE_CHECKING:
    from src.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# 1. Minimal State
# ---------------------------------------------------------
class SmPCRAGState(TypedDict):
    question: str
    context: str          # Final formatted context string
    answer: str           # Generated answer
    retrieved_docs: List[Document]
    chat_history: str     # Optional, if memory enabled
    session_id: Optional[str]


# ---------------------------------------------------------
# 2. Baseline Icelandic Prompt
# ---------------------------------------------------------
SYSTEM_INSTRUCTIONS = """Þú ert heilbrigðissérfræðingur sem svarar spurningum um SmPC lyfjaupplýsingar á íslensku.

Leiðbeiningar:
- Notaðu aðeins upplýsingarnar í 'Upplýsingar úr skjölum' hér að neðan.
- Ef svarið finnst ekki í textanum, segðu skýrt að upplýsingarnar séu ekki í tiltæku efni.
- Svaraðu af fagmennsku, í stuttu og skýru máli.
- Bættu ALDREI við upplýsingum sem ekki eru í upprunalegu SmPC textunum. Ekki gefa almenn svör. Segðu "Ég fann ekki upplýsingar um þetta í heimildum".
- Bættu við tilvísunum í lok svars samkvæmt sniðinu:
  [drug_id, kafli section_number: section_title]
"""

PROMPT_TEMPLATE = """{system}

{history_block}
Upplýsingar úr skjölum:
{context}

Spurning:
{question}

Svar:"""


# ---------------------------------------------------------
# 3. Minimal Clean RAG Graph
# ---------------------------------------------------------
class SmPCRAGGraph:
    """
    Minimal two‑node RAG graph:
        START → retrieval → generation → END
    """

    def __init__(
        self,
        vector_store_manager: "VectorStoreManager",
        memory_store: Optional[Dict[str, ConversationBufferMemory]] = None,
        model_name: str = "gpt-4o-mini",
    ):
        self.vector_store_manager = vector_store_manager
        self.memory_store = memory_store or {}
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.tracer: Optional[OpikTracer] = None

        # this will set both self.graph and self.tracer
        self._build_graph()

    # -----------------------------------------------------
    # Build LangGraph
    # -----------------------------------------------------
    def _build_graph(self):
        builder = StateGraph(SmPCRAGState)
        # CRITICAL FIX: Remove RunnableLambda wrappers - pass functions directly
        # This matches the working implementation and ensures proper state flow
        builder.add_node("memory", self._memory_node)
        builder.add_node("retrieval", self._retrieval_node)
        builder.add_node("generation", self._generation_node)

        builder.add_edge(START, "memory")
        builder.add_edge("memory", "retrieval")
        builder.add_edge("retrieval", "generation")
        builder.add_edge("generation", END)

        # 1 Compile the graph
        self.graph = builder.compile()

        # 2 Configure Opik tracer using the compiled graph
        self.tracer = self._configure_opik()

    # -----------------------------------------------------
    # Configure Opik Tracer
    # -----------------------------------------------------
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

    # -----------------------------------------------------
    # 4. Memory node (lightweight, optional)
    # -----------------------------------------------------
    def _memory_node(self, state: SmPCRAGState) -> Dict[str, Any]:
        """
        Load conversation history from memory if session_id is provided.
        
        Uses isinstance() checks to correctly identify HumanMessage and AIMessage
        objects from ConversationBufferMemory with return_messages=True.
        """
        session_id = state.get("session_id")
        history_str = ""

        logger.info(f"[MEMORY NODE] Processing state with session_id: {session_id}")
        logger.info(f"[MEMORY NODE] Memory store has {len(self.memory_store)} sessions: {list(self.memory_store.keys())}")

        if not session_id:
            logger.info("[MEMORY NODE] No session_id provided, skipping memory loading")
            return {"chat_history": history_str}

        if session_id not in self.memory_store:
            logger.info(f"[MEMORY NODE] Session {session_id} not found in memory_store")
            return {"chat_history": history_str}

        try:
            mem = self.memory_store[session_id]
            logger.info(f"[MEMORY NODE] Loading memory for session {session_id}")
            
            mem_vars = mem.load_memory_variables({})
            logger.info(f"[MEMORY NODE] Memory variables keys: {list(mem_vars.keys())}")
            
            messages = mem_vars.get("history", [])
            logger.info(f"[MEMORY NODE] Found {len(messages)} messages in memory")
            
            if not messages:
                logger.info(f"[MEMORY NODE] No messages found in memory for session {session_id}")
                return {"chat_history": history_str}
            
            # Convert last 4 messages into readable Icelandic Q/A lines
            history_lines = []
            for idx, m in enumerate(messages[-4:]):
                logger.debug(f"[MEMORY NODE] Processing message {idx}: type={type(m).__name__}")
                if isinstance(m, HumanMessage):
                    history_lines.append(f"Spurning: {m.content}")
                    logger.debug(f"[MEMORY NODE] Added human message: {m.content[:50]}...")
                elif isinstance(m, AIMessage):
                    history_lines.append(f"Svar: {m.content}")
                    logger.debug(f"[MEMORY NODE] Added AI message: {m.content[:50]}...")
                else:
                    # Fallback for unexpected message types (shouldn't happen with ConversationBufferMemory)
                    logger.warning(f"[MEMORY NODE] Unexpected message type: {type(m)}, treating as AI message")
                    history_lines.append(f"Svar: {m.content}")
            
            history_str = "\n".join(history_lines)
            logger.info(f"[MEMORY NODE] Formatted {len(history_lines)} messages into chat history (length: {len(history_str)} chars)")
            logger.debug(f"[MEMORY NODE] Chat history preview: {history_str[:200]}...")
            
        except Exception as e:
            logger.error(f"[MEMORY NODE] Error loading memory for session {session_id}: {e}", exc_info=True)
            # Return empty history on error to avoid breaking the flow
            history_str = ""

        logger.info(f"[MEMORY NODE] Returning chat_history with length: {len(history_str)}")
        return {"chat_history": history_str}

    # -----------------------------------------------------
    # 4.5. Helper: Expand drugs by active ingredients
    # -----------------------------------------------------
    def _expand_drugs_by_ingredients(self, detected_drugs: List[str]) -> List[str]:
        """
        Expand detected drugs to include all drugs with the same active ingredient(s).
        
        For example, if "Íbúfen" is detected, this will also include Nurofen, Alvofen,
        and other brands containing Ibuprofenum.
        
        Args:
            detected_drugs: List of detected drug IDs
            
        Returns:
            Expanded list of drug IDs including all drugs with same active ingredients
        """
        if not detected_drugs:
            return detected_drugs
        
        try:
            from src.ingredients_manager import IngredientsManager
            ingredients_manager = IngredientsManager()
            
            expanded_drugs = set(detected_drugs)  # Start with detected brands
            available_medications = self.vector_store_manager.get_unique_medications()
            
            # For each detected drug, find its active ingredient(s) and expand
            for drug_id in detected_drugs:
                # Get active ingredients for this drug
                ingredients = ingredients_manager.get_ingredients_for_drug(drug_id)
                
                if not ingredients:
                    # No ingredients found, keep original drug
                    continue
                
                # For each ingredient, find all drugs containing it
                for ingredient in ingredients:
                    drugs_with_ingredient = ingredients_manager.get_drugs_by_ingredient(ingredient)
                    
                    # Filter to only drugs that exist in vector store
                    for drug in drugs_with_ingredient:
                        # Normalize drug name for matching
                        drug_normalized = drug.lower().replace("_smpc", "").replace("_smPC", "").strip()
                        
                        # Try to match drug name to available medications
                        for available in available_medications:
                            available_normalized = available.lower().replace("_smpc", "").replace("_smPC", "").strip()
                            
                            # Match if normalized names are similar
                            if (drug_normalized == available_normalized or
                                drug_normalized in available_normalized or
                                available_normalized in drug_normalized):
                                expanded_drugs.add(available)
                                break
            
            if len(expanded_drugs) > len(detected_drugs):
                logger.info(f"Expanded medication filter from {len(detected_drugs)} to {len(expanded_drugs)} drugs based on active ingredients")
                logger.debug(f"Original: {detected_drugs}")
                logger.debug(f"Expanded: {sorted(expanded_drugs)}")
            
            return list(expanded_drugs)
            
        except Exception as e:
            logger.warning(f"Error expanding medications by ingredient: {e}", exc_info=True)
            # Fall back to original detected_drugs on error
            return detected_drugs
    
    # -----------------------------------------------------
    # 5. Retrieval node (with drug detection and filtering)
    # -----------------------------------------------------
    def _retrieval_node(self, state: SmPCRAGState) -> Dict[str, Any]:
        question = state["question"]
        chat_history = state.get("chat_history", "")

        # 1. Detect medications mentioned in the current question
        detected_drugs = detect_medications(question, self.vector_store_manager.all_drugs_list)
        
        # 2. If no medications found in current question, check chat history
        if not detected_drugs and chat_history:
            logger.info("No medications detected in current question, checking chat history")
            detected_drugs = detect_medications(chat_history, self.vector_store_manager.all_drugs_list)
            if detected_drugs:
                logger.info(f"Found medications in chat history: {detected_drugs}")
        
        # 2.5. Expand detected drugs to include all drugs with same active ingredient
        if detected_drugs:
            detected_drugs = self._expand_drugs_by_ingredients(detected_drugs)
        
        # 3. Enhance query with minimal context from previous conversation
        # This helps with follow-up questions like "en ef barnið er 14 ára?" after "er í lagi að gefa barni íbúfen?"
        # We extract just the previous question to provide context without overwhelming the query
        enhanced_question = question
        if chat_history:
            # Extract the last question from history (format: "Spurning: ...")
            history_lines = chat_history.split("\n")
            previous_question = None
            for line in reversed(history_lines):
                if line.startswith("Spurning:"):
                    previous_question = line.replace("Spurning:", "").strip()
                    break
            
            # If we found a previous question and it's different from current, add minimal context
            if previous_question and previous_question != question:
                # Add just the medication context if we detected drugs from history
                if detected_drugs:
                    # We already have medication filter, just add age/context keywords from previous question
                    # Extract key terms that might help (age, condition, etc.)
                    enhanced_question = f"{question} (tengt við: {previous_question})"
                    logger.debug(f"Enhanced query with previous question context")
                else:
                    # No medications found, add full previous question for better retrieval
                    enhanced_question = f"{question} {previous_question}"
                    logger.debug(f"Enhanced query with previous question (no medications found)")
        
        if detected_drugs:
            logger.info(f"Detected medications: {detected_drugs}")
            # 4. Filtered retrieval: only retrieve documents for detected drugs
            retriever = self.vector_store_manager.get_retriever_with_filter(
                drug_ids=detected_drugs,
                top_k=Config.RETRIEVAL_TOP_K
            )
        else:
            logger.info("No medications detected, using general retrieval")
            # 5. General fallback: retrieve from all drugs
            retriever = self.vector_store_manager.get_retriever(top_k=Config.RETRIEVAL_TOP_K)
        
        # Use enhanced question for retrieval to improve relevance
        retrieval_chain = RunnableLambda(lambda _: enhanced_question) | retriever

        try:
            docs = retrieval_chain.invoke({})
        except AttributeError:
            docs = retriever.get_relevant_documents(enhanced_question)

        context = self._format_context(docs)

        return {
            "retrieved_docs": docs,
            "context": context,
        }

    # -----------------------------------------------------
    # Helper: format retrieved documents into clean text
    # -----------------------------------------------------
    def _format_context(self, docs: List[Document]) -> str:
        parts = []
        for d in docs:
            md = d.metadata
            drug_id = md.get("drug_id", md.get("medication_name", "Unknown"))
            section_num = md.get("section_number", "Unknown")
            section_title = md.get("section_title", "Unknown")

            block = f"""
Heimild: 
- Lyf: {drug_id}
- Kafli: {section_num}: {section_title}
{d.page_content}
"""
            parts.append(block.strip())

        return "\n---\n".join(parts)

    # -----------------------------------------------------
    # 6. Generation node
    # -----------------------------------------------------
    def _generation_node(self, state: SmPCRAGState) -> Dict[str, Any]:
        question = state["question"]
        context = state["context"]
        chat_history = state.get("chat_history", "")

        logger.info(f"[GENERATION NODE] Received chat_history length: {len(chat_history)}")
        logger.info(f"[GENERATION NODE] Chat_history content: {repr(chat_history[:200]) if chat_history else 'EMPTY'}")
        logger.info(f"[GENERATION NODE] Full state keys: {list(state.keys())}")

        history_block = f"Samtal hingað til:\n{chat_history}\n\n" if chat_history else ""
        
        logger.info(f"[GENERATION NODE] History block length: {len(history_block)}")
        if history_block:
            logger.debug(f"[GENERATION NODE] History block preview: {history_block[:300]}...")

        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["system", "history_block", "context", "question"],
        )

        final_prompt = prompt.format(
            system=SYSTEM_INSTRUCTIONS,
            history_block=history_block,
            context=context,
            question=question,
        )
        
        logger.info(f"[GENERATION NODE] Final prompt length: {len(final_prompt)}")
        logger.debug(f"[GENERATION NODE] Final prompt preview (last 500 chars): ...{final_prompt[-500:]}")

        generation_chain = RunnableLambda(lambda _: final_prompt) | self.llm

        response = generation_chain.invoke({})
        answer = response.content if hasattr(response, "content") else str(response)

        return {"answer": answer}

    # -----------------------------------------------------
    # Public entrypoint
    # -----------------------------------------------------
    def process(
        self,
        question: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a question through the RAG graph with memory and Opik tracing.
        
        Uses thread_id for Opik session grouping and ensures memory always persists.
        If session_id is None, generates a UUID to ensure memory works automatically.
        """
        # Generate thread_id: use provided session_id or create UUID
        # This ensures memory always works and Opik traces are properly grouped
        thread_id = session_id if session_id else str(uuid.uuid4())
        
        initial_state: SmPCRAGState = {
            "question": question,
            "session_id": thread_id,  # Use thread_id as session_id in state
            "context": "",
            "retrieved_docs": [],
            "chat_history": "",
            "answer": "",
        }

        logger.info(f"[PROCESS] Starting process with thread_id: {thread_id}, question: {question[:100]}...")

        # Prepare callbacks
        callbacks = []
        if self.tracer:
            callbacks.append(self.tracer)
        
        # Use thread_id in config for Opik session grouping and LangGraph compatibility
        config = {
            "configurable": {"thread_id": thread_id},
            "callbacks": callbacks if callbacks else None
        }
        
        result = self.graph.invoke(initial_state, config=config)

        logger.info(f"[PROCESS] Graph execution completed. Result keys: {list(result.keys())}")
        logger.info(f"[PROCESS] Result chat_history length: {len(result.get('chat_history', ''))}")

        # Always persist memory using thread_id (removed conditional check)
        # This ensures memory works even when user doesn't provide session_id
        mem = self.memory_store.setdefault(
        thread_id, ConversationBufferMemory(return_messages=True, input_key="input", output_key="output")
        )
        logger.info(f"[PROCESS] Saving to memory - thread_id: {thread_id}")
        logger.info(f"[PROCESS] Question: {question[:100]}...")
        logger.info(f"[PROCESS] Answer length: {len(result.get('answer', ''))}")
        
        mem.save_context({"input": question}, {"output": result["answer"]})
        
        # Verify memory was saved
        mem_vars = mem.load_memory_variables({})
        saved_messages = mem_vars.get("history", [])
        logger.info(f"[PROCESS] Memory saved successfully. Total messages in memory: {len(saved_messages)}")
        if saved_messages:
            logger.debug(f"[PROCESS] Last message types: {[type(m).__name__ for m in saved_messages[-2:]]}")

        return {
            "answer": result["answer"],
            "sources": result.get("retrieved_docs", []),
        }
