"""Streamlit interface for MCP Server POC."""
import logging
import sys
from pathlib import Path
from typing import List, Optional

import streamlit as st

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from src.pdf_fetcher import fetch_and_extract_pdf
from src.chunker import chunk_document
from src.vector_store import VectorStoreManager
from src.rag_chain import create_qa_chain, query_rag
from src.query_disambiguation import should_disambiguate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="MCP Server POC - SmPC Document Q&A",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if "vector_store_manager" not in st.session_state:
    # Initialize vector store manager on startup to load existing data
    try:
        st.session_state.vector_store_manager = VectorStoreManager()
        # Check if there's existing data
        doc_count = st.session_state.vector_store_manager.get_document_count()
        if doc_count > 0:
            st.session_state.document_processed = True
            logger.info(f"Loaded existing vector store with {doc_count} chunks")
    except Exception as e:
        logger.warning(f"Could not initialize vector store on startup: {e}")
        st.session_state.vector_store_manager = None

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False
if "selected_medication" not in st.session_state:
    st.session_state.selected_medication = None
if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = Config.LLM_PROVIDER
if "last_llm_provider" not in st.session_state:
    st.session_state.last_llm_provider = None


def process_pdf(
    pdf_url: str,
    llm_provider: str,
    update_if_exists: bool = False
) -> tuple[bool, str]:
    """
    Process PDF: download, chunk, and index.
    
    Args:
        pdf_url: URL of PDF to process
        llm_provider: LLM provider to use
        update_if_exists: If True, update existing document; if False, skip
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Initialize vector store if not already done
        if st.session_state.vector_store_manager is None:
            st.session_state.vector_store_manager = VectorStoreManager()
        
        vector_store_manager = st.session_state.vector_store_manager
        
        with st.spinner("Sæki PDF skjal..."):
            # Fetch and extract PDF
            document = fetch_and_extract_pdf(pdf_url)
            
            # Check if document already exists
            if vector_store_manager.document_exists(document.filename):
                if not update_if_exists:
                    return False, f"Skjalið '{document.filename}' er þegar í gagnagrunni. Veldu 'Uppfæra' ef þú vilt skipta út."
                else:
                    # Remove existing document before re-adding
                    removed_count = vector_store_manager.remove_document(document.filename)
                    if removed_count > 0:
                        st.info(f"🗑️ Fjarlægði {removed_count} kafla fyrir '{document.filename}' (uppfærir skjal...)")
            
            st.success(f"PDF sótt: {document.filename} ({document.medication_name})")
        
        with st.spinner("Skipti skjali í kafla..."):
            # Chunk document
            chunks = chunk_document(
                document,
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP
            )
            st.success(f"Skjali skipt í {len(chunks)} kafla")
        
        with st.spinner("Bæti við vektor gagnagrunn..."):
            # Add chunks (don't clear collection - support multiple documents)
            vector_store_manager.add_chunks(chunks)
            st.success("Kaflar bætt við vektor gagnagrunn")
        
        # Invalidate QA chain cache since new data was added
        # Main logic will recreate it if needed
        st.session_state.qa_chain = None
        st.session_state.last_llm_provider = None
        
        st.session_state.document_processed = True
        return True, f"✅ Skjal '{document.filename}' unnið með!"
        
    except Exception as e:
        logger.error(f"Error processing PDF: {e}", exc_info=True)
        return False, f"Villa kom upp við vinnslu skjals: {str(e)}"


def process_batch_pdfs(pdf_urls: List[str], llm_provider: str) -> None:
    """Process multiple PDFs in batch."""
    if not pdf_urls:
        st.warning("Engar PDF slóðir gefnar")
        return
    
    # Initialize vector store
    if st.session_state.vector_store_manager is None:
        st.session_state.vector_store_manager = VectorStoreManager()
    
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, url in enumerate(pdf_urls):
        url = url.strip()
        if not url:
            continue
        
        status_text.text(f"Vinnur úr skjali {i+1}/{len(pdf_urls)}: {url[:50]}...")
        success, message = process_pdf(url, llm_provider, update_if_exists=True)
        results.append((url, success, message))
        progress_bar.progress((i + 1) / len(pdf_urls))
    
    status_text.empty()
    progress_bar.empty()
    
    # Show results
    st.subheader("Niðurstöður")
    for url, success, message in results:
        if success:
            st.success(f"✅ {url[:60]}... - {message}")
        else:
            st.error(f"❌ {url[:60]}... - {message}")
    
    # Invalidate QA chain cache since new data was added
    # Main logic will recreate it if needed
    st.session_state.qa_chain = None
    st.session_state.last_llm_provider = None
    st.session_state.document_processed = True


def main():
    """Main Streamlit application."""
    st.title("📄 MCP Server POC - SmPC Document Q&A")
    st.markdown("""
    Þetta er prófunarútgáfa af MCP server sem getur svarað spurningum um lyfjaupplýsingar 
    úr SmPC skjölum með notkun á RAG (Retrieval-Augmented Generation) tækni.
    
    **Eiginleikar:**
    - Sækir og vinnur úr SmPC PDF skjölum
    - Skiptir skjölum í kafla með varðveislu á samhengi
    - Svara spurningum á íslensku með tilvísunum í kafla
    - Stuðningur við bæði Google Gemini og OpenAI GPT-4.1
    - Stuðningur við mörg skjöl og lyfjafrágang
    """)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Stillingar")
        
        # LLM Provider selection
        llm_provider = st.selectbox(
            "LLM Veitandi",
            options=["gemini", "gpt5"],
            index=0 if st.session_state.llm_provider == "gemini" else 1,
            help="Veldu hvaða LLM veitanda á að nota",
            key="llm_provider_selectbox"
        )
        # Store in session state for consistency
        st.session_state.llm_provider = llm_provider
        
        # Tab selection for single vs batch
        tab1, tab2 = st.tabs(["Eitt skjal", "Fjöldi skjala"])
        
        with tab1:
            # Single PDF URL input
            pdf_url = st.text_input(
                "PDF URL",
                value=Config.PDF_URL,
                help="Slóð að SmPC PDF skjali",
                key="single_pdf_url"
            )
            
            # Process PDF button
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Vinna úr PDF", type="primary", key="process_single"):
                    # Validate API keys
                    if llm_provider == "gemini" and not Config.GOOGLE_API_KEY:
                        st.error("GOOGLE_API_KEY vantar!")
                        return
                    if llm_provider == "gpt5" and not Config.OPENAI_API_KEY:
                        st.error("OPENAI_API_KEY vantar!")
                        return
                    
                    success, message = process_pdf(pdf_url, llm_provider, update_if_exists=False)
                    if success:
                        st.success(message)
                    else:
                        st.warning(message)
                        if "þegar í gagnagrunni" in message:
                            if st.button("Uppfæra skjal", key="update_single"):
                                success, msg = process_pdf(pdf_url, llm_provider, update_if_exists=True)
                                if success:
                                    st.success(msg)
                                else:
                                    st.error(msg)
            
            with col2:
                if st.button("🗑️ Hreinsa gagnagrunn", key="clear_store"):
                    if st.session_state.vector_store_manager:
                        st.session_state.vector_store_manager.clear_collection()
                        st.session_state.qa_chain = None
                        st.session_state.document_processed = False
                        st.session_state.selected_medication = None
                        st.success("Gagnagrunnur hreinsaður")
        
        with tab2:
            # Batch PDF URLs input
            batch_urls = st.text_area(
                "PDF URLs (eitt á hverri línu)",
                help="Sláðu inn fleiri en eina PDF slóð, eitt á hverri línu",
                height=150,
                key="batch_urls"
            )
            
            if st.button("🔄 Vinna úr öllum", type="primary", key="process_batch"):
                # Validate API keys
                if llm_provider == "gemini" and not Config.GOOGLE_API_KEY:
                    st.error("GOOGLE_API_KEY vantar!")
                    return
                if llm_provider == "gpt5" and not Config.OPENAI_API_KEY:
                    st.error("OPENAI_API_KEY vantar!")
                    return
                
                urls = [url.strip() for url in batch_urls.split("\n") if url.strip()]
                if urls:
                    process_batch_pdfs(urls, llm_provider)
                else:
                    st.warning("Engar PDF slóðir gefnar")
        
        # Status
        st.divider()
        st.subheader("Staða")
        if st.session_state.vector_store_manager:
            doc_count = st.session_state.vector_store_manager.get_document_count()
            processed_docs = st.session_state.vector_store_manager.get_unique_documents()
            medications = st.session_state.vector_store_manager.get_unique_medications()
            
            if doc_count > 0:
                st.success(f"✅ {doc_count} kaflar í gagnagrunni")
                
                # Show processed PDFs
                if processed_docs:
                    st.info(f"📄 Unnin skjöl: {len(processed_docs)}")
                    with st.expander("Skoða unnin skjöl", expanded=True):
                        for doc in processed_docs:
                            st.caption(
                                f"📄 **{doc['filename']}**\n"
                                f"   💊 {doc['medication_name']} • {doc['chunk_count']} kaflar"
                            )
                
                if medications:
                    st.info(f"📊 Lyf í gagnagrunni: {len(medications)}")
                    for med in medications:
                        st.caption(f"  • {med}")
            else:
                st.info("⏳ Engin skjöl unnin enn")
        else:
            st.info("⏳ Engin skjöl unnin enn")
    
    # Main content area
    if not st.session_state.document_processed or not st.session_state.vector_store_manager:
        st.info("""
        👈 Byrjaðu á að vinna úr PDF skjali með því að:
        1. Slá inn PDF URL(s) í hliðarstiku
        2. Velja LLM veitanda
        3. Smella á "Vinna úr PDF" takkann
        """)
        return
    
    # Ensure QA chain is created if we have data but no chain
    # Also recreate if provider changed
    current_llm_provider = st.session_state.get("llm_provider", Config.LLM_PROVIDER)
    cached_provider = st.session_state.get("last_llm_provider")
    needs_qa_chain = (
        st.session_state.vector_store_manager and 
        (not st.session_state.qa_chain or cached_provider != current_llm_provider)
    )
    if needs_qa_chain:
        with st.spinner("Bý til RAG keðju..."):
            qa_chain = create_qa_chain(
                st.session_state.vector_store_manager,
                provider=current_llm_provider
            )
            st.session_state.qa_chain = qa_chain
            st.session_state.last_llm_provider = current_llm_provider
    
    vector_store_manager = st.session_state.vector_store_manager
    
    # Medication selector
    medications = vector_store_manager.get_unique_medications()
    if medications:
        st.header("💊 Veldu lyf")
        selected_medication = st.selectbox(
            "Lyf (valkvætt - til að sía spurningar)",
            options=[None] + medications,
            format_func=lambda x: "Allir lyf" if x is None else x,
            index=0 if st.session_state.selected_medication is None else (
                medications.index(st.session_state.selected_medication) + 1
                if st.session_state.selected_medication in medications else 0
            ),
            key="medication_selector"
        )
        st.session_state.selected_medication = selected_medication
        
        if selected_medication:
            st.info(f"🔍 Síað eftir: **{selected_medication}**")
    
    # Query interface
    st.header("💬 Spyrja um skjal")
    
    # Query input
    question = st.text_input(
        "Spurning (á íslensku)",
        placeholder="T.d. Hver er skammturinn fyrir þessa lyf?",
        help="Sláðu inn spurningu á íslensku um lyfjaupplýsingarnar",
        key="query_input"
    )
    
    # Check for disambiguation if no medication selected
    disambiguation_info = None
    if question and not st.session_state.selected_medication:
        disambiguation_info = should_disambiguate(question, vector_store_manager)
    
    if disambiguation_info and disambiguation_info["needs_disambiguation"]:
        st.warning(disambiguation_info["clarification_prompt"])
        if disambiguation_info["matching_medications"]:
            selected_from_disambiguation = st.selectbox(
                "Veldu lyf:",
                options=disambiguation_info["matching_medications"],
                key="disambiguation_selector"
            )
            st.session_state.selected_medication = selected_from_disambiguation
    
    if st.button("🔍 Leita", type="primary", key="search_button") and question:
        if not st.session_state.qa_chain:
            st.error("RAG keðja er ekki tilbúin. Vinsamlegast vinndu úr PDF skjali fyrst.")
            return
        
        # Use selected medication for filtering
        medication_filter = st.session_state.selected_medication
        
        # Use cached chain if no filter, otherwise create filtered chain
        # This is efficient since filtered chains are only created when needed
        current_provider = st.session_state.get("llm_provider", Config.LLM_PROVIDER)
        if medication_filter:
            # Create chain with medication filter (only when filter is active)
            qa_chain = create_qa_chain(
                vector_store_manager,
                provider=current_provider,
                medication_filter=medication_filter
            )
        else:
            # Use cached base chain (no filter)
            qa_chain = st.session_state.qa_chain
        
        with st.spinner("Leita að svari..."):
            result = query_rag(qa_chain, question, medication_filter=medication_filter)
        
        # Display answer
        st.subheader("Svar")
        st.write(result["answer"])
        
        # Display sources
        if result["sources"]:
            st.subheader("Heimildir")
            for i, source in enumerate(result["sources"], 1):
                medication_name = source.get("medication_name", "Unknown")
                with st.expander(f"📄 Heimild {i}: {source['section']} - {medication_name} (Síða {source['page']})"):
                    st.write(f"**Lyf:** {medication_name}")
                    st.write(f"**Kafli:** {source['section']}")
                    st.write(f"**Skjal:** {source['source']}")
                    st.write(f"**Síða:** {source['page']}")
                    st.write(f"**Texti:**")
                    st.write(source['text'])
        else:
            st.info("Engar heimildir fundust")
    
    # Example questions
    st.divider()
    st.subheader("💡 Dæmi um spurningar")
    example_questions = [
        "Hver er skammturinn fyrir þessa lyf?",
        "Hverjar eru andmæli við þessum lyfjum?",
        "Hverjar eru aukaverkanir?",
        "Hvaða lyfjaviðbrögð geta orðið?",
        "Hvað á að gera við ofskömmtun?",
    ]
    
    cols = st.columns(len(example_questions))
    for i, example in enumerate(example_questions):
        with cols[i]:
            if st.button(f"❓", key=f"example_{i}", help=example):
                st.session_state.query_input = example
    
    # Footer
    st.divider()
    st.markdown("""
    ---
    **MCP Server POC** - Prófunarútgáfa
    
    Þessi kerfi er byggt á RAG (Retrieval-Augmented Generation) tækni með LangChain,
    Chroma vektor gagnagrunni, og Google Gemini eða OpenAI GPT-5 Mini fyrir íslensku.
    """)


if __name__ == "__main__":
    # Validate configuration
    try:
        Config.validate()
    except ValueError as e:
        st.error(f"Stillingavilla: {e}")
        st.info("Vinsamlegast stilltu .env skrá með nauðsynlegum API lyklum.")
        st.stop()
    
    main()
