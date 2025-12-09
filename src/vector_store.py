"""Vector store management with Chroma and multilingual embeddings."""
import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

# Disable ChromaDB telemetry to avoid PostHog compatibility errors
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

# Monkey-patch PostHog BEFORE ChromaDB imports it
# ChromaDB calls PostHog's capture() with positional arguments, but newer
# PostHog versions expect different signatures. This patch prevents errors.
try:
    import posthog
    
    # Patch PostHog client class to make capture() a no-op
    if hasattr(posthog, 'Client'):
        original_client_init = posthog.Client.__init__
        
        def patched_client_init(self, *args, **kwargs):
            """Initialize PostHog client with patched capture method."""
            original_client_init(self, *args, **kwargs)
            # Replace capture method with no-op that accepts any arguments
            self.capture = lambda *args, **kwargs: None
        
        posthog.Client.__init__ = patched_client_init
    
    # Also patch module-level capture if it exists
    if hasattr(posthog, 'capture'):
        posthog.capture = lambda *args, **kwargs: None
        
except (ImportError, AttributeError):
    # PostHog not available, continue without patching
    pass

import chromadb
from chromadb.config import Settings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import Field, ConfigDict

from config import Config
from src.chunker import Chunk

logger = logging.getLogger(__name__)

# Patch ChromaDB's telemetry modules AFTER chromadb is imported
# This suppresses any remaining telemetry errors
try:
    import chromadb.telemetry.events as chroma_telemetry
    
    # Patch TelemetryEvent.capture() method
    if hasattr(chroma_telemetry, 'TelemetryEvent'):
        original_capture = getattr(chroma_telemetry.TelemetryEvent, 'capture', None)
        if original_capture:
            def noop_capture(*args, **kwargs):
                """No-op function to suppress telemetry errors."""
                pass
            chroma_telemetry.TelemetryEvent.capture = staticmethod(noop_capture)
            
except (ImportError, AttributeError):
    # Telemetry module structure may vary, PostHog patch should handle it
    pass

try:
    import chromadb.telemetry.product_telemetry as product_telemetry
    if hasattr(product_telemetry, 'ProductTelemetryClient'):
        # Patch the client's capture method
        original_client_capture = getattr(product_telemetry.ProductTelemetryClient, 'capture', None)
        if original_client_capture:
            def noop_client_capture(self, *args, **kwargs):
                """No-op function to suppress telemetry client errors."""
                pass
            product_telemetry.ProductTelemetryClient.capture = noop_client_capture
except (ImportError, AttributeError):
    # Telemetry client may not exist in all ChromaDB versions
    pass


class InstrumentedRetriever(BaseRetriever):
    """
    Wrapper retriever for LangChain 0.3.x compatibility.
    
    Note: Opik tracing is handled at the graph level via OpikTracer callback,
    not through manual logging in this class.
    """
    
    # Pydantic v2 field declarations
    base_retriever: BaseRetriever = Field(description="The base retriever to wrap")
    medication_filter: Optional[str] = Field(default=None, description="Optional medication name filter for logging")
    
    # Allow arbitrary types since BaseRetriever is not JSON serializable
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def __init__(self, base_retriever: BaseRetriever, medication_filter: Optional[str] = None):
        """
        Initialize instrumented retriever.
        
        Args:
            base_retriever: The base retriever to wrap
            medication_filter: Optional medication name filter for logging
        """
        super().__init__(base_retriever=base_retriever, medication_filter=medication_filter)
    
    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        """
        Retrieve relevant documents.
        
        Args:
            query: Query string
            run_manager: Optional run manager
            
        Returns:
            List of relevant documents
        """
        logger.info(f"Retrieving documents for query: {query[:50]}...")
        if self.medication_filter:
            logger.info(f"Medication filter: {self.medication_filter}")
        
        # Perform retrieval using the base retriever
        # Use invoke() for LangChain 0.3.x compatibility
        # Opik tracing is handled automatically by OpikTracer callback
        try:
            docs = self.base_retriever.invoke(query)
        except AttributeError:
            # Fallback for older LangChain versions
            docs = self.base_retriever.get_relevant_documents(query)
        
        return docs
    
    async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        """
        Async retrieve relevant documents.
        
        Args:
            query: Query string
            run_manager: Optional run manager
            
        Returns:
            List of relevant documents
        """
        # Use base retriever's async method if available
        try:
            return await self.base_retriever.ainvoke(query)
        except AttributeError:
            # Fallback to sync method
            return self._get_relevant_documents(query, run_manager=run_manager)


class VectorStoreManager:
    """Manages vector store operations with Chroma."""
    
    def __init__(self, collection_name: str = "smpc_documents"):
        """
        Initialize vector store manager.
        
        Args:
            collection_name: Name of the Chroma collection
        """
        self.collection_name = collection_name
        self.embeddings = self._create_embeddings()
        self.vector_store: Optional[Chroma] = None
        self.all_drugs: List[str] = []
        self._unique_medications: List[str] = []  # Cached unique medications list
        self._initialize_store()
        # Load all drug IDs after store initialization
        self.all_drugs = self.get_all_drug_ids()
        # Cache unique medications - use all_drugs as they contain the same medication names
        # This avoids expensive database query on every call to get_unique_medications()
        self._unique_medications = sorted(list(set(self.all_drugs)))  # Deduplicate and sort
        logger.info(f"Loaded {len(self.all_drugs)} unique drug IDs")
        logger.info(f"Cached {len(self._unique_medications)} unique medications")
    
    def _create_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Create multilingual embeddings model.
        
        Returns:
            HuggingFaceEmbeddings instance
        """
        logger.info(f"Loading embedding model: {Config.EMBEDDING_MODEL}")
        embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        return embeddings
    
    def _initialize_store(self) -> None:
        """Initialize or load existing Chroma vector store."""
        Config.ensure_directories()
        
        persist_directory = str(Config.VECTOR_STORE_PATH)
        
        logger.info(f"Initializing vector store at {persist_directory}")
        
        # Create ChromaDB client with telemetry explicitly disabled
        client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection with HNSW parameters optimized for large collections
        # These parameters MUST be set at collection creation time - they cannot be changed later
        # For 62k+ documents, we need higher M and ef_construction values
        # NOTE: In ChromaDB 0.4.22, HNSW parameters may not be fully supported via metadata
        # We'll try to create with parameters, but fall back to default if needed
        collection = None
        try:
            # Try to get existing collection first
            collection = client.get_collection(name=self.collection_name)
            logger.info(f"Using existing collection '{self.collection_name}'")
        except Exception:
            # Collection doesn't exist, create it
            # Try with HNSW parameters first (as integers, not strings)
            try:
                collection = client.create_collection(
                    name=self.collection_name,
                    metadata={
                        "hnsw:space": "cosine",
                        "hnsw:M": 32,  # Integer, not string
                        "hnsw:construction_ef": 400,  # Integer, not string
                        "hnsw:search_ef": 100,  # Integer, not string
                    }
                )
                logger.info(f"Created collection '{self.collection_name}' with HNSW parameters: M=32, ef_construction=400")
            except Exception as e:
                # If HNSW parameters fail, create without them (will use defaults)
                logger.warning(f"Could not create collection with HNSW parameters: {e}")
                logger.warning("Creating collection with default parameters")
                try:
                    collection = client.create_collection(name=self.collection_name)
                    logger.info(f"Created collection '{self.collection_name}' with default parameters")
                except Exception as create_error:
                    # Collection might have been created by another process, try to get it
                    logger.warning(f"Collection creation failed: {create_error}, attempting to get existing collection")
                    collection = client.get_collection(name=self.collection_name)
        
        # Initialize Chroma with persistent storage and explicit client
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            client=client,
            persist_directory=persist_directory,
        )
        
        # CRITICAL FIX: Ensure collection segments are initialized by adding and removing a dummy document
        # ROOT CAUSE: Empty collections don't have segments initialized until first document is added
        # By adding and removing a dummy document, we force ChromaDB to create all necessary segments
        # This prevents StopIteration errors when adding the first real batch of documents
        try:
            doc_count = self.vector_store._collection.count()
            if doc_count == 0:
                # Collection is empty - add a dummy document to initialize segments
                logger.info("Collection is empty, initializing segments with dummy document...")
                dummy_text = "dummy_initialization"
                max_init_retries = 3
                init_success = False
                
                for init_attempt in range(max_init_retries):
                    try:
                        self.vector_store.add_texts(
                            texts=[dummy_text],
                            ids=["__dummy_init__"],
                            metadatas=[{"__init__": True}]
                        )
                        # Immediately remove it
                        self.vector_store._collection.delete(ids=["__dummy_init__"])
                        logger.info("Collection segments initialized successfully")
                        init_success = True
                        break
                    except StopIteration:
                        # Even the dummy add can fail with StopIteration - retry with delay
                        if init_attempt < max_init_retries - 1:
                            logger.debug(f"Segment initialization retry {init_attempt + 1}/{max_init_retries}...")
                            time.sleep(0.5 * (init_attempt + 1))  # Increasing delay
                        else:
                            logger.warning("Could not initialize segments with dummy document after retries")
                            logger.warning("Segments will be initialized on first real document add")
                    except Exception as init_error:
                        # Other errors - log and give up
                        logger.warning(f"Could not initialize with dummy document: {init_error}")
                        logger.warning("Segments will be initialized on first real document add")
                        break
                
                if not init_success:
                    logger.warning("Segment initialization incomplete - first real add may trigger StopIteration")
        except Exception as count_error:
            # If count fails, collection might not be fully initialized
            # This is okay - segments will be created on first add
            logger.debug(f"Could not check collection count: {count_error}")
        
        logger.info("Vector store initialized")
    
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """
        Add chunks to vector store.
        
        Args:
            chunks: List of Chunk objects to add
        """
        if not chunks:
            logger.warning("No chunks to add")
            return
        
        logger.info(f"Adding {len(chunks)} chunks to vector store")
        
        texts = [chunk.text for chunk in chunks]
        # Use chunk.metadata directly, ensuring medication_name and drug_id are set
        metadatas = []
        for chunk in chunks:
            metadata = dict(chunk.metadata) if chunk.metadata else {}
            # Ensure required fields are present
            metadata.setdefault("section", chunk.section_title)
            metadata.setdefault("source", chunk.source_document)
            metadata.setdefault("page", chunk.page_number)
            metadata.setdefault("medication_name", chunk.medication_name)
            # Ensure drug_id is set - use from metadata if present, otherwise use medication_name
            metadata.setdefault("drug_id", metadata.get("drug_id", chunk.medication_name))
            # Handle ATC codes: ChromaDB via LangChain doesn't accept lists in metadata
            # Convert list to comma-separated string for storage
            if "atc_codes" in metadata:
                atc_codes = metadata["atc_codes"]
                if isinstance(atc_codes, list):
                    if len(atc_codes) > 0:
                        # Convert list to comma-separated string
                        metadata["atc_codes"] = ",".join(str(code) for code in atc_codes)
                    else:
                        # Remove empty list
                        del metadata["atc_codes"]
                elif not atc_codes:
                    # Remove empty/None value
                    del metadata["atc_codes"]
            metadatas.append(metadata)
        ids = [
            f"{chunk.source_document}_{i}_{chunk.section_title}"
            for i, chunk in enumerate(chunks)
        ]
        
        # Handle ChromaDB segment initialization issue when collection is empty
        # This can happen after clearing the collection - segments may not be initialized yet
        max_retries = 3
        retry_delay = 0.5
        
        for attempt in range(max_retries):
            try:
                self.vector_store.add_texts(
                    texts=texts,
                    metadatas=metadatas,
                    ids=ids
                )
                break  # Success, exit retry loop
            except StopIteration as e:
                # ChromaDB segment manager issue - segments not initialized yet
                if attempt < max_retries - 1:
                    logger.warning(
                        f"ChromaDB segment initialization issue (attempt {attempt + 1}/{max_retries}). "
                        f"Retrying after {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    # Last attempt failed - try adding one document at a time to force initialization
                    logger.warning(
                        "Bulk add failed after retries. Adding documents one at a time to force segment initialization..."
                    )
                    for i, (text, metadata, doc_id) in enumerate(zip(texts, metadatas, ids)):
                        try:
                            self.vector_store.add_texts(
                                texts=[text],
                                metadatas=[metadata],
                                ids=[doc_id]
                            )
                            if (i + 1) % 10 == 0:
                                logger.info(f"Added {i + 1}/{len(chunks)} chunks individually...")
                        except Exception as individual_error:
                            logger.error(
                                f"Failed to add chunk {i} (id: {doc_id}): {individual_error}"
                            )
                            raise
                    break  # Successfully added all individually
            except Exception as e:
                # Other errors - don't retry, just raise
                logger.error(f"Error adding chunks to vector store: {e}")
                raise
        
        # Persist to disk
        self.vector_store.persist()
        
        # Refresh drug list cache after adding chunks
        self.all_drugs = self.get_all_drug_ids()
        self._unique_medications = sorted(list(set(self.all_drugs)))  # Sync medications cache with deduplication
        
        logger.info(f"Successfully added {len(chunks)} chunks")
    
    def clear_collection(self) -> None:
        """Clear all documents from the vector store."""
        logger.warning("Clearing vector store collection")
        
        # Delete collection and recreate with proper HNSW parameters
        client = chromadb.PersistentClient(
            path=str(Config.VECTOR_STORE_PATH),
            settings=Settings(anonymized_telemetry=False)
        )
        try:
            client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection '{self.collection_name}'")
        except Exception as e:
            logger.warning(f"Collection may not exist: {e}")
        
        # Reinitialize (will create collection with new HNSW parameters)
        self._initialize_store()
        
        # Reset drug list caches
        self.all_drugs = []
        self._unique_medications = []
        
        logger.info("Vector store cleared and recreated with proper HNSW parameters")
    
    def get_retriever(self, top_k: Optional[int] = None):
        """
        Get retriever for querying the vector store.
        
        Args:
            top_k: Number of documents to retrieve (defaults to Config.RETRIEVAL_TOP_K)
            
        Returns:
            VectorStoreRetriever instance (wrapped with instrumentation if Opik available)
        """
        if top_k is None:
            top_k = Config.RETRIEVAL_TOP_K
        
        base_retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 12, "fetch_k": 20, "lambda_mult": 0.3}  # Reduced fetch_k from 40 to 20
        )
        
        # Wrap retriever (Opik tracing is handled at graph level via OpikTracer callback)
        return InstrumentedRetriever(base_retriever, medication_filter=None)
    
    def get_document_count(self) -> int:
        """
        Get number of documents in the vector store.
        
        Returns:
            Number of documents
        """
        if not self.vector_store:
            return 0
        
        collection = self.vector_store._collection
        return collection.count() if collection else 0
    
    def document_exists(self, source_document: str) -> bool:
        """
        Check if a document has already been indexed.
        
        Args:
            source_document: Filename of the document to check
            
        Returns:
            True if document exists in vector store, False otherwise
        """
        if not self.vector_store:
            return False
        
        try:
            collection = self.vector_store._collection
            # Query for documents with matching source metadata
            results = collection.get(
                where={"source": source_document},
                limit=1
            )
            return len(results.get("ids", [])) > 0
        except Exception as e:
            logger.warning(f"Error checking document existence: {e}")
            return False
    
    def remove_document(self, source_document: str) -> int:
        """
        Remove all chunks for a specific document from the vector store.
        
        Args:
            source_document: Filename of the document to remove
            
        Returns:
            Number of chunks removed
        """
        if not self.vector_store:
            return 0
        
        try:
            collection = self.vector_store._collection
            # Get all IDs for this document
            results = collection.get(
                where={"source": source_document}
            )
            ids_to_delete = results.get("ids", [])
            
            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
                self.vector_store.persist()
                logger.info(f"Removed {len(ids_to_delete)} chunks for document: {source_document}")
                return len(ids_to_delete)
            return 0
        except Exception as e:
            logger.error(f"Error removing document: {e}")
            return 0
    
    def get_unique_medications(self) -> List[str]:
        """
        Get list of unique medication names from the vector store.
        
        Returns cached list for performance - avoids querying all documents.
        
        Returns:
            List of unique medication names
        """
        # Return cached list for performance (populated during __init__)
        if self._unique_medications:
            return self._unique_medications
        
        # Fallback: use all_drugs if available (same data, different source)
        if self.all_drugs:
            self._unique_medications = list(self.all_drugs)
            return self._unique_medications
        
        # Last resort: query the database (slow path, should rarely happen)
        if not self.vector_store:
            return []
        
        try:
            logger.warning("get_unique_medications: Cache miss, querying database (slow)")
            collection = self.vector_store._collection
            # Get all documents and extract unique medication names
            results = collection.get()
            metadatas = results.get("metadatas", [])
            
            medications = set()
            for metadata in metadatas:
                medication_name = metadata.get("medication_name")
                if medication_name:
                    medications.add(medication_name)
            
            self._unique_medications = sorted(list(medications))
            return self._unique_medications
        except Exception as e:
            logger.error(f"Error getting unique medications: {e}")
            return []
    
    def get_all_drug_ids(self) -> List[str]:
        """
        Get list of all unique drug_id values from the vector store.
        
        Queries Chroma collection in batches to ensure we get all unique drug_ids,
        even with large collections (e.g., 177k+ documents).
        Falls back to medication_name for backward compatibility with older documents.
        This is the authoritative source of all available drugs.
        
        Returns:
            Sorted list of unique drug_id strings
        """
        if self.vector_store is None:
            logger.warning("Vector store not initialized")
            return []
        
        try:
            collection = self.vector_store._collection
            if not collection:
                return []
            
            # First, get all IDs from the collection (this is fast, no metadata)
            all_ids_result = collection.get(include=[])
            all_ids = all_ids_result.get("ids", [])
            total_docs = len(all_ids)
            
            if total_docs == 0:
                logger.debug("Collection is empty, no drug IDs to extract")
                return []
            
            logger.info(f"Processing {total_docs} documents to extract unique drug IDs...")
            
            # Process in batches to avoid memory issues
            batch_size = 10000
            drug_ids = set()
            
            for i in range(0, total_docs, batch_size):
                batch_ids = all_ids[i:i + batch_size]
                
                # Get metadata for this batch
                batch_results = collection.get(
                    ids=batch_ids,
                    include=["metadatas"]
                )
                metadatas = batch_results.get("metadatas", [])
                
                # Extract unique drug_id values from this batch
                for metadata in metadatas:
                    # Prefer drug_id, fallback to medication_name for backward compatibility
                    drug_id = metadata.get("drug_id") or metadata.get("medication_name")
                    if drug_id:
                        drug_ids.add(drug_id)
                
                # Log progress for large collections
                if (i + batch_size) % 50000 == 0 or (i + batch_size) >= total_docs:
                    logger.info(f"Processed {min(i + batch_size, total_docs)}/{total_docs} documents, found {len(drug_ids)} unique drug IDs so far...")
            
            result = sorted(list(drug_ids))
            logger.info(f"Found {len(result)} unique drug IDs in vector store (from {total_docs} documents)")
            return result
        except (StopIteration, AttributeError, Exception) as e:
            # Collection may be empty or not fully initialized
            logger.debug(f"Could not get drug IDs (collection may be empty): {e}")
            return []

    @property
    def all_drugs_list(self) -> List[str]:
        """
        Lazy-loaded accessor for all known drug IDs.
        
        Returns:
            List of unique drug IDs from the vector store
        """
        if not self.all_drugs:
            self.all_drugs = self.get_all_drug_ids()
        return self.all_drugs
    
    def get_unique_documents(self) -> List[Dict[str, Any]]:
        """
        Get list of unique processed documents with their metadata.
        
        Returns:
            List of dictionaries with 'filename', 'medication_name', and 'chunk_count' keys
        """
        if not self.vector_store:
            return []
        
        try:
            collection = self.vector_store._collection
            results = collection.get()
            metadatas = results.get("metadatas", [])
            ids = results.get("ids", [])
            
            # Group by source document
            doc_info: Dict[str, Dict[str, Any]] = {}
            for metadata, doc_id in zip(metadatas, ids):
                source = metadata.get("source", "Unknown")
                medication_name = metadata.get("medication_name", "Unknown")
                
                if source not in doc_info:
                    doc_info[source] = {
                        "filename": source,
                        "medication_name": medication_name,
                        "chunk_count": 0
                    }
                doc_info[source]["chunk_count"] += 1
            
            # Convert to sorted list
            return sorted(
                list(doc_info.values()),
                key=lambda x: x["filename"]
            )
        except Exception as e:
            logger.error(f"Error getting unique documents: {e}")
            return []
    
    def get_retriever_with_filter(
        self, 
        medication_name: Optional[str] = None,
        drug_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None
    ):
        """
        Get retriever with optional medication filtering.
        
        Supports filtering by either a single medication_name (legacy) or
        multiple drug_ids (preferred). Uses drug_id field for more reliable filtering.
        
        Args:
            medication_name: Optional single medication name to filter by (legacy parameter)
            drug_ids: Optional list of drug_id strings to filter by (preferred)
            top_k: Number of documents to retrieve (defaults to Config.RETRIEVAL_TOP_K)
            
        Returns:
            VectorStoreRetriever instance with optional filter (wrapped with instrumentation if Opik available)
        """
        if top_k is None:
            top_k = Config.RETRIEVAL_TOP_K
        
        search_kwargs = {"k": top_k}
        
        # Prefer drug_ids over medication_name (more reliable)
        if drug_ids:
            if len(drug_ids) == 1:
                # Single drug_id: simple equality filter
                search_kwargs["filter"] = {"drug_id": drug_ids[0]}
            else:
                # Multiple drug_ids: use ChromaDB $in operator
                search_kwargs["filter"] = {"drug_id": {"$in": drug_ids}}
        elif medication_name:
            # Legacy support: filter by medication_name
            search_kwargs["filter"] = {"medication_name": medication_name}
        
        base_retriever = self.vector_store.as_retriever(
            search_kwargs=search_kwargs
        )
        
        # Wrap retriever (Opik tracing is handled at graph level via OpikTracer callback)
        filter_value = ", ".join(drug_ids) if drug_ids else medication_name
        return InstrumentedRetriever(base_retriever, medication_filter=filter_value)
    
    def get_drugs_by_atc(self, atc_code: str) -> List[str]:
        """
        Get all drug_ids that have a specific ATC code.
        
        Args:
            atc_code: ATC code (can be partial, e.g., "A10BA" or full "A10BA02")
            
        Returns:
            List of unique drug_id strings
        """
        if not self.vector_store:
            return []
        
        try:
            collection = self.vector_store._collection
            
            # ATC codes are stored as comma-separated strings
            # ChromaDB doesn't support $contains for strings, so we need to:
            # Get all documents and filter in Python
            # This is less efficient but necessary for string matching
            
            # Get all documents (we'll filter by atc_codes in Python)
            # Limit to reasonable batch size to avoid memory issues
            results = collection.get(
                limit=100000,  # Large limit to get all docs
                include=["metadatas"]
            )
            
            drug_ids = set()
            metadatas = results.get("metadatas", [])
            for metadata in metadatas:
                atc_codes_str = metadata.get("atc_codes", "")
                if atc_codes_str:
                    # Parse comma-separated string
                    codes = [c.strip() for c in atc_codes_str.split(",")]
                    # Check if any code matches (exact or starts with)
                    if any(code == atc_code or code.startswith(atc_code) or atc_code in code for code in codes):
                        drug_id = metadata.get("drug_id")
                        if drug_id:
                            drug_ids.add(drug_id)
            
            return sorted(list(drug_ids))
        except Exception as e:
            logger.error(f"Error getting drugs by ATC code: {e}", exc_info=True)
            return []
    
    def get_retriever_by_atc(
        self,
        atc_code: str,
        top_k: Optional[int] = None
    ):
        """
        Get retriever filtered by ATC code.
        
        Args:
            atc_code: ATC code to filter by
            top_k: Number of documents to retrieve
            
        Returns:
            VectorStoreRetriever instance filtered by ATC code
        """
        if top_k is None:
            top_k = Config.RETRIEVAL_TOP_K
        
        # Get all drugs with this ATC code
        drug_ids = self.get_drugs_by_atc(atc_code)
        
        if not drug_ids:
            # No drugs found with this ATC code
            # Return a retriever that filters to non-existent drug_id (will return empty)
            search_kwargs = {
                "k": top_k,
                "filter": {"drug_id": "__NO_DRUGS_WITH_ATC__"}
            }
            base_retriever = self.vector_store.as_retriever(search_kwargs=search_kwargs)
            return InstrumentedRetriever(base_retriever, medication_filter=f"ATC:{atc_code}")
        
        # Use drug_id filtering (ChromaDB supports $in for multiple values)
        if len(drug_ids) == 1:
            search_kwargs = {
                "k": top_k,
                "filter": {"drug_id": drug_ids[0]}
            }
        else:
            search_kwargs = {
                "k": top_k,
                "filter": {"drug_id": {"$in": drug_ids}}
            }
        
        base_retriever = self.vector_store.as_retriever(
            search_kwargs=search_kwargs
        )
        
        return InstrumentedRetriever(base_retriever, medication_filter=f"ATC:{atc_code}")
    
    def get_retriever_by_ingredients(
        self,
        ingredient_names: List[str],
        top_k: Optional[int] = None
    ):
        """
        Get retriever filtered by active ingredients.
        
        Uses IngredientsManager to find all drugs with these ingredients,
        then filters vector store by those drug_ids.
        
        Args:
            ingredient_names: List of active ingredient names (INN names)
            top_k: Number of documents to retrieve
            
        Returns:
            VectorStoreRetriever instance filtered by active ingredients
        """
        if top_k is None:
            top_k = Config.RETRIEVAL_TOP_K
        
        if not ingredient_names:
            # No ingredients specified, return empty retriever
            search_kwargs = {
                "k": top_k,
                "filter": {"drug_id": "__NO_INGREDIENTS_SPECIFIED__"}
            }
            base_retriever = self.vector_store.as_retriever(search_kwargs=search_kwargs)
            return InstrumentedRetriever(base_retriever, medication_filter="Ingredients:None")
        
        # Use singleton IngredientsManager for performance (avoids reloading JSON on every call)
        try:
            from src.ingredients_manager import get_ingredients_manager
            ingredients_manager = get_ingredients_manager()
        except Exception as e:
            logger.error(f"Could not load IngredientsManager: {e}")
            # Return empty retriever
            search_kwargs = {
                "k": top_k,
                "filter": {"drug_id": "__INGREDIENTS_MANAGER_ERROR__"}
            }
            base_retriever = self.vector_store.as_retriever(search_kwargs=search_kwargs)
            return InstrumentedRetriever(base_retriever, medication_filter="Ingredients:Error")
        
        # Get all drugs with any of the specified ingredients
        all_drug_ids = set()
        for ingredient in ingredient_names:
            drugs_with_ingredient = ingredients_manager.get_drugs_by_ingredient(ingredient)
            all_drug_ids.update(drugs_with_ingredient)
        
        if not all_drug_ids:
            # No drugs found with these ingredients
            search_kwargs = {
                "k": top_k,
                "filter": {"drug_id": "__NO_DRUGS_WITH_INGREDIENTS__"}
            }
            base_retriever = self.vector_store.as_retriever(search_kwargs=search_kwargs)
            ingredient_str = ", ".join(ingredient_names)
            return InstrumentedRetriever(base_retriever, medication_filter=f"Ingredients:{ingredient_str}")
        
        # Filter to only drugs that exist in vector store
        available_medications = self.get_unique_medications()
        available_drug_ids = []
        
        for drug_id in all_drug_ids:
            # Normalize drug name for matching
            # Convert underscores to spaces to handle mismatch between ingredients index (spaces)
            # and vector store (underscores), e.g., "Dicloxacillin Bluefish" vs "Dicloxacillin_Bluefish_SmPC"
            drug_normalized = drug_id.lower().replace("_smpc", "").replace("_smPC", "").replace("_", " ").strip()
            
            # Try to match drug name to available medications
            for available in available_medications:
                available_normalized = available.lower().replace("_smpc", "").replace("_smPC", "").replace("_", " ").strip()
                
                # Match if normalized names are similar
                if (drug_normalized == available_normalized or
                    drug_normalized in available_normalized or
                    available_normalized in drug_normalized):
                    available_drug_ids.append(available)
                    break
        
        if not available_drug_ids:
            # No matching drugs in vector store
            search_kwargs = {
                "k": top_k,
                "filter": {"drug_id": "__NO_MATCHING_DRUGS_IN_STORE__"}
            }
            base_retriever = self.vector_store.as_retriever(search_kwargs=search_kwargs)
            ingredient_str = ", ".join(ingredient_names)
            return InstrumentedRetriever(base_retriever, medication_filter=f"Ingredients:{ingredient_str}")
        
        # Use drug_id filtering (ChromaDB supports $in for multiple values)
        if len(available_drug_ids) == 1:
            search_kwargs = {
                "k": top_k,
                "filter": {"drug_id": available_drug_ids[0]}
            }
        else:
            search_kwargs = {
                "k": top_k,
                "filter": {"drug_id": {"$in": available_drug_ids}}
            }
        
        base_retriever = self.vector_store.as_retriever(
            search_kwargs=search_kwargs
        )
        
        ingredient_str = ", ".join(ingredient_names)
        return InstrumentedRetriever(base_retriever, medication_filter=f"Ingredients:{ingredient_str}")