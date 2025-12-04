"""Vector store management with Chroma and multilingual embeddings."""
import logging
import os
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
        self._initialize_store()
        # Load all drug IDs after store initialization
        self.all_drugs = self.get_all_drug_ids()
        logger.info(f"Loaded {len(self.all_drugs)} unique drug IDs")
    
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
        # This prevents telemetry events that conflict with Opik
        client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize Chroma with persistent storage and explicit client
        # Note: persist_directory is required even when using PersistentClient
        # for the LangChain wrapper's persist() method to work
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            client=client,
            persist_directory=persist_directory,
        )
        
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
            # Ensure ATC codes are included (may be empty list if not enriched)
            if "atc_codes" not in metadata:
                metadata["atc_codes"] = []
            metadatas.append(metadata)
        ids = [
            f"{chunk.source_document}_{i}_{chunk.section_title}"
            for i, chunk in enumerate(chunks)
        ]
        
        self.vector_store.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        # Persist to disk
        self.vector_store.persist()
        
        # Refresh drug list cache after adding chunks
        self.all_drugs = self.get_all_drug_ids()
        
        logger.info(f"Successfully added {len(chunks)} chunks")
    
    def clear_collection(self) -> None:
        """Clear all documents from the vector store."""
        logger.warning("Clearing vector store collection")
        
        # Delete collection and recreate
        if self.vector_store:
            # Get client and delete collection
            # Disable telemetry to avoid PostHog compatibility errors
            client = chromadb.PersistentClient(
                path=str(Config.VECTOR_STORE_PATH),
                settings=Settings(anonymized_telemetry=False)
            )
            try:
                client.delete_collection(self.collection_name)
            except Exception as e:
                logger.warning(f"Collection may not exist: {e}")
        
        # Reinitialize
        self._initialize_store()
        
        # Reset drug list cache
        self.all_drugs = []
        
        logger.info("Vector store cleared")
    
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
        
        Returns:
            List of unique medication names
        """
        if not self.vector_store:
            return []
        
        try:
            collection = self.vector_store._collection
            # Get all documents and extract unique medication names
            results = collection.get()
            metadatas = results.get("metadatas", [])
            
            medications = set()
            for metadata in metadatas:
                medication_name = metadata.get("medication_name")
                if medication_name:
                    medications.add(medication_name)
            
            return sorted(list(medications))
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
        if not self.vector_store:
            logger.warning("Vector store not initialized")
            return []
        
        try:
            collection = self.vector_store._collection
            
            # First, get all IDs from the collection (this is fast, no metadata)
            all_ids_result = collection.get(include=[])
            all_ids = all_ids_result.get("ids", [])
            total_docs = len(all_ids)
            
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
        except Exception as e:
            logger.error(f"Error getting all drug IDs: {e}", exc_info=True)
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
            
            # Query for documents with matching ATC codes
            # ChromaDB supports array contains queries
            results = collection.get(
                where={"atc_codes": {"$contains": atc_code}},
                include=["metadatas"]
            )
            
            drug_ids = set()
            metadatas = results.get("metadatas", [])
            for metadata in metadatas:
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
        
        search_kwargs = {
            "k": top_k,
            "filter": {"atc_codes": {"$contains": atc_code}}
        }
        
        base_retriever = self.vector_store.as_retriever(
            search_kwargs=search_kwargs
        )
        
        return InstrumentedRetriever(base_retriever, medication_filter=f"ATC:{atc_code}")
