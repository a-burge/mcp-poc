"""Vector store management with Chroma and multilingual embeddings."""
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from config import Config
from src.chunker import Chunk

logger = logging.getLogger(__name__)

# Try to import opik for instrumentation
try:
    import opik
    OPIK_AVAILABLE = True
except ImportError:
    OPIK_AVAILABLE = False


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
        self._initialize_store()
    
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
        
        # Initialize Chroma with persistent storage
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
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
        # Use chunk.metadata directly, ensuring medication_name is set
        metadatas = []
        for chunk in chunks:
            metadata = dict(chunk.metadata) if chunk.metadata else {}
            # Ensure required fields are present
            metadata.setdefault("section", chunk.section_title)
            metadata.setdefault("source", chunk.source_document)
            metadata.setdefault("page", chunk.page_number)
            metadata.setdefault("medication_name", chunk.medication_name)
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
        
        logger.info(f"Successfully added {len(chunks)} chunks")
    
    def clear_collection(self) -> None:
        """Clear all documents from the vector store."""
        logger.warning("Clearing vector store collection")
        
        # Delete collection and recreate
        if self.vector_store:
            # Get client and delete collection
            client = chromadb.PersistentClient(
                path=str(Config.VECTOR_STORE_PATH)
            )
            try:
                client.delete_collection(self.collection_name)
            except Exception as e:
                logger.warning(f"Collection may not exist: {e}")
        
        # Reinitialize
        self._initialize_store()
        
        logger.info("Vector store cleared")
    
    def get_retriever(self, top_k: Optional[int] = None):
        """
        Get retriever for querying the vector store.
        
        Args:
            top_k: Number of documents to retrieve (defaults to Config.RETRIEVAL_TOP_K)
            
        Returns:
            VectorStoreRetriever instance
        """
        if top_k is None:
            top_k = Config.RETRIEVAL_TOP_K
        
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": top_k}
        )
        
        # Wrap retriever to add Opik instrumentation
        if OPIK_AVAILABLE:
            original_get_relevant_documents = retriever.get_relevant_documents
            
            def instrumented_get_relevant_documents(query: str, **kwargs):
                """Wrapper to log retrieval operations with Opik."""
                logger.info(f"Retrieving documents for query: {query[:50]}...")
                
                # Log retrieval start
                opik.log_event("retrieval_start", {
                    "query": query,
                    "top_k": top_k,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                # Perform retrieval
                docs = original_get_relevant_documents(query, **kwargs)
                
                # Log retrieval results
                retrieval_metadata = []
                for doc in docs:
                    metadata = doc.metadata
                    retrieval_metadata.append({
                        "drug_id": metadata.get("drug_id", metadata.get("medication_name", "Unknown")),
                        "section_number": metadata.get("section_number", "Unknown"),
                        "section_title": metadata.get("section_title", metadata.get("section", "Unknown")),
                        "canonical_key": metadata.get("canonical_key", "Unknown"),
                        "version_hash": metadata.get("version_hash", "Unknown"),
                        "similarity_score": getattr(doc, "score", None)
                    })
                
                opik.log_event("retrieval_complete", {
                    "query": query,
                    "chunks_retrieved": len(docs),
                    "metadata": retrieval_metadata,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                return docs
            
            retriever.get_relevant_documents = instrumented_get_relevant_documents
        
        return retriever
    
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
    
    def get_retriever_with_filter(self, medication_name: Optional[str] = None, top_k: Optional[int] = None):
        """
        Get retriever with optional medication filtering.
        
        Args:
            medication_name: Optional medication name to filter by
            top_k: Number of documents to retrieve (defaults to Config.RETRIEVAL_TOP_K)
            
        Returns:
            VectorStoreRetriever instance with optional filter
        """
        if top_k is None:
            top_k = Config.RETRIEVAL_TOP_K
        
        search_kwargs = {"k": top_k}
        
        # Add medication filter if specified
        if medication_name:
            search_kwargs["filter"] = {"medication_name": medication_name}
        
        retriever = self.vector_store.as_retriever(
            search_kwargs=search_kwargs
        )
        
        # Wrap retriever to add Opik instrumentation
        if OPIK_AVAILABLE:
            original_get_relevant_documents = retriever.get_relevant_documents
            
            def instrumented_get_relevant_documents(query: str, **kwargs):
                """Wrapper to log retrieval operations with Opik."""
                logger.info(f"Retrieving documents for query: {query[:50]}... (filter: {medication_name})")
                
                # Log retrieval start
                opik.log_event("retrieval_start", {
                    "query": query,
                    "medication_filter": medication_name,
                    "top_k": top_k,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                # Perform retrieval
                docs = original_get_relevant_documents(query, **kwargs)
                
                # Log retrieval results
                retrieval_metadata = []
                for doc in docs:
                    metadata = doc.metadata
                    retrieval_metadata.append({
                        "drug_id": metadata.get("drug_id", metadata.get("medication_name", "Unknown")),
                        "section_number": metadata.get("section_number", "Unknown"),
                        "section_title": metadata.get("section_title", metadata.get("section", "Unknown")),
                        "canonical_key": metadata.get("canonical_key", "Unknown"),
                        "version_hash": metadata.get("version_hash", "Unknown"),
                        "similarity_score": getattr(doc, "score", None)
                    })
                
                opik.log_event("retrieval_complete", {
                    "query": query,
                    "medication_filter": medication_name,
                    "chunks_retrieved": len(docs),
                    "metadata": retrieval_metadata,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                return docs
            
            retriever.get_relevant_documents = instrumented_get_relevant_documents
        
        return retriever
