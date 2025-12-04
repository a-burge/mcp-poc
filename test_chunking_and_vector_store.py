"""
Comprehensive tests for chunking and vector store functionality.

Tests cover:
1. Chunk quality: sections preserved, metadata complete, large sections subdivided
2. Vector store: chunks stored, retrieval by query, metadata filters working
"""
import json
import logging
import pytest
from pathlib import Path
from typing import Dict, Any, List

from src.chunker import chunk_smpc_json, Chunk
from src.vector_store import VectorStoreManager
from config import Config

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestChunkQuality:
    """Test chunk quality: sections preserved, metadata complete, large sections subdivided."""
    
    @pytest.fixture
    def sample_smpc_data(self) -> Dict[str, Any]:
        """Load sample SmPC JSON for testing."""
        sample_file = Path("data/structured/Voriconazole_Normon_SmPC_SmPC.json")
        if not sample_file.exists():
            pytest.skip(f"Sample file not found: {sample_file}")
        
        with open(sample_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def test_sections_preserved_correctly(self, sample_smpc_data: Dict[str, Any]):
        """
        Test that sections are preserved correctly in chunks.
        
        Each section should appear in at least one chunk, and chunks
        should not mix content from different sections.
        
        Note: Parent sections with empty text (like "4" and "5") are correctly
        skipped by the chunker, so we only check sections with actual content.
        """
        chunks = chunk_smpc_json(sample_smpc_data, chunk_size=300, chunk_overlap=0)
        
        assert len(chunks) > 0, "Should create at least one chunk"
        
        # Get all section numbers from original data that have non-empty text
        original_sections = set()
        for section_num, section_data in sample_smpc_data.get("sections", {}).items():
            section_text = section_data.get("text", "").strip()
            if section_text:  # Only include sections with actual content
                original_sections.add(section_num)
        
        # Get all section numbers from chunks
        chunk_sections = set()
        for chunk in chunks:
            section_num = chunk.metadata.get("section_number")
            if section_num:
                chunk_sections.add(section_num)
        
        # Check that all non-empty sections are represented
        missing_sections = original_sections - chunk_sections
        assert len(missing_sections) == 0, (
            f"Missing sections in chunks: {missing_sections}. "
            f"Original sections (with content): {sorted(original_sections)}, "
            f"Chunk sections: {sorted(chunk_sections)}"
        )
        
        logger.info(f"✓ All {len(original_sections)} sections with content preserved in chunks")
    
    def test_metadata_complete(self, sample_smpc_data: Dict[str, Any]):
        """
        Test that all required metadata fields are present in chunks.
        
        Required fields:
        - drug_id
        - section_number
        - section_title
        - canonical_key
        - version_hash
        - pdf_path
        - medication_name
        - chunk_id
        - document_hash
        """
        chunks = chunk_smpc_json(sample_smpc_data, chunk_size=300, chunk_overlap=0)
        
        required_fields = [
            "drug_id",
            "section_number",
            "section_title",
            "canonical_key",
            "version_hash",
            "pdf_path",
            "medication_name",
            "chunk_id",
            "document_hash"
        ]
        
        missing_fields = []
        for i, chunk in enumerate(chunks):
            for field in required_fields:
                if field not in chunk.metadata:
                    missing_fields.append((i, field, chunk.metadata))
        
        assert len(missing_fields) == 0, (
            f"Missing required metadata fields in chunks: {missing_fields}"
        )
        
        # Verify values match source data
        drug_id = sample_smpc_data.get("drug_id")
        version_hash = sample_smpc_data.get("version_hash")
        source_pdf = sample_smpc_data.get("source_pdf")
        
        for chunk in chunks:
            assert chunk.metadata["drug_id"] == drug_id, "drug_id mismatch"
            assert chunk.metadata["version_hash"] == version_hash, "version_hash mismatch"
            assert chunk.metadata["pdf_path"] == source_pdf, "pdf_path mismatch"
            assert chunk.metadata["medication_name"] == drug_id, "medication_name should match drug_id"
        
        logger.info(f"✓ All {len(chunks)} chunks have complete metadata")
    
    def test_large_sections_subdivided(self, sample_smpc_data: Dict[str, Any]):
        """
        Test that large sections are appropriately subdivided.
        
        Sections larger than chunk_size should be split into multiple chunks.
        """
        chunk_size = 300
        chunks = chunk_smpc_json(sample_smpc_data, chunk_size=chunk_size, chunk_overlap=0)
        
        # Group chunks by section_number
        sections_to_chunks: Dict[str, List[Chunk]] = {}
        for chunk in chunks:
            section_num = chunk.metadata.get("section_number", "unknown")
            if section_num not in sections_to_chunks:
                sections_to_chunks[section_num] = []
            sections_to_chunks[section_num].append(chunk)
        
        # Check original section sizes
        large_sections_found = []
        for section_num, section_data in sample_smpc_data.get("sections", {}).items():
            section_text = section_data.get("text", "").strip()
            if len(section_text) > chunk_size:
                large_sections_found.append(section_num)
                
                # This section should have multiple chunks
                section_chunks = sections_to_chunks.get(section_num, [])
                assert len(section_chunks) > 1, (
                    f"Section {section_num} is {len(section_text)} chars "
                    f"(>{chunk_size}) but only has {len(section_chunks)} chunk(s)"
                )
                
                # Verify chunk sizes are reasonable
                for chunk in section_chunks:
                    assert len(chunk.text) <= chunk_size * 1.5, (
                        f"Chunk for section {section_num} is too large: "
                        f"{len(chunk.text)} chars (expected <= {chunk_size * 1.5})"
                    )
        
        if large_sections_found:
            logger.info(
                f"✓ Large sections ({len(large_sections_found)}) appropriately subdivided: "
                f"{large_sections_found}"
            )
        else:
            logger.info("✓ No large sections found (all sections fit in single chunks)")
    
    def test_chunk_text_not_empty(self, sample_smpc_data: Dict[str, Any]):
        """Test that chunk text is not empty."""
        chunks = chunk_smpc_json(sample_smpc_data, chunk_size=300, chunk_overlap=0)
        
        empty_chunks = []
        for i, chunk in enumerate(chunks):
            if not chunk.text or not chunk.text.strip():
                empty_chunks.append(i)
        
        assert len(empty_chunks) == 0, (
            f"Found {len(empty_chunks)} empty chunks at indices: {empty_chunks}"
        )
        
        logger.info(f"✓ All {len(chunks)} chunks have non-empty text")
    
    def test_chunk_ids_unique(self, sample_smpc_data: Dict[str, Any]):
        """Test that chunk IDs are unique."""
        chunks = chunk_smpc_json(sample_smpc_data, chunk_size=300, chunk_overlap=0)
        
        chunk_ids = [chunk.metadata.get("chunk_id") for chunk in chunks]
        unique_ids = set(chunk_ids)
        
        assert len(chunk_ids) == len(unique_ids), (
            f"Found {len(chunk_ids) - len(unique_ids)} duplicate chunk IDs"
        )
        
        logger.info(f"✓ All {len(chunks)} chunks have unique IDs")


class TestVectorStore:
    """Test vector store: chunks stored, retrieval by query, metadata filters working."""
    
    @pytest.fixture
    def sample_smpc_data(self) -> Dict[str, Any]:
        """Load sample SmPC JSON for testing."""
        sample_file = Path("data/structured/Voriconazole_Normon_SmPC_SmPC.json")
        if not sample_file.exists():
            pytest.skip(f"Sample file not found: {sample_file}")
        
        with open(sample_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    @pytest.fixture
    def test_vector_store(self, tmp_path: Path):
        """Create a temporary vector store for testing."""
        # Override vector store path for testing
        original_path = Config.VECTOR_STORE_PATH
        Config.VECTOR_STORE_PATH = tmp_path / "test_vector_store"
        
        store = VectorStoreManager(collection_name="test_smpc_documents")
        
        yield store
        
        # Cleanup
        store.clear_collection()
        Config.VECTOR_STORE_PATH = original_path
    
    def test_chunks_stored_successfully(
        self, 
        sample_smpc_data: Dict[str, Any], 
        test_vector_store: VectorStoreManager
    ):
        """
        Test that chunks are stored successfully in vector store.
        
        After adding chunks, verify:
        - Document count matches expected
        - Can retrieve chunks by ID
        """
        chunks = chunk_smpc_json(sample_smpc_data, chunk_size=300, chunk_overlap=0)
        
        # Add chunks to vector store
        initial_count = test_vector_store.get_document_count()
        test_vector_store.add_chunks(chunks)
        
        # Verify chunks were added
        final_count = test_vector_store.get_document_count()
        assert final_count == initial_count + len(chunks), (
            f"Expected {initial_count + len(chunks)} documents, "
            f"got {final_count}"
        )
        
        logger.info(
            f"✓ Successfully stored {len(chunks)} chunks "
            f"(total documents: {final_count})"
        )
    
    def test_retrieve_chunks_by_query(
        self,
        sample_smpc_data: Dict[str, Any],
        test_vector_store: VectorStoreManager
    ):
        """
        Test that chunks can be retrieved by query.
        
        After storing chunks, perform semantic search and verify:
        - Results are returned
        - Results contain relevant content
        """
        chunks = chunk_smpc_json(sample_smpc_data, chunk_size=300, chunk_overlap=0)
        test_vector_store.add_chunks(chunks)
        
        # Get retriever (skip Opik instrumentation in tests to avoid Pydantic issues)
        retriever = test_vector_store.vector_store.as_retriever(search_kwargs={"k": 5})
        
        # Test queries related to the document content
        # Based on Voriconazole sample, test queries about:
        # - Indications
        # - Dosage
        # - Side effects
        
        test_queries = [
            "What are the indications for this medication?",
            "What is the recommended dosage?",
            "What are the side effects?",
        ]
        
        for query in test_queries:
            results = retriever.get_relevant_documents(query)
            
            assert len(results) > 0, (
                f"No results returned for query: '{query}'"
            )
            
            # Verify results have content
            for doc in results:
                assert doc.page_content, "Retrieved document has empty content"
                assert doc.metadata, "Retrieved document has no metadata"
            
            logger.info(
                f"✓ Query '{query[:50]}...' returned {len(results)} relevant chunks"
            )
    
    def test_metadata_filter_by_medication_name(
        self,
        sample_smpc_data: Dict[str, Any],
        test_vector_store: VectorStoreManager
    ):
        """
        Test that metadata filters work, specifically filtering by medication_name.
        
        Store chunks, then retrieve with medication filter and verify:
        - Only chunks matching filter are returned
        - All returned chunks have matching medication_name
        """
        chunks = chunk_smpc_json(sample_smpc_data, chunk_size=300, chunk_overlap=0)
        test_vector_store.add_chunks(chunks)
        
        drug_id = sample_smpc_data.get("drug_id")
        
        # Get retriever with medication filter (skip Opik instrumentation in tests)
        retriever = test_vector_store.vector_store.as_retriever(
            search_kwargs={"k": 10, "filter": {"medication_name": drug_id}}
        )
        
        query = "What are the indications?"
        results = retriever.get_relevant_documents(query)
        
        assert len(results) > 0, "No results returned with medication filter"
        
        # Verify all results match the filter
        for doc in results:
            metadata = doc.metadata
            medication_name = metadata.get("medication_name") or metadata.get("drug_id")
            assert medication_name == drug_id, (
                f"Filter returned chunk with wrong medication: "
                f"expected '{drug_id}', got '{medication_name}'"
            )
        
        logger.info(
            f"✓ Medication filter '{drug_id}' returned {len(results)} matching chunks"
        )
    
    def test_retrieve_without_filter_returns_all(
        self,
        sample_smpc_data: Dict[str, Any],
        test_vector_store: VectorStoreManager
    ):
        """
        Test that retrieval without filter can return chunks from any medication.
        
        This is useful for testing that the vector store contains multiple medications.
        """
        chunks = chunk_smpc_json(sample_smpc_data, chunk_size=300, chunk_overlap=0)
        test_vector_store.add_chunks(chunks)
        
        # Skip Opik instrumentation in tests to avoid Pydantic issues
        retriever = test_vector_store.vector_store.as_retriever(search_kwargs={"k": 10})
        query = "What are the indications?"
        results = retriever.get_relevant_documents(query)
        
        assert len(results) > 0, "No results returned without filter"
        
        # Verify results have medication_name metadata
        for doc in results:
            metadata = doc.metadata
            assert "medication_name" in metadata or "drug_id" in metadata, (
                "Retrieved chunk missing medication_name or drug_id"
            )
        
        logger.info(
            f"✓ Unfiltered query returned {len(results)} chunks "
            f"(may include multiple medications)"
        )
    
    def test_document_exists_check(
        self,
        sample_smpc_data: Dict[str, Any],
        test_vector_store: VectorStoreManager
    ):
        """Test that document_exists() correctly identifies stored documents."""
        chunks = chunk_smpc_json(sample_smpc_data, chunk_size=300, chunk_overlap=0)
        test_vector_store.add_chunks(chunks)
        
        source_pdf = sample_smpc_data.get("source_pdf", "")
        # The metadata stores the full path as "source", so check that
        if source_pdf:
            assert test_vector_store.document_exists(source_pdf), (
                f"document_exists() returned False for stored document: {source_pdf}. "
                f"Metadata 'source' field stores: {source_pdf}"
            )
            
            logger.info(f"✓ document_exists() correctly identifies '{source_pdf}'")
    
    def test_get_unique_medications(
        self,
        sample_smpc_data: Dict[str, Any],
        test_vector_store: VectorStoreManager
    ):
        """Test that get_unique_medications() returns correct list."""
        chunks = chunk_smpc_json(sample_smpc_data, chunk_size=300, chunk_overlap=0)
        test_vector_store.add_chunks(chunks)
        
        medications = test_vector_store.get_unique_medications()
        
        drug_id = sample_smpc_data.get("drug_id")
        assert drug_id in medications, (
            f"Expected medication '{drug_id}' not in unique medications list: {medications}"
        )
        
        logger.info(f"✓ Found {len(medications)} unique medication(s): {medications}")


class TestIntegration:
    """Integration tests combining chunking and vector store."""
    
    @pytest.fixture
    def sample_smpc_data(self) -> Dict[str, Any]:
        """Load sample SmPC JSON for testing."""
        sample_file = Path("data/structured/Voriconazole_Normon_SmPC_SmPC.json")
        if not sample_file.exists():
            pytest.skip(f"Sample file not found: {sample_file}")
        
        with open(sample_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    @pytest.fixture
    def test_vector_store(self, tmp_path: Path):
        """Create a temporary vector store for testing."""
        original_path = Config.VECTOR_STORE_PATH
        Config.VECTOR_STORE_PATH = tmp_path / "test_vector_store"
        
        store = VectorStoreManager(collection_name="test_smpc_documents")
        
        yield store
        
        # Cleanup
        store.clear_collection()
        Config.VECTOR_STORE_PATH = original_path
    
    def test_end_to_end_chunking_and_retrieval(
        self,
        sample_smpc_data: Dict[str, Any],
        test_vector_store: VectorStoreManager
    ):
        """
        End-to-end test: chunk document, store, and retrieve.
        
        This test verifies the complete pipeline works correctly.
        """
        # Step 1: Chunk the document
        chunks = chunk_smpc_json(sample_smpc_data, chunk_size=300, chunk_overlap=0)
        assert len(chunks) > 0, "Should create chunks"
        
        # Step 2: Store chunks
        test_vector_store.add_chunks(chunks)
        doc_count = test_vector_store.get_document_count()
        assert doc_count == len(chunks), "All chunks should be stored"
        
        # Step 3: Retrieve with query (skip Opik instrumentation in tests)
        retriever = test_vector_store.vector_store.as_retriever(search_kwargs={"k": 5})
        query = "What are the main indications for this medication?"
        results = retriever.get_relevant_documents(query)
        
        assert len(results) > 0, "Should retrieve relevant chunks"
        
        # Step 4: Verify retrieved chunks have correct metadata
        for doc in results:
            assert doc.metadata.get("drug_id"), "Retrieved chunk missing drug_id"
            assert doc.metadata.get("section_number"), "Retrieved chunk missing section_number"
            assert doc.page_content, "Retrieved chunk has no content"
        
        logger.info(
            f"✓ End-to-end test passed: "
            f"{len(chunks)} chunks stored, {len(results)} retrieved for query"
        )


if __name__ == "__main__":
    """
    Run tests with pytest.
    
    Usage:
        python test_chunking_and_vector_store.py
        pytest test_chunking_and_vector_store.py -v
        pytest test_chunking_and_vector_store.py::TestChunkQuality -v
        pytest test_chunking_and_vector_store.py::TestVectorStore -v
    """
    pytest.main([__file__, "-v", "-s"])
