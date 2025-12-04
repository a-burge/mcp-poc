"""
Standalone test script for chunking and vector store functionality.

Can be run directly without pytest. Provides detailed output about test results.

Usage:
    python test_chunking_and_vector_store_standalone.py
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

from src.chunker import chunk_smpc_json, Chunk
from src.vector_store import VectorStoreManager
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_chunk_quality(sample_file: Path) -> Tuple[bool, List[str]]:
    """
    Test chunk quality: sections preserved, metadata complete, large sections subdivided.
    
    Returns:
        Tuple of (all_passed: bool, issues: List[str])
    """
    logger.info("=" * 60)
    logger.info("TESTING CHUNK QUALITY")
    logger.info("=" * 60)
    
    if not sample_file.exists():
        return False, [f"Sample file not found: {sample_file}"]
    
    with open(sample_file, "r", encoding="utf-8") as f:
        sample_smpc_data = json.load(f)
    
    chunks = chunk_smpc_json(sample_smpc_data, chunk_size=300, chunk_overlap=0)
    issues = []
    
    # Test 1: Sections preserved correctly
    logger.info("\n1. Testing sections preserved correctly...")
    # Only check sections with actual content (parent sections with empty text are correctly skipped)
    original_sections = set()
    for section_num, section_data in sample_smpc_data.get("sections", {}).items():
        section_text = section_data.get("text", "").strip()
        if section_text:  # Only include sections with actual content
            original_sections.add(section_num)
    
    chunk_sections = set()
    for chunk in chunks:
        section_num = chunk.metadata.get("section_number")
        if section_num:
            chunk_sections.add(section_num)
    
    missing_sections = original_sections - chunk_sections
    if missing_sections:
        issues.append(f"Missing sections in chunks: {sorted(missing_sections)}")
        logger.error(f"  ✗ Missing sections: {sorted(missing_sections)}")
    else:
        logger.info(f"  ✓ All {len(original_sections)} sections with content preserved in chunks")
    
    # Test 2: Metadata complete
    logger.info("\n2. Testing metadata completeness...")
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
                missing_fields.append((i, field))
    
    if missing_fields:
        issues.append(f"Missing required metadata fields: {missing_fields}")
        logger.error(f"  ✗ Missing fields in chunks: {missing_fields}")
    else:
        logger.info(f"  ✓ All {len(chunks)} chunks have complete metadata")
    
    # Verify values match source data
    drug_id = sample_smpc_data.get("drug_id")
    version_hash = sample_smpc_data.get("version_hash")
    source_pdf = sample_smpc_data.get("source_pdf")
    
    for chunk in chunks:
        if chunk.metadata["drug_id"] != drug_id:
            issues.append(f"drug_id mismatch in chunk {chunk.metadata.get('chunk_id')}")
        if chunk.metadata["version_hash"] != version_hash:
            issues.append(f"version_hash mismatch in chunk {chunk.metadata.get('chunk_id')}")
        if chunk.metadata["pdf_path"] != source_pdf:
            issues.append(f"pdf_path mismatch in chunk {chunk.metadata.get('chunk_id')}")
    
    # Test 3: Large sections subdivided
    logger.info("\n3. Testing large sections subdivided appropriately...")
    chunk_size = 300
    sections_to_chunks: Dict[str, List[Chunk]] = {}
    for chunk in chunks:
        section_num = chunk.metadata.get("section_number", "unknown")
        if section_num not in sections_to_chunks:
            sections_to_chunks[section_num] = []
        sections_to_chunks[section_num].append(chunk)
    
    large_sections_found = []
    for section_num, section_data in sample_smpc_data.get("sections", {}).items():
        section_text = section_data.get("text", "").strip()
        if len(section_text) > chunk_size:
            large_sections_found.append(section_num)
            section_chunks = sections_to_chunks.get(section_num, [])
            
            if len(section_chunks) <= 1:
                issues.append(
                    f"Section {section_num} is {len(section_text)} chars "
                    f"(>{chunk_size}) but only has {len(section_chunks)} chunk(s)"
                )
                logger.error(
                    f"  ✗ Section {section_num} not subdivided "
                    f"({len(section_text)} chars, {len(section_chunks)} chunks)"
                )
            else:
                logger.info(
                    f"  ✓ Section {section_num} subdivided into {len(section_chunks)} chunks "
                    f"({len(section_text)} chars)"
                )
            
            # Check chunk sizes
            for chunk in section_chunks:
                if len(chunk.text) > chunk_size * 1.5:
                    issues.append(
                        f"Chunk for section {section_num} too large: "
                        f"{len(chunk.text)} chars"
                    )
    
    if not large_sections_found:
        logger.info("  ✓ No large sections found (all fit in single chunks)")
    
    # Test 4: Chunk text not empty
    logger.info("\n4. Testing chunk text not empty...")
    empty_chunks = []
    for i, chunk in enumerate(chunks):
        if not chunk.text or not chunk.text.strip():
            empty_chunks.append(i)
    
    if empty_chunks:
        issues.append(f"Found {len(empty_chunks)} empty chunks")
        logger.error(f"  ✗ Found {len(empty_chunks)} empty chunks")
    else:
        logger.info(f"  ✓ All {len(chunks)} chunks have non-empty text")
    
    # Test 5: Chunk IDs unique
    logger.info("\n5. Testing chunk IDs are unique...")
    chunk_ids = [chunk.metadata.get("chunk_id") for chunk in chunks]
    unique_ids = set(chunk_ids)
    
    if len(chunk_ids) != len(unique_ids):
        issues.append(f"Found {len(chunk_ids) - len(unique_ids)} duplicate chunk IDs")
        logger.error(f"  ✗ Found duplicate chunk IDs")
    else:
        logger.info(f"  ✓ All {len(chunks)} chunks have unique IDs")
    
    return len(issues) == 0, issues


def test_vector_store(sample_file: Path, test_store_path: Path) -> Tuple[bool, List[str]]:
    """
    Test vector store: chunks stored, retrieval by query, metadata filters working.
    
    Returns:
        Tuple of (all_passed: bool, issues: List[str])
    """
    logger.info("\n" + "=" * 60)
    logger.info("TESTING VECTOR STORE")
    logger.info("=" * 60)
    
    if not sample_file.exists():
        return False, [f"Sample file not found: {sample_file}"]
    
    with open(sample_file, "r", encoding="utf-8") as f:
        sample_smpc_data = json.load(f)
    
    # Create test vector store
    original_path = Config.VECTOR_STORE_PATH
    Config.VECTOR_STORE_PATH = test_store_path
    test_vector_store = VectorStoreManager(collection_name="test_smpc_documents")
    issues = []
    
    try:
        # Test 1: Chunks stored successfully
        logger.info("\n1. Testing chunks stored successfully...")
        chunks = chunk_smpc_json(sample_smpc_data, chunk_size=300, chunk_overlap=0)
        
        initial_count = test_vector_store.get_document_count()
        test_vector_store.add_chunks(chunks)
        final_count = test_vector_store.get_document_count()
        
        expected_count = initial_count + len(chunks)
        if final_count != expected_count:
            issues.append(
                f"Expected {expected_count} documents, got {final_count}"
            )
            logger.error(f"  ✗ Document count mismatch: {final_count} != {expected_count}")
        else:
            logger.info(
                f"  ✓ Successfully stored {len(chunks)} chunks "
                f"(total documents: {final_count})"
            )
        
        # Test 2: Retrieve chunks by query
        logger.info("\n2. Testing retrieval by query...")
        # Skip Opik instrumentation in tests to avoid Pydantic issues
        retriever = test_vector_store.vector_store.as_retriever(search_kwargs={"k": 5})
        
        test_queries = [
            "What are the indications for this medication?",
            "What is the recommended dosage?",
            "What are the side effects?",
        ]
        
        for query in test_queries:
            results = retriever.get_relevant_documents(query)
            
            if len(results) == 0:
                issues.append(f"No results returned for query: '{query}'")
                logger.error(f"  ✗ No results for: '{query[:50]}...'")
            else:
                # Verify results have content
                for doc in results:
                    if not doc.page_content:
                        issues.append(f"Retrieved document has empty content for query: '{query}'")
                    if not doc.metadata:
                        issues.append(f"Retrieved document has no metadata for query: '{query}'")
                
                logger.info(
                    f"  ✓ Query '{query[:50]}...' returned {len(results)} relevant chunks"
                )
        
        # Test 3: Metadata filter by medication_name
        logger.info("\n3. Testing metadata filter by medication_name...")
        drug_id = sample_smpc_data.get("drug_id")
        
        # Skip Opik instrumentation in tests
        retriever_filtered = test_vector_store.vector_store.as_retriever(
            search_kwargs={"k": 10, "filter": {"medication_name": drug_id}}
        )
        
        query = "What are the indications?"
        results = retriever_filtered.get_relevant_documents(query)
        
        if len(results) == 0:
            issues.append("No results returned with medication filter")
            logger.error("  ✗ No results with medication filter")
        else:
            # Verify all results match the filter
            wrong_medications = []
            for doc in results:
                metadata = doc.metadata
                medication_name = metadata.get("medication_name") or metadata.get("drug_id")
                if medication_name != drug_id:
                    wrong_medications.append(medication_name)
            
            if wrong_medications:
                issues.append(
                    f"Filter returned chunks with wrong medications: {wrong_medications}"
                )
                logger.error(f"  ✗ Wrong medications in filtered results: {wrong_medications}")
            else:
                logger.info(
                    f"  ✓ Medication filter '{drug_id}' returned {len(results)} matching chunks"
                )
        
        # Test 4: Document exists check
        logger.info("\n4. Testing document_exists() check...")
        source_pdf = sample_smpc_data.get("source_pdf", "")
        
        if source_pdf:
            # Metadata stores the full path as "source"
            if not test_vector_store.document_exists(source_pdf):
                issues.append(f"document_exists() returned False for stored document")
                logger.error(f"  ✗ document_exists() failed for '{source_pdf}'")
            else:
                logger.info(f"  ✓ document_exists() correctly identifies '{source_pdf}'")
        
        # Test 5: Get unique medications
        logger.info("\n5. Testing get_unique_medications()...")
        medications = test_vector_store.get_unique_medications()
        
        if drug_id not in medications:
            issues.append(f"Expected medication '{drug_id}' not in unique medications list")
            logger.error(f"  ✗ Medication '{drug_id}' not found in list: {medications}")
        else:
            logger.info(f"  ✓ Found {len(medications)} unique medication(s): {medications}")
        
    finally:
        # Cleanup
        test_vector_store.clear_collection()
        Config.VECTOR_STORE_PATH = original_path
    
    return len(issues) == 0, issues


def main():
    """Run all tests and report results."""
    logger.info("Starting chunking and vector store tests...")
    
    # Find sample file
    sample_file = Path("data/structured/Voriconazole_Normon_SmPC_SmPC.json")
    
    if not sample_file.exists():
        logger.error(f"Sample file not found: {sample_file}")
        logger.info("Please ensure you have processed at least one SmPC PDF.")
        return
    
    # Test chunk quality
    chunk_passed, chunk_issues = test_chunk_quality(sample_file)
    
    # Test vector store
    test_store_path = Path("data/vector_store_test")
    test_store_path.mkdir(parents=True, exist_ok=True)
    
    vector_passed, vector_issues = test_vector_store(sample_file, test_store_path)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    logger.info(f"\nChunk Quality Tests: {'PASSED' if chunk_passed else 'FAILED'}")
    if chunk_issues:
        logger.error(f"  Issues found: {len(chunk_issues)}")
        for issue in chunk_issues:
            logger.error(f"    - {issue}")
    
    logger.info(f"\nVector Store Tests: {'PASSED' if vector_passed else 'FAILED'}")
    if vector_issues:
        logger.error(f"  Issues found: {len(vector_issues)}")
        for issue in vector_issues:
            logger.error(f"    - {issue}")
    
    all_passed = chunk_passed and vector_passed
    logger.info(f"\nOverall: {'ALL TESTS PASSED ✓' if all_passed else 'SOME TESTS FAILED ✗'}")
    
    if all_passed:
        logger.info("\n✓ Your chunking and vector store implementation is working correctly!")
    else:
        logger.error("\n✗ Please review the issues above and fix them.")


if __name__ == "__main__":
    main()
