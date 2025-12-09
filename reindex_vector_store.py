#!/usr/bin/env python3
"""
Re-index vector store with new chunk settings.

This script:
1. Loads all existing JSON files from data/structured/
2. Removes old chunks from vector store
3. Re-chunks with new settings (from Config: CHUNK_SIZE, CHUNK_OVERLAP)
4. Validates that chunks never cross section boundaries (optional)
5. Adds new chunks to vector store

No PDF processing required - only uses existing JSON files.
"""
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from config import Config
from src.smpc_parser import is_valid_smpc
from src.chunker import chunk_smpc_json, Chunk
from src.vector_store import VectorStoreManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_chunk_boundaries(chunks: List[Chunk], sections: Dict[str, Any]) -> bool:
    """
    Validate that chunks never cross section boundaries.
    
    The chunker already ensures this by processing each section independently,
    but this function provides explicit verification.
    
    Args:
        chunks: List of Chunk objects to validate
        sections: Dictionary of sections from JSON data
        
    Returns:
        True if all chunks are valid, False if any violations found
    """
    violations = []
    
    for chunk in chunks:
        section_num = chunk.metadata.get("section_number")
        chunk_text = chunk.text
        
        # Check that section_number exists in sections
        if section_num not in sections:
            violations.append(
                f"Chunk {chunk.metadata.get('chunk_id')} references non-existent section {section_num}"
            )
            continue
        
        # Get the expected section text
        section_data = sections[section_num]
        section_text = section_data.get("text", "").strip()
        
        if not section_text:
            # Empty sections are skipped, so chunks shouldn't reference them
            violations.append(
                f"Chunk {chunk.metadata.get('chunk_id')} references empty section {section_num}"
            )
            continue
        
        # Verify chunk text is contained within section text
        # Normalize whitespace for comparison (chunks may have different whitespace due to overlap)
        chunk_text_normalized = " ".join(chunk_text.split())
        section_text_normalized = " ".join(section_text.split())
        
        # Check if chunk text appears in section text (allowing for overlap within section)
        # Overlap is acceptable - chunks within the same section can overlap
        if chunk_text_normalized not in section_text_normalized:
            # For overlapping chunks, check if any significant portion matches
            # (overlap means some text will be repeated)
            chunk_words = set(chunk_text_normalized.split())
            section_words = set(section_text_normalized.split())
            
            # If less than 80% of chunk words are in section, it's likely a violation
            if chunk_words and len(chunk_words & section_words) / len(chunk_words) < 0.8:
                violations.append(
                    f"Chunk {chunk.metadata.get('chunk_id')} (section {section_num}) "
                    f"contains text not found in section {section_num}"
                )
    
    if violations:
        logger.warning(f"Found {len(violations)} chunk boundary violations:")
        for violation in violations[:10]:  # Show first 10
            logger.warning(f"  - {violation}")
        if len(violations) > 10:
            logger.warning(f"  ... and {len(violations) - 10} more violations")
        return False
    
    return True


def reindex_json_file(
    json_path: Path,
    vector_store: VectorStoreManager,
    validate: bool = True
) -> Dict[str, Any]:
    """
    Re-index a JSON file in the vector store with new chunk settings.
    
    Args:
        json_path: Path to JSON file to re-index
        vector_store: VectorStoreManager instance
        validate: If True, validate chunk boundaries (optional, for performance)
        
    Returns:
        Dictionary with indexing results:
        - "success": bool
        - "drug_id": str
        - "chunks_created": int
        - "reason": str (if failed)
    """
    try:
        # Load JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        drug_id = data.get("drug_id", json_path.stem)
        source_pdf = data.get("source_pdf", "")
        sections = data.get("sections", {})
        
        # Validate structure
        if not is_valid_smpc(data):
            return {
                "success": False,
                "drug_id": drug_id,
                "chunks_created": 0,
                "reason": "Invalid SmPC structure"
            }
        
        # Remove existing chunks for this document
        # Use source_pdf filename as the document identifier
        # BUT: Skip if collection is empty (just cleared)
        if source_pdf:
            pdf_filename = Path(source_pdf).name
            try:
                # Check if collection has any documents before trying to remove
                doc_count = vector_store.get_document_count()
                if doc_count > 0:
                    removed_count = vector_store.remove_document(pdf_filename)
                    if removed_count > 0:
                        logger.info(f"Removed {removed_count} old chunks for {drug_id}")
                else:
                    logger.debug(f"Collection is empty, skipping removal for {drug_id}")
            except (StopIteration, Exception) as e:
                # Collection may be empty or not fully initialized
                logger.debug(f"Could not check/remove documents (collection may be empty): {e}")
                # Continue anyway - we'll just add new chunks
        
        # Re-chunk with new settings from Config
        chunks = chunk_smpc_json(
            data,
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        
        logger.info(f"Created {len(chunks)} chunks for {drug_id} (chunk_size={Config.CHUNK_SIZE}, overlap={Config.CHUNK_OVERLAP})")
        
        # Optional validation of chunk boundaries
        if validate:
            if not validate_chunk_boundaries(chunks, sections):
                logger.warning(f"Chunk boundary validation failed for {drug_id}, but continuing...")
        
        # Add json_path to metadata
        for chunk in chunks:
            chunk.metadata["json_path"] = str(json_path)
        
        # Add chunks to vector store
        vector_store.add_chunks(chunks)
        
        logger.info(f"Successfully re-indexed {drug_id} with {len(chunks)} chunks")
        
        return {
            "success": True,
            "drug_id": drug_id,
            "chunks_created": len(chunks)
        }
    
    except Exception as e:
        logger.error(f"Error re-indexing {json_path.name}: {e}", exc_info=True)
        return {
            "success": False,
            "drug_id": json_path.stem,
            "chunks_created": 0,
            "reason": f"Error: {str(e)}"
        }


def main(
    clear_first: bool = False,
    max_files: Optional[int] = None,
    validate: bool = True
) -> None:
    """
    Main function to re-index all JSON files with new chunk settings.
    
    Args:
        clear_first: If True, clear entire vector store before re-indexing
        max_files: Optional limit on number of files to process (for testing)
        validate: If True, validate chunk boundaries (can be disabled for performance)
    """
    logger.info("Starting vector store re-indexing with new chunk settings")
    logger.info(f"Chunk settings: size={Config.CHUNK_SIZE}, overlap={Config.CHUNK_OVERLAP}")
    
    # Ensure directories exist
    Config.ensure_directories()
    
    structured_dir = Config.STRUCTURED_DIR
    
    if not structured_dir.exists():
        logger.error(f"Structured directory does not exist: {structured_dir}")
        return
    
    # Find all JSON files
    json_files = sorted(list(structured_dir.glob("*_SmPC.json")))
    
    if not json_files:
        logger.warning(f"No JSON files found in {structured_dir}")
        return
    
    # Limit files if specified
    if max_files:
        json_files = json_files[:max_files]
        logger.info(f"Processing first {max_files} files (testing mode)")
    
    logger.info(f"Found {len(json_files)} JSON files to re-index")
    
    # Initialize vector store
    vector_store = VectorStoreManager()
    
    # Clear vector store if requested
    if clear_first:
        logger.info("Clearing existing vector store...")
        vector_store.clear_collection()
        logger.info("Vector store cleared")
    
    results = {
        "reindexed": [],
        "failed": []
    }
    
    # Re-index all files
    logger.info("=" * 60)
    logger.info("Re-indexing documents with new chunk settings")
    logger.info("=" * 60)
    
    for i, json_path in enumerate(json_files, 1):
        logger.info(f"[{i}/{len(json_files)}] Processing {json_path.name}...")
        result = reindex_json_file(json_path, vector_store, validate=validate)
        
        if result["success"]:
            results["reindexed"].append(result)
        else:
            results["failed"].append(result)
    
    # Summary
    logger.info("=" * 60)
    logger.info("Re-indexing Summary:")
    logger.info(f"  Documents re-indexed: {len(results['reindexed'])}")
    logger.info(f"  Failed: {len(results['failed'])}")
    logger.info("=" * 60)
    
    if results["reindexed"]:
        total_chunks = sum(r.get("chunks_created", 0) for r in results["reindexed"])
        logger.info(f"Successfully re-indexed documents (total chunks: {total_chunks}):")
        for result in results["reindexed"][:20]:  # Show first 20
            logger.info(f"  - {result['drug_id']}: {result.get('chunks_created', 0)} chunks")
        if len(results["reindexed"]) > 20:
            logger.info(f"  ... and {len(results['reindexed']) - 20} more documents")
    
    if results["failed"]:
        logger.error("Failed files:")
        for result in results["failed"]:
            logger.error(f"  - {result['drug_id']}: {result.get('reason', 'Unknown')}")
    
    # Final vector store stats
    doc_count = vector_store.get_document_count()
    unique_drugs = len(vector_store.all_drugs_list)
    logger.info(f"Final vector store stats:")
    logger.info(f"  Total documents (chunks): {doc_count}")
    logger.info(f"  Unique drugs: {unique_drugs}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Re-index vector store with new chunk settings using existing JSON files"
    )
    parser.add_argument(
        "--clear-first",
        action="store_true",
        help="Clear entire vector store before re-indexing (for clean re-index)"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit number of files to process (for testing)"
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip chunk boundary validation for faster processing"
    )
    
    args = parser.parse_args()
    
    main(
        clear_first=args.clear_first,
        max_files=args.max_files,
        validate=not args.no_validate
    )
