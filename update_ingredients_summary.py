#!/usr/bin/env python3
"""
Update all existing JSON files with ingredients_summary section and re-index in vector store.

This script:
1. Loads all existing JSON files from data/structured/
2. Adds ingredients_summary section to each (if sections 2 and 6.1 exist)
3. Re-saves the updated JSON files
4. Removes old chunks from vector store
5. Re-chunks and re-indexes updated documents
"""
import logging
import json
from pathlib import Path
from typing import Dict, Any, List

from config import Config
from src.smpc_parser import create_ingredients_summary, is_valid_smpc
from src.chunker import chunk_smpc_json
from src.vector_store import VectorStoreManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def update_json_file(json_path: Path) -> Dict[str, Any]:
    """
    Update a single JSON file with ingredients_summary section.
    
    Args:
        json_path: Path to JSON file to update
        
    Returns:
        Dictionary with update results:
        - "success": bool
        - "drug_id": str
        - "updated": bool (whether ingredients_summary was added/updated)
        - "reason": str (if failed)
    """
    try:
        # Load existing JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        drug_id = data.get("drug_id", json_path.stem)
        
        # Check if already has ingredients_summary
        sections = data.get("sections", {})
        has_existing = "ingredients_summary" in sections
        
        # Create ingredients summary
        ingredients_summary = create_ingredients_summary(sections)
        
        if not ingredients_summary:
            return {
                "success": True,
                "drug_id": drug_id,
                "updated": False,
                "reason": "No ingredients_summary created (missing section 2 or 6.1)"
            }
        
        # Add or update ingredients_summary section
        sections["ingredients_summary"] = ingredients_summary
        data["sections"] = sections
        
        # Save updated JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(
            f"{'Updated' if has_existing else 'Added'} ingredients_summary for {drug_id}"
        )
        
        return {
            "success": True,
            "drug_id": drug_id,
            "updated": True,
            "was_new": not has_existing
        }
    
    except Exception as e:
        logger.error(f"Error updating {json_path.name}: {e}", exc_info=True)
        return {
            "success": False,
            "drug_id": json_path.stem,
            "updated": False,
            "reason": f"Error: {str(e)}"
        }


def reindex_json_file(
    json_path: Path,
    vector_store: VectorStoreManager
) -> Dict[str, Any]:
    """
    Re-index a JSON file in the vector store.
    
    Args:
        json_path: Path to JSON file to re-index
        vector_store: VectorStoreManager instance
        
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
        if source_pdf:
            pdf_filename = Path(source_pdf).name
            removed_count = vector_store.remove_document(pdf_filename)
            if removed_count > 0:
                logger.info(f"Removed {removed_count} old chunks for {drug_id}")
        
        # Chunk updated JSON
        chunks = chunk_smpc_json(
            data,
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        
        logger.info(f"Created {len(chunks)} chunks for {drug_id}")
        
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
    update_only: bool = False,
    reindex_only: bool = False,
    dry_run: bool = False
) -> None:
    """
    Main function to update all JSON files with ingredients_summary.
    
    Args:
        update_only: If True, only update JSON files, don't re-index
        reindex_only: If True, only re-index, don't update JSON files
        dry_run: If True, show what would be done but don't make changes
    """
    logger.info("Starting ingredients_summary update process")
    
    # Ensure directories exist
    Config.ensure_directories()
    
    structured_dir = Config.STRUCTURED_DIR
    
    if not structured_dir.exists():
        logger.error(f"Structured directory does not exist: {structured_dir}")
        return
    
    # Find all JSON files
    json_files = list(structured_dir.glob("*_SmPC.json"))
    
    if not json_files:
        logger.warning(f"No JSON files found in {structured_dir}")
        return
    
    logger.info(f"Found {len(json_files)} JSON files to process")
    
    if dry_run:
        logger.info("[DRY RUN] Would update the following files:")
        for json_file in json_files:
            logger.info(f"  - {json_file.name}")
        return
    
    # Initialize vector store (only if we need to re-index)
    vector_store = None
    if not update_only:
        vector_store = VectorStoreManager()
    
    results = {
        "updated": [],
        "reindexed": [],
        "failed": []
    }
    
    # Step 1: Update JSON files
    if not reindex_only:
        logger.info("=" * 60)
        logger.info("Step 1: Updating JSON files with ingredients_summary")
        logger.info("=" * 60)
        
        for json_path in json_files:
            result = update_json_file(json_path)
            
            if result["success"]:
                if result.get("updated"):
                    results["updated"].append(result)
            else:
                results["failed"].append(result)
    
    # Step 2: Re-index in vector store
    if not update_only:
        logger.info("=" * 60)
        logger.info("Step 2: Re-indexing updated documents in vector store")
        logger.info("=" * 60)
        
        for json_path in json_files:
            result = reindex_json_file(json_path, vector_store)
            
            if result["success"]:
                results["reindexed"].append(result)
            else:
                results["failed"].append(result)
    
    # Summary
    logger.info("=" * 60)
    logger.info("Update Summary:")
    if not reindex_only:
        logger.info(f"  JSON files updated: {len(results['updated'])}")
    if not update_only:
        logger.info(f"  Documents re-indexed: {len(results['reindexed'])}")
    logger.info(f"  Failed: {len(results['failed'])}")
    logger.info("=" * 60)
    
    if results["updated"]:
        logger.info("Successfully updated files:")
        for result in results["updated"]:
            status = "NEW" if result.get("was_new") else "UPDATED"
            logger.info(f"  - {result['drug_id']}: {status}")
    
    if results["reindexed"]:
        logger.info("Successfully re-indexed documents:")
        for result in results["reindexed"]:
            logger.info(f"  - {result['drug_id']}: {result.get('chunks_created', 0)} chunks")
    
    if results["failed"]:
        logger.error("Failed files:")
        for result in results["failed"]:
            logger.error(f"  - {result['drug_id']}: {result.get('reason', 'Unknown')}")
    
    # Final vector store stats
    if vector_store:
        doc_count = vector_store.get_document_count()
        logger.info(f"Total documents in vector store: {doc_count}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Update all JSON files with ingredients_summary and re-index in vector store"
    )
    parser.add_argument(
        "--update-only",
        action="store_true",
        help="Only update JSON files, don't re-index in vector store"
    )
    parser.add_argument(
        "--reindex-only",
        action="store_true",
        help="Only re-index, don't update JSON files (assumes they're already updated)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done but don't make changes"
    )
    
    args = parser.parse_args()
    
    if args.update_only and args.reindex_only:
        parser.error("Cannot specify both --update-only and --reindex-only")
    
    main(
        update_only=args.update_only,
        reindex_only=args.reindex_only,
        dry_run=args.dry_run
    )
