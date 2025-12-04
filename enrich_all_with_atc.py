"""Batch script to enrich all structured JSON files with ATC codes and re-index vector store."""
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

from enrich_with_atc import enrich_json_file, update_drug_mappings
from src.atc_matcher import ATCMatcher
from src.chunker import chunk_smpc_json
from src.vector_store import VectorStoreManager
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def reindex_json_file(
    json_path: Path,
    vector_store: VectorStoreManager,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Re-index a JSON file in the vector store with ATC metadata.
    
    Args:
        json_path: Path to structured JSON file
        vector_store: VectorStoreManager instance
        dry_run: If True, don't actually re-index
        
    Returns:
        Dictionary with re-indexing results
    """
    try:
        # Load JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        drug_id = data.get("drug_id", json_path.stem)
        source_pdf = data.get("source_pdf", "")
        
        # Remove existing chunks for this document
        if source_pdf:
            pdf_filename = Path(source_pdf).name
            removed_count = vector_store.remove_document(pdf_filename)
            if removed_count > 0:
                logger.info(f"Removed {removed_count} old chunks for {drug_id}")
        
        # Re-chunk with ATC codes included
        chunks = chunk_smpc_json(
            data,
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        
        logger.info(f"Created {len(chunks)} chunks for {drug_id} (with ATC codes)")
        
        # Add to vector store if not dry run
        if not dry_run:
            vector_store.add_chunks(chunks)
            logger.info(f"Re-indexed {drug_id} with {len(chunks)} chunks")
        
        return {
            "success": True,
            "drug_id": drug_id,
            "chunks_created": len(chunks),
            "has_atc_codes": bool(data.get("atc_codes"))
        }
        
    except Exception as e:
        logger.error(f"Error re-indexing {json_path}: {e}", exc_info=True)
        return {
            "success": False,
            "drug_id": json_path.stem,
            "reason": str(e)
        }


def main(
    dry_run: bool = False,
    enrich_only: bool = False,
    reindex_only: bool = False
) -> None:
    """
    Main function to enrich all JSON files and re-index vector store.
    
    Args:
        dry_run: If True, don't save changes
        enrich_only: If True, only enrich JSON files, don't re-index
        reindex_only: If True, only re-index, don't enrich
    """
    Config.ensure_directories()
    
    structured_dir = Config.STRUCTURED_DIR
    
    if not structured_dir.exists():
        logger.error(f"Structured directory not found: {structured_dir}")
        return
    
    json_files = sorted(list(structured_dir.glob("*_SmPC.json")))
    
    if not json_files:
        logger.warning(f"No JSON files found in {structured_dir}")
        return
    
    logger.info(f"Found {len(json_files)} JSON files to process")
    
    # Step 1: Enrich with ATC codes (if not reindex_only)
    enrichment_results = []
    if not reindex_only:
        logger.info("=" * 60)
        logger.info("Step 1: Enriching JSON files with ATC codes")
        logger.info("=" * 60)
        
        matcher = ATCMatcher()
        
        for i, json_file in enumerate(json_files, 1):
            logger.info(f"[{i}/{len(json_files)}] Enriching {json_file.name}...")
            result = enrich_json_file(json_file, matcher, dry_run=dry_run)
            enrichment_results.append(result)
        
        # Update drug mappings file
        if not dry_run:
            update_drug_mappings(enrichment_results)
        
        # Summary
        successful = sum(1 for r in enrichment_results if r.get("success") and not r.get("skipped"))
        skipped = sum(1 for r in enrichment_results if r.get("skipped"))
        failed = len(enrichment_results) - successful - skipped
        
        print(f"\nEnrichment summary:")
        print(f"  Total files: {len(enrichment_results)}")
        print(f"  Successful: {successful}")
        print(f"  Skipped (already enriched): {skipped}")
        print(f"  Failed: {failed}")
    
    # Step 2: Re-index vector store (if not enrich_only)
    if not enrich_only:
        logger.info("=" * 60)
        logger.info("Step 2: Re-indexing vector store with ATC metadata")
        logger.info("=" * 60)
        
        vector_store = VectorStoreManager()
        
        reindex_results = []
        for i, json_file in enumerate(json_files, 1):
            logger.info(f"[{i}/{len(json_files)}] Re-indexing {json_file.name}...")
            result = reindex_json_file(json_file, vector_store, dry_run=dry_run)
            reindex_results.append(result)
        
        # Summary
        successful = sum(1 for r in reindex_results if r.get("success"))
        failed = len(reindex_results) - successful
        total_chunks = sum(r.get("chunks_created", 0) for r in reindex_results)
        with_atc = sum(1 for r in reindex_results if r.get("has_atc_codes"))
        
        print(f"\nRe-indexing summary:")
        print(f"  Total files: {len(reindex_results)}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Total chunks created: {total_chunks}")
        print(f"  Files with ATC codes: {with_atc}")
    
    logger.info("Batch enrichment and re-indexing complete!")


if __name__ == "__main__":
    import sys
    
    dry_run = "--dry-run" in sys.argv
    enrich_only = "--enrich-only" in sys.argv
    reindex_only = "--reindex-only" in sys.argv
    
    main(dry_run=dry_run, enrich_only=enrich_only, reindex_only=reindex_only)
