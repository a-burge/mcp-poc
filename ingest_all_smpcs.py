"""
Ingestion script for SmPC PDFs: parse, validate, chunk, and embed.

Scans /data/raw_source_docs/ for PDF files, parses them via smpc_parser,
validates SmPC structure, saves structured JSON, and indexes in vector store.
"""
import logging
import json
from pathlib import Path
from typing import Dict, Any, List

from config import Config
from src.smpc_parser import build_smpc_json, is_valid_smpc
from src.chunker import chunk_smpc_json
from src.vector_store import VectorStoreManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def scan_pdf_files(source_dir: Path) -> List[Path]:
    """
    Scan directory for PDF files (case-insensitive).
    
    Args:
        source_dir: Directory to scan
        
    Returns:
        List of PDF file paths
    """
    pdf_files = []
    if not source_dir.exists():
        logger.warning(f"Source directory does not exist: {source_dir}")
        return pdf_files
    
    for file_path in source_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() == '.pdf':
            pdf_files.append(file_path)
    
    logger.info(f"Found {len(pdf_files)} PDF files in {source_dir}")
    return pdf_files


def process_smpc_pdf(
    pdf_path: Path,
    vector_store: VectorStoreManager,
    structured_dir: Path,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Process a single SmPC PDF file.
    
    Args:
        pdf_path: Path to PDF file
        vector_store: VectorStoreManager instance
        structured_dir: Directory to save structured JSON
        dry_run: If True, don't save or index, just validate
        
    Returns:
        Dictionary with processing results:
        - "success": bool
        - "drug_id": str
        - "reason": str (if failed)
        - "chunks_created": int (if successful)
    """
    logger.info(f"Processing PDF: {pdf_path.name}")
    
    try:
        # Step 1: Parse PDF to structured JSON
        smpc_data = build_smpc_json(str(pdf_path))
        drug_id = smpc_data.get("drug_id", pdf_path.stem)
        
        # Step 2: Validate SmPC structure
        if not is_valid_smpc(smpc_data):
            logger.warning(
                f"Skipping {pdf_path.name}: Not a valid SmPC structure "
                f"(may be reminder card, leaflet, or patient instructions)"
            )
            return {
                "success": False,
                "drug_id": drug_id,
                "reason": "Invalid SmPC structure"
            }
        
        logger.info(f"Valid SmPC detected for: {drug_id}")
        
        if dry_run:
            logger.info(f"[DRY RUN] Would process {drug_id}")
            return {
                "success": True,
                "drug_id": drug_id,
                "reason": "Dry run - not processed"
            }
        
        # Step 3: Save structured JSON
        json_filename = f"{drug_id}_SmPC.json"
        json_path = structured_dir / json_filename
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(smpc_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved structured JSON to: {json_path}")
        
        # Step 4: Chunk sections
        chunks = chunk_smpc_json(
            smpc_data,
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        
        logger.info(f"Created {len(chunks)} chunks for {drug_id}")
        
        # Step 5: Add json_path to metadata
        for chunk in chunks:
            chunk.metadata["json_path"] = str(json_path)
        
        # Step 6: Embed and store in vector store
        vector_store.add_chunks(chunks)
        
        logger.info(f"Successfully indexed {drug_id} with {len(chunks)} chunks")
        
        return {
            "success": True,
            "drug_id": drug_id,
            "chunks_created": len(chunks),
            "json_path": str(json_path)
        }
    
    except Exception as e:
        logger.error(f"Error processing {pdf_path.name}: {e}", exc_info=True)
        return {
            "success": False,
            "drug_id": pdf_path.stem,
            "reason": f"Processing error: {str(e)}"
        }


def main(dry_run: bool = False, clear_existing: bool = False) -> None:
    """
    Main ingestion function.
    
    Args:
        dry_run: If True, validate but don't save or index
        clear_existing: If True, clear vector store before ingestion
    """
    logger.info("Starting SmPC ingestion process")
    
    # Ensure directories exist
    Config.ensure_directories()
    
    source_dir = Config.RAW_SOURCE_DOCS_DIR
    structured_dir = Config.STRUCTURED_DIR
    
    logger.info(f"Source directory: {source_dir}")
    logger.info(f"Structured JSON directory: {structured_dir}")
    
    # Initialize vector store
    vector_store = VectorStoreManager()
    
    if clear_existing:
        logger.warning("Clearing existing vector store...")
        vector_store.clear_collection()
    
    # Scan for PDF files
    pdf_files = scan_pdf_files(source_dir)
    
    if not pdf_files:
        logger.warning("No PDF files found to process")
        return
    
    # Process each PDF
    results = {
        "successful": [],
        "failed": [],
        "skipped": []
    }
    
    for pdf_path in pdf_files:
        result = process_smpc_pdf(
            pdf_path,
            vector_store,
            structured_dir,
            dry_run=dry_run
        )
        
        if result["success"]:
            results["successful"].append(result)
        elif result.get("reason") == "Invalid SmPC structure":
            results["skipped"].append(result)
        else:
            results["failed"].append(result)
    
    # Summary
    logger.info("=" * 60)
    logger.info("Ingestion Summary:")
    logger.info(f"  Successful: {len(results['successful'])}")
    logger.info(f"  Skipped (not SmPC): {len(results['skipped'])}")
    logger.info(f"  Failed: {len(results['failed'])}")
    logger.info("=" * 60)
    
    if results["successful"]:
        logger.info("Successfully processed drugs:")
        for result in results["successful"]:
            logger.info(f"  - {result['drug_id']}: {result.get('chunks_created', 0)} chunks")
    
    if results["skipped"]:
        logger.info("Skipped files (not valid SmPC):")
        for result in results["skipped"]:
            logger.info(f"  - {result['drug_id']}: {result.get('reason', 'Unknown')}")
    
    if results["failed"]:
        logger.error("Failed files:")
        for result in results["failed"]:
            logger.error(f"  - {result['drug_id']}: {result.get('reason', 'Unknown')}")
    
    # Final vector store stats
    doc_count = vector_store.get_document_count()
    logger.info(f"Total documents in vector store: {doc_count}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Ingest SmPC PDFs: parse, validate, chunk, and embed"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate but don't save or index"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear vector store before ingestion"
    )
    
    args = parser.parse_args()
    
    main(dry_run=args.dry_run, clear_existing=args.clear)
