"""Test Mistral OCR extraction on first 10 files from sample_50_results.txt."""
import json
import re
import logging
import time
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import fitz  # PyMuPDF

from config import Config
from src.smpc_extractor_mistral import extract_with_mistral_ocr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_filenames_from_sample_file(sample_file: Path, num_files: int = 10) -> List[str]:
    """Extract first N filenames from sample_50_results.txt."""
    filenames = []
    with open(sample_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('Filename: '):
                filename = line.replace('Filename: ', '').strip()
                filenames.append(filename)
                if len(filenames) >= num_files:
                    break
    return filenames


def find_pdf_file(filename: str, search_dir: Path) -> Path:
    """Find PDF file in directory tree (case-insensitive)."""
    # Try exact match first
    exact_path = search_dir / filename
    if exact_path.exists():
        return exact_path
    
    # Try case-insensitive search
    filename_lower = filename.lower()
    for pdf_path in search_dir.rglob('*.pdf'):
        if pdf_path.name.lower() == filename_lower:
            return pdf_path
    
    # Try partial match
    for pdf_path in search_dir.rglob('*.pdf'):
        if filename_lower in pdf_path.name.lower() or pdf_path.name.lower() in filename_lower:
            return pdf_path
    
    raise FileNotFoundError(f"Could not find file: {filename} in {search_dir}")


def get_pdf_page_count(pdf_path: Path) -> int:
    """Get the number of pages in a PDF file."""
    try:
        with fitz.open(pdf_path) as doc:
            return len(doc)
    except Exception as e:
        logger.error(f"Error reading PDF {pdf_path}: {e}")
        raise


def test_mistral_ocr_on_file(pdf_path: Path, output_dir: Path) -> Dict[str, Any]:
    """Test Mistral OCR on a single file and save detailed output."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing: {pdf_path.name}")
    logger.info(f"{'='*80}")
    
    result = {
        "filename": pdf_path.name,
        "file_path": str(pdf_path),
        "success": False,
        "error": None,
        "sections_count": 0,
        "sections": {},
        "validation_report": {},
        "extraction_time": None
    }
    
    try:
        start_time = time.time()
        
        # Extract with Mistral OCR
        smpc_data = extract_with_mistral_ocr(str(pdf_path))
        
        elapsed_time = time.time() - start_time
        result["extraction_time"] = elapsed_time
        result["success"] = True
        result["sections_count"] = len(smpc_data.get("sections", {}))
        result["sections"] = smpc_data.get("sections", {})
        result["validation_report"] = smpc_data.get("validation_report", {})
        
        # Save full output to JSON file
        output_filename = pdf_path.stem + "_mistral_ocr_output.json"
        output_path = output_dir / output_filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(smpc_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✓ Success: Extracted {result['sections_count']} sections in {elapsed_time:.2f}s")
        logger.info(f"  Saved to: {output_path}")
        
        # Print section summary
        logger.info(f"\nSection Summary:")
        for section_num in sorted(result["sections"].keys(), key=lambda x: (
            [int(part) for part in x.split('.') if part.isdigit()] if any(part.isdigit() for part in x.split('.')) else [999]
        )):
            section = result["sections"][section_num]
            title = section.get("title", "")[:50]
            text_preview = section.get("text", "")[:100].replace('\n', ' ')
            logger.info(f"  {section_num}: {title}")
            logger.info(f"    Text preview: {text_preview}...")
        
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"✗ Failed: {e}", exc_info=True)
    
    return result


def main():
    """Main test function."""
    # Setup paths
    project_root = Path(__file__).parent
    sample_file = project_root / "sample_50_results.txt"
    source_dir = Config.RAW_SOURCE_DOCS_DIR
    output_dir = project_root / "test_outputs" / "mistral_ocr"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Project root: {project_root}")
    logger.info(f"Source directory: {source_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Extract first 10 filenames
    logger.info(f"\nExtracting first 10 filenames from {sample_file.name}...")
    all_filenames = extract_filenames_from_sample_file(sample_file, num_files=10)
    
    # Filter to only files containing "smpc" (case-insensitive)
    smpc_filenames = [f for f in all_filenames if "smpc" in f.lower()]
    logger.info(f"Found {len(all_filenames)} total filenames, {len(smpc_filenames)} containing 'smpc' (case-insensitive)")
    
    if not smpc_filenames:
        logger.warning("No files containing 'smpc' found in the first 10 files. Exiting.")
        return
    
    # Filter to only files with 8 pages or fewer
    filenames_with_paths = []
    for filename in smpc_filenames:
        try:
            pdf_path = find_pdf_file(filename, source_dir)
            page_count = get_pdf_page_count(pdf_path)
            if page_count <= 8:
                filenames_with_paths.append((filename, pdf_path, page_count))
                logger.info(f"  ✓ {filename}: {page_count} pages")
            else:
                logger.info(f"  ✗ {filename}: {page_count} pages (skipped, >8 pages)")
        except (FileNotFoundError, Exception) as e:
            logger.warning(f"  ⚠ {filename}: Could not check page count - {e}")
    
    logger.info(f"\nFound {len(filenames_with_paths)} files with 8 pages or fewer:")
    for i, (filename, _, page_count) in enumerate(filenames_with_paths, 1):
        logger.info(f"  {i}. {filename} ({page_count} pages)")
    
    if not filenames_with_paths:
        logger.warning("No files with 8 pages or fewer found. Exiting.")
        return
    
    # Test each file
    results = []
    for i, (filename, pdf_path, page_count) in enumerate(filenames_with_paths, 1):
        logger.info(f"\n\n{'#'*80}")
        logger.info(f"Processing file {i}/{len(filenames_with_paths)}: {filename} ({page_count} pages)")
        logger.info(f"{'#'*80}")
        
        try:
            # Test Mistral OCR
            result = test_mistral_ocr_on_file(pdf_path, output_dir)
            results.append(result)
            
        except Exception as e:
            logger.error(f"✗ Unexpected error: {e}", exc_info=True)
            results.append({
                "filename": filename,
                "success": False,
                "error": str(e)
            })
    
    # Save summary
    summary = {
        "test_date": datetime.now().isoformat(),
        "total_files": len(filenames_with_paths),
        "successful": sum(1 for r in results if r.get("success")),
        "failed": sum(1 for r in results if not r.get("success")),
        "results": results
    }
    
    summary_path = output_dir / "test_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # Print final summary
    logger.info(f"\n\n{'='*80}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Total files tested: {summary['total_files']}")
    logger.info(f"Successful: {summary['successful']}")
    logger.info(f"Failed: {summary['failed']}")
    logger.info(f"\nDetailed results saved to: {summary_path}")
    logger.info(f"Individual outputs saved to: {output_dir}")
    
    # Print per-file summary
    logger.info(f"\nPer-file results:")
    for result in results:
        status = "✓" if result.get("success") else "✗"
        filename = result.get("filename", "Unknown")
        if result.get("success"):
            sections = result.get("sections_count", 0)
            time_taken = result.get("extraction_time", 0)
            logger.info(f"  {status} {filename}: {sections} sections in {time_taken:.2f}s")
        else:
            error = result.get("error", "Unknown error")
            logger.info(f"  {status} {filename}: {error}")


if __name__ == "__main__":
    main()