"""
Mistral OCR extractor for SmPC PDFs.

Extracts structured sections from SmPC PDFs using Mistral OCR API,
then converts the output to match the existing build_smpc_json() format.

Note: Mistral OCR document annotations have an 8-page limit per request.
Documents exceeding 8 pages are automatically split into chunks and processed separately,
then merged back together.
"""
import base64
import hashlib
import datetime
import logging
import time
import re
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import fitz  # PyMuPDF

from pydantic import BaseModel, Field
from mistralai import Mistral
from mistralai.extra import response_format_from_pydantic_model

from config import Config
from src.smpc_parser import canonical_key_for, file_md5

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# Mistral OCR limits
MAX_PAGES_PER_ANNOTATION = 8  # Document annotations are limited to 8 pages


# Pydantic models for structured extraction
class Section(BaseModel):
    """A single section from an SmPC document."""
    number: str = Field(..., description="Section number (e.g., '1', '4.3', '6.1')")
    heading: str = Field(
        ...,
        description=(
            "Full heading text including number (e.g., '4.3 Frábendingar'). "
            "CRITICAL: This is Icelandic text. Ensure accurate recognition of Icelandic characters: "
            "þ (thorn), Ð (eth), Æ (ash), ö, ý, í. Do NOT confuse: p/þ, b/þ, f/g, ó/ö, y/ý, i/í, E/Æ, D/Ð."
        )
    )
    title: str = Field(
        ...,
        description=(
            "Section title without number (e.g., 'Frábendingar'). "
            "CRITICAL: This is Icelandic text. Ensure accurate recognition of Icelandic characters: "
            "þ (thorn), Ð (eth), Æ (ash), ö, ý, í. Do NOT confuse: p/þ, b/þ, f/g, ó/ö, y/ý, i/í, E/Æ, D/Ð."
        )
    )
    text: str = Field(
        ...,
        description=(
            "Full text content of the section. "
            "CRITICAL: This is Icelandic text. Ensure accurate recognition of Icelandic characters: "
            "þ (thorn), Ð (eth), Æ (ash), ö, ý, í. Do NOT confuse: p/þ, b/þ, f/g, ó/ö, y/ý, i/í, E/Æ, D/Ð."
        )
    )


class DocumentAnnotation(BaseModel):
    """Document annotation containing all sections from an SmPC document.
    
    Extract all numbered sections from this Icelandic SmPC (Summary of Product Characteristics) document.
    Sections are typically numbered like:
    - Top-level: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    - Subsections: 4.1, 4.2, 4.3, etc.
    
    Do NOT include false positives like '1 mg' (dosage) or '1 ár' (duration) as sections.
    Only extract actual section headings that follow the pattern: number followed by a meaningful title.
    If you see very large blocks of text, look closer to see if you have missed section headings.

    IMPORTANT: This document is in Icelandic. Pay special attention to accurate recognition of 
    Icelandic-specific characters: þ (thorn), Ð (eth), Æ (ash), ö, ý, í. Common OCR errors to avoid:
    - p or b instead of þ (thorn)
    - f instead of g
    - ó instead of ö
    - y instead of ý
    - i instead of í
    - E or / instead of Æ
    - D instead of Ð
    """
    sections: List[Section] = Field(
        ...,
        description=(
            "List of all sections found in the document. "
            "Ensure all Icelandic characters are accurately recognized."
        )
    )


def correct_icelandic_characters(text: str) -> str:
    """
    Post-process text to correct common Icelandic character misrepresentations from OCR.
    
    This function applies systematic corrections based on documented OCR errors:
    - p/b → þ (thorn) in specific contexts
    - f → g in specific contexts
    - ó → ö in specific contexts
    - y → ý in specific contexts
    - i → í in specific contexts
    - E or / → Æ in specific contexts
    - D → Ð in specific contexts
    
    Args:
        text: Text that may contain OCR character errors
        
    Returns:
        Text with corrected Icelandic characters
    """
    if not text:
        return text
    
    corrected = text
    
    # Word-level corrections (most reliable)
    word_corrections = {
        # Common words with þ errors
        r'\bpekkta\b': 'þekkta',
        r'\bpekkt\b': 'þekkt',
        r'\blíkamspunga\b': 'líkamsþunga',
        r'\bfrúktósaópol\b': 'frúktósaóþol',
        r'\bGeymslupól\b': 'Geymsluþol',
        r'\bhámarkspéttni\b': 'hámarksþéttni',
        r'\bbéttni\b': 'þéttni',
        r'\bbekkt\b': 'þekkt',
        # Common words with g errors
        r'\blyfjagjóf\b': 'lyfjagjöf',
        # Common words with ö errors
        r'\bórsjaldan\b': 'örsjaldan',
        r'\boll\b': 'öll',  # Context-dependent, but common
        # Common words with ý errors
        r'\bFenoximetyl\b': 'Fenoxýmetýl',
        r'\bsíruþolið\b': 'sýruþolið',
        # Common words with í errors
        r'\bpenicillini\b': 'penicillíni',
        r'\bklínisk\b': 'klínísk',
        # Common words with Æ errors
        r'\bLYFJAFR/ÉDILEGAR\b': 'LYFJAFRÆÐILEGAR',
        r'\bLYFJAFRÉDILEGAR\b': 'LYFJAFRÆÐILEGAR',
        r'\bLYFJAGERDARFR/ÉDILEGAR\b': 'LYFJAGERÐARFRÆÐILEGAR',
        r'\bLYFJAGERDARFRÉDILEGAR\b': 'LYFJAGERÐARFRÆÐILEGAR',
        # Common words with Ð errors
        r'\bLYFJAGERDAR\b': 'LYFJAGERÐAR',
        # Other common word errors
        r'\bvarðaðarorð\b': 'varúðarorð',
        r'\bsæsinna\b': 'svæsinna',
        r'\bmeltingarópægingar\b': 'meltingaróþægindi',
        r'\bPynd\b': 'Þyngd',
        r'\bÞáðmiskerfi\b': 'Ónæmiskerfi',
        r'\bsíklalyfja\b': 'sýklalyfja',
    }
    
    for pattern, replacement in word_corrections.items():
        corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
    
    # Character-level corrections (more aggressive, context-dependent)
    # These are applied more carefully to avoid false positives
    
    # p → þ corrections (only in specific contexts where p doesn't make sense in Icelandic)
    # We'll be conservative here and only fix known patterns
    corrected = re.sub(r'\b([a-záéíóúýæö])p([a-záéíóúýæö]+)\b', 
                      lambda m: m.group(1) + 'þ' + m.group(2) if any(
                          word in m.group(0).lower() for word in ['þekk', 'þétt', 'þol', 'þung']
                      ) else m.group(0), corrected, flags=re.IGNORECASE)
    
    # b → þ corrections (only in specific contexts)
    corrected = re.sub(r'\b([a-záéíóúýæö])b([a-záéíóúýæö]+)\b',
                      lambda m: m.group(1) + 'þ' + m.group(2) if any(
                          word in m.group(0).lower() for word in ['þekk', 'þétt']
                      ) else m.group(0), corrected, flags=re.IGNORECASE)
    
    return corrected


def get_pdf_page_count(pdf_path: str) -> int:
    """
    Get the number of pages in a PDF file.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Number of pages in the PDF
    """
    try:
        with fitz.open(pdf_path) as doc:
            return len(doc)
    except Exception as e:
        logger.error(f"Error reading PDF {pdf_path}: {e}")
        raise


def split_pdf_into_chunks(
    pdf_path: str,
    max_pages: int = MAX_PAGES_PER_ANNOTATION
) -> List[Tuple[str, int, int]]:
    """
    Split a PDF into chunks of at most max_pages each.
    
    Args:
        pdf_path: Path to the source PDF file
        max_pages: Maximum number of pages per chunk (default: 8)
        
    Returns:
        List of tuples: (chunk_pdf_path, start_page, end_page) for each chunk.
        Chunk PDFs are temporary files that should be cleaned up after use.
    """
    pdf_path_obj = Path(pdf_path)
    total_pages = get_pdf_page_count(str(pdf_path_obj))
    
    if total_pages <= max_pages:
        # No splitting needed
        return [(str(pdf_path_obj), 0, total_pages - 1)]
    
    logger.info(
        f"Splitting PDF {pdf_path_obj.name} into chunks "
        f"({total_pages} pages, {max_pages} pages per chunk)"
    )
    
    chunks = []
    temp_dir = tempfile.mkdtemp(prefix="mistral_ocr_chunks_")
    
    try:
        with fitz.open(pdf_path) as source_doc:
            chunk_num = 0
            for start_page in range(0, total_pages, max_pages):
                end_page = min(start_page + max_pages - 1, total_pages - 1)
                
                # Create new PDF for this chunk
                chunk_doc = fitz.open()
                chunk_doc.insert_pdf(source_doc, from_page=start_page, to_page=end_page)
                
                # Save to temporary file
                chunk_filename = f"{pdf_path_obj.stem}_chunk_{chunk_num:03d}.pdf"
                chunk_path = Path(temp_dir) / chunk_filename
                chunk_doc.save(str(chunk_path))
                chunk_doc.close()
                
                chunks.append((str(chunk_path), start_page, end_page))
                logger.debug(
                    f"Created chunk {chunk_num}: pages {start_page}-{end_page} "
                    f"({end_page - start_page + 1} pages)"
                )
                chunk_num += 1
        
        logger.info(f"Split PDF into {len(chunks)} chunks")
        return chunks
        
    except Exception as e:
        logger.error(f"Error splitting PDF {pdf_path}: {e}", exc_info=True)
        # Clean up any created chunks
        for chunk_path, _, _ in chunks:
            try:
                Path(chunk_path).unlink(missing_ok=True)
            except Exception:
                pass
        raise


def encode_pdf_base64(pdf_path: str) -> str:
    """
    Encode PDF file to base64 string.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Base64-encoded string of PDF content
    """
    with open(pdf_path, "rb") as pdf_file:
        return base64.b64encode(pdf_file.read()).decode('utf-8')


def process_single_chunk(
    client: Mistral,
    chunk_path: str,
    start_page: int,
    end_page: int,
    document_annotation_format: Any
) -> Dict[str, Any]:
    """
    Process a single PDF chunk with Mistral OCR API.
    
    Args:
        client: Initialized Mistral client
        chunk_path: Path to the PDF chunk file
        start_page: Starting page number in original document (0-indexed)
        end_page: Ending page number in original document (0-indexed)
        document_annotation_format: Pydantic model format for document annotation
        
    Returns:
        Dictionary with sections extracted from this chunk
        
    Raises:
        RuntimeError: If API call fails after retries
    """
    # Encode chunk PDF to base64
    pdf_base64 = encode_pdf_base64(chunk_path)
    logger.debug(
        f"Encoded chunk (pages {start_page}-{end_page}) to base64 "
        f"({len(pdf_base64)} characters)"
    )
    
    # Retry logic for API calls
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.debug(
                f"Processing chunk pages {start_page}-{end_page} "
                f"(attempt {attempt}/{MAX_RETRIES})..."
            )
            start_time = time.time()
            
            ocr_response = client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "document_url",
                    "document_url": f"data:application/pdf;base64,{pdf_base64}"
                },
                document_annotation_format=document_annotation_format
            )
            
            elapsed_time = time.time() - start_time
            logger.debug(
                f"Chunk pages {start_page}-{end_page} processed in {elapsed_time:.2f}s"
            )
            
            # Extract sections from response
            chunk_sections = extract_sections_from_response(ocr_response)
            return chunk_sections
            
        except Exception as e:
            last_error = e
            logger.warning(
                f"Mistral OCR API call failed for chunk pages {start_page}-{end_page} "
                f"(attempt {attempt}/{MAX_RETRIES}): {e}"
            )
            
            # Check if it's a rate limit error (429) or server error (5xx)
            should_retry = False
            if hasattr(e, 'status_code'):
                if e.status_code == 429:  # Rate limit
                    should_retry = True
                    retry_delay = RETRY_DELAY * attempt  # Exponential backoff
                    logger.info(f"Rate limit hit, waiting {retry_delay}s before retry...")
                    time.sleep(retry_delay)
                elif 500 <= e.status_code < 600:  # Server error
                    should_retry = True
                    logger.info(f"Server error, waiting {RETRY_DELAY}s before retry...")
                    time.sleep(RETRY_DELAY)
            
            if not should_retry or attempt == MAX_RETRIES:
                break
    
    # All retries exhausted
    raise RuntimeError(
        f"Mistral OCR extraction failed for chunk pages {start_page}-{end_page} "
        f"after {MAX_RETRIES} attempts: {str(last_error)}"
    ) from last_error


def extract_sections_from_response(mistral_response: Any) -> Dict[str, Dict[str, Any]]:
    """
    Extract sections dictionary from Mistral OCR response.
    
    Args:
        mistral_response: Response from Mistral OCR API
        
    Returns:
        Dictionary mapping section numbers to section data
    """
    sections_data = {}
    
    # Access document_annotation from response
    if hasattr(mistral_response, 'document_annotation'):
        response_content = mistral_response.document_annotation
    elif hasattr(mistral_response, 'data'):
        response_content = mistral_response.data
    else:
        # Fallback: try to use response directly
        response_content = mistral_response
    
    # Parse JSON if it's a string
    if isinstance(response_content, str):
        import json
        try:
            response_content = json.loads(response_content)
        except json.JSONDecodeError:
            logger.warning("Could not parse Mistral response as JSON, using raw text")
            response_content = {"sections": []}
    
    # Extract sections array
    if isinstance(response_content, dict):
        sections_list = response_content.get("sections", [])
    elif isinstance(response_content, list):
        sections_list = response_content
    else:
        logger.warning(f"Unexpected Mistral response format: {type(response_content)}")
        sections_list = []
    
    # Build sections dictionary
    for section_data in sections_list:
        if not isinstance(section_data, dict):
            continue
            
        number = section_data.get("number", "")
        heading = section_data.get("heading", "")
        title = section_data.get("title", "")
        text = section_data.get("text", "")
        
        if not number:
            continue
        
        # Apply Icelandic character corrections
        heading = correct_icelandic_characters(heading)
        title = correct_icelandic_characters(title)
        text = correct_icelandic_characters(text)
        
        # Determine parent section
        parent = None
        if "." in number:
            parent = number.rsplit(".", 1)[0]
        elif number in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]:
            # Top-level section, no parent
            parent = None
        else:
            # Try to extract parent from number
            parts = number.split(".")
            if len(parts) > 1:
                parent = ".".join(parts[:-1])
        
        # Get canonical key
        canonical_key = canonical_key_for(number)
        
        # Build section entry
        sections_data[number] = {
            "number": number,
            "parent": parent,
            "heading": heading or f"{number} {title}".strip(),
            "title": title or heading.replace(number, "").strip(),
            "canonical_key": canonical_key,
            "text": text,
            "children": []  # Will be populated during merge
        }
    
    return sections_data


def merge_chunk_results(
    all_chunk_sections: List[Dict[str, Dict[str, Any]]]
) -> Dict[str, Dict[str, Any]]:
    """
    Merge sections from multiple chunks, handling duplicates and combining text.
    
    When the same section appears in multiple chunks (e.g., a section spans
    across chunk boundaries), we merge the text content.
    
    Args:
        all_chunk_sections: List of section dictionaries from each chunk
        
    Returns:
        Merged sections dictionary
    """
    merged_sections = {}
    
    for chunk_sections in all_chunk_sections:
        for section_num, section_data in chunk_sections.items():
            if section_num not in merged_sections:
                # First occurrence of this section
                merged_sections[section_num] = section_data.copy()
            else:
                # Section already exists - merge text content
                existing_text = merged_sections[section_num].get("text", "")
                new_text = section_data.get("text", "")
                
                # Combine texts, avoiding duplicates
                if new_text and new_text not in existing_text:
                    # Append new text if it's different
                    if existing_text:
                        merged_sections[section_num]["text"] = f"{existing_text}\n\n{new_text}"
                    else:
                        merged_sections[section_num]["text"] = new_text
                
                # Update other fields if they're more complete in the new chunk
                if not merged_sections[section_num].get("heading") and section_data.get("heading"):
                    merged_sections[section_num]["heading"] = section_data["heading"]
                if not merged_sections[section_num].get("title") and section_data.get("title"):
                    merged_sections[section_num]["title"] = section_data["title"]
    
    # Build parent-child relationships
    for number, section in merged_sections.items():
        if section["parent"] and section["parent"] in merged_sections:
            parent_section = merged_sections[section["parent"]]
            if number not in parent_section["children"]:
                parent_section["children"].append(number)
    
    return merged_sections


def extract_with_mistral_ocr(
    pdf_path: str,
    drug_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract structured sections from SmPC PDF using Mistral OCR API.
    
    Automatically handles documents exceeding 8 pages by splitting them into chunks,
    processing each chunk separately, and merging the results.
    
    Args:
        pdf_path: Path to the SmPC PDF file
        drug_id: Optional drug identifier. If not provided, uses the PDF filename stem.
        
    Returns:
        Dictionary matching build_smpc_json() format:
        - "drug_id": str
        - "source_pdf": str
        - "version_hash": str
        - "extracted_at": str (ISO8601)
        - "sections": Dict[str, Dict] - section number -> section data
        - "validation_report": Dict with detection method and validation results
        
    Raises:
        ValueError: If API key is missing or PDF path is invalid
        RuntimeError: If API call fails
    """
    pdf_path_obj = Path(pdf_path)
    if not pdf_path_obj.exists():
        raise ValueError(f"PDF path does not exist: {pdf_path_obj}")
    if not pdf_path_obj.is_file():
        raise ValueError(f"PDF path is not a file: {pdf_path_obj}")
    
    if not Config.MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY is required for Mistral OCR extraction")
    
    if drug_id is None:
        drug_id = pdf_path_obj.stem
    
    # Check if document needs splitting
    total_pages = get_pdf_page_count(str(pdf_path_obj))
    logger.info(
        f"Extracting with Mistral OCR: {pdf_path_obj.name} ({total_pages} pages)"
    )
    
    # Initialize Mistral client
    client = Mistral(api_key=Config.MISTRAL_API_KEY)
    
    # Create document annotation format from Pydantic model
    document_annotation_format = response_format_from_pydantic_model(DocumentAnnotation)
    
    # Split PDF into chunks if needed
    chunks = split_pdf_into_chunks(str(pdf_path_obj), MAX_PAGES_PER_ANNOTATION)
    
    if len(chunks) == 1:
        # Single chunk - process directly (original behavior)
        chunk_path, start_page, end_page = chunks[0]
        logger.info("Processing single chunk (document ≤8 pages)")
        
        try:
            chunk_sections = process_single_chunk(
                client, chunk_path, start_page, end_page, document_annotation_format
            )
            merged_sections = chunk_sections
        finally:
            # Clean up temporary chunk file if it was created
            if chunk_path != str(pdf_path_obj):
                try:
                    Path(chunk_path).unlink(missing_ok=True)
                    Path(chunk_path).parent.rmdir()  # Remove temp directory if empty
                except Exception as e:
                    logger.warning(f"Could not clean up chunk file {chunk_path}: {e}")
    else:
        # Multiple chunks - process each and merge
        logger.info(f"Processing {len(chunks)} chunks separately and merging results")
        all_chunk_sections = []
        temp_files_to_cleanup = []
        
        try:
            for chunk_idx, (chunk_path, start_page, end_page) in enumerate(chunks):
                logger.info(
                    f"Processing chunk {chunk_idx + 1}/{len(chunks)} "
                    f"(pages {start_page}-{end_page})"
                )
                
                # Track temporary files for cleanup
                if chunk_path != str(pdf_path_obj):
                    temp_files_to_cleanup.append(chunk_path)
                
                chunk_sections = process_single_chunk(
                    client, chunk_path, start_page, end_page, document_annotation_format
                )
                all_chunk_sections.append(chunk_sections)
                logger.info(
                    f"Chunk {chunk_idx + 1} extracted {len(chunk_sections)} sections"
                )
            
            # Merge results from all chunks
            logger.info("Merging results from all chunks...")
            merged_sections = merge_chunk_results(all_chunk_sections)
            logger.info(
                f"Merged {len(merged_sections)} unique sections from {len(chunks)} chunks"
            )
            
        finally:
            # Clean up temporary chunk files
            for chunk_path in temp_files_to_cleanup:
                try:
                    Path(chunk_path).unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"Could not clean up chunk file {chunk_path}: {e}")
            
            # Try to remove temp directory if empty
            if temp_files_to_cleanup:
                try:
                    temp_dir = Path(temp_files_to_cleanup[0]).parent
                    if temp_dir.exists() and not any(temp_dir.iterdir()):
                        temp_dir.rmdir()
                except Exception as e:
                    logger.debug(f"Could not remove temp directory: {e}")
    
    # Convert to final format
    smpc_data = {
        "drug_id": drug_id,
        "source_pdf": str(pdf_path_obj),
        "version_hash": file_md5(str(pdf_path_obj)),
        "extracted_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sections": merged_sections,
        "validation_report": {
            "detection_method": "mistral_ocr",
            "num_sections": len(merged_sections),
            "sections_detected": list(merged_sections.keys()),
            "num_chunks": len(chunks),
            "total_pages": total_pages,
        }
    }
    
    logger.info(
        f"Extracted {len(merged_sections)} sections from {drug_id} using Mistral OCR "
        f"({len(chunks)} chunk{'s' if len(chunks) > 1 else ''})"
    )
    
    return smpc_data


