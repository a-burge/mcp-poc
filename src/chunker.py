"""Section-based document chunking module."""
import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.pdf_fetcher import PDFDocument

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a document chunk with metadata."""
    
    text: str
    section_title: str
    source_document: str
    page_number: int
    medication_name: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize metadata dict if not provided."""
        if self.metadata is None:
            self.metadata = {}


def detect_sections(text: str) -> List[tuple[int, str, str]]:
    """
    Detect section boundaries in text using pattern matching.
    
    Looks for common section patterns like:
    - "1. INDICATIONS"
    - "2. DOSAGE"
    - Numbered sections with uppercase titles
    
    Args:
        text: Full document text
        
    Returns:
        List of (start_index, end_index, section_title) tuples
    """
    sections: List[tuple[int, str, str]] = []
    
    # Pattern for numbered sections (e.g., "1. INDICATIONS", "2. DOSAGE")
    section_pattern = re.compile(
        r'^(\d+\.?\s+[A-ZÁÉÍÓÚÝÞÆÖ][A-ZÁÉÍÓÚÝÞÆÖ\s]+)$',
        re.MULTILINE
    )
    
    # Also look for common SmPC section headers
    common_sections = [
        r'^\d+\.?\s+INDICATIONS',
        r'^\d+\.?\s+DOSAGE',
        r'^\d+\.?\s+CONTRAINDICATIONS',
        r'^\d+\.?\s+WARNINGS',
        r'^\d+\.?\s+ADVERSE',
        r'^\d+\.?\s+INTERACTIONS',
        r'^\d+\.?\s+OVERDOSE',
    ]
    
    lines = text.split('\n')
    current_section_start = 0
    current_section_title = "Introduction"
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        
        # Check if line matches section pattern
        if section_pattern.match(line_stripped) or any(
            re.match(pattern, line_stripped, re.IGNORECASE)
            for pattern in common_sections
        ):
            # Save previous section
            if i > current_section_start:
                sections.append((
                    current_section_start,
                    i,
                    current_section_title
                ))
            
            # Start new section
            current_section_start = i
            current_section_title = line_stripped
    
    # Add final section
    if len(lines) > current_section_start:
        sections.append((
            current_section_start,
            len(lines),
            current_section_title
        ))
    
    logger.info(f"Detected {len(sections)} sections")
    return sections


def extract_drug_name(text: str) -> str:
    """
    Attempt to extract drug name from document text.
    
    Looks for common patterns at the beginning of SmPC documents.
    
    Args:
        text: Document text
        
    Returns:
        Drug name if found, empty string otherwise
    """
    # Look for drug name in first few lines
    lines = text.split('\n')[:10]
    
    for line in lines:
        # Common pattern: drug name might be in all caps or title case
        if len(line.strip()) > 5 and len(line.strip()) < 100:
            # Simple heuristic: if line looks like a title/name
            if line.strip().isupper() or (
                line.strip()[0].isupper() and 
                not line.strip().endswith('.')
            ):
                return line.strip()
    
    return ""


def chunk_document(
    document: PDFDocument,
    chunk_size: int = 300,
    chunk_overlap: int = 50
) -> List[Chunk]:
    """
    Chunk document using section-based approach.
    
    Preserves section integrity by never splitting within a section.
    If a section is too large, subdivides by subsections.
    
    Args:
        document: PDFDocument to chunk
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of Chunk objects with metadata
    """
    logger.info(f"Chunking document: {document.filename}")
    
    # Use medication name from PDFDocument (extracted from filename)
    medication_name = document.medication_name if hasattr(document, 'medication_name') else ""
    
    # Fallback: try to extract from text if not available
    if not medication_name:
        medication_name = extract_drug_name(document.text)
    
    # Detect sections
    sections = detect_sections(document.text)
    
    if not sections:
        # Fallback: treat entire document as one section
        logger.warning("No sections detected, treating as single section")
        sections = [(0, len(document.text.split('\n')), "Document")]
    
    chunks: List[Chunk] = []
    
    # Create text splitter for subdividing large sections
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    
    lines = document.text.split('\n')
    
    for section_start, section_end, section_title in sections:
        # Extract section text
        section_lines = lines[section_start:section_end]
        section_text = "\n".join(section_lines)
        
        # Determine page number (approximate based on position)
        page_number = 1
        for page_num, page_text in document.pages:
            if section_text[:100] in page_text or page_text[:100] in section_text:
                page_number = page_num
                break
        
        # If section is small enough, use as single chunk
        if len(section_text) <= chunk_size:
            chunks.append(Chunk(
                text=section_text,
                section_title=section_title,
                source_document=document.filename,
                page_number=page_number,
                medication_name=medication_name,
                metadata={
                    "section": section_title,
                    "page": page_number,
                    "source": document.filename,
                    "medication_name": medication_name,
                }
            ))
        else:
            # Subdivide large section
            sub_chunks = text_splitter.split_text(section_text)
            
            for i, sub_chunk in enumerate(sub_chunks):
                chunks.append(Chunk(
                    text=sub_chunk,
                    section_title=section_title,
                    source_document=document.filename,
                    page_number=page_number,
                    medication_name=medication_name,
                    metadata={
                        "section": section_title,
                        "page": page_number,
                        "source": document.filename,
                        "medication_name": medication_name,
                        "sub_section": i + 1,
                    }
                ))
    
    logger.info(f"Created {len(chunks)} chunks from {len(sections)} sections")
    return chunks


def chunk_smpc_json(
    smpc_data: Dict[str, Any],
    chunk_size: int = 300,
    chunk_overlap: int = 0
) -> List[Chunk]:
    """
    Chunk structured SmPC JSON by section/subsection.
    
    Preserves section integrity by never splitting within a section.
    If a section is too large, subdivides using text splitter.
    
    Args:
        smpc_data: Dictionary from smpc_parser.build_smpc_json() containing:
            - "drug_id": Drug identifier
            - "source_pdf": Path to source PDF
            - "version_hash": MD5 hash of PDF
            - "sections": Dictionary of sections keyed by section number
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks (should be 0 for sections)
        
    Returns:
        List of Chunk objects with full metadata including:
        - drug_id, section_number, section_title, canonical_key
        - version_hash, pdf_path, json_path, extracted_at
        - document_hash, chunk_id
    """
    drug_id = smpc_data.get("drug_id", "unknown")
    source_pdf = smpc_data.get("source_pdf", "unknown")
    version_hash = smpc_data.get("version_hash", "")
    extracted_at = smpc_data.get("extracted_at", "")
    sections = smpc_data.get("sections", {})
    atc_codes = smpc_data.get("atc_codes", [])  # ATC codes from enriched JSON
    
    logger.info(f"Chunking SmPC JSON for drug: {drug_id}")
    
    chunks: List[Chunk] = []
    
    # Create text splitter for subdividing large sections
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    
    # Process each section
    for section_num, section_data in sections.items():
        section_title = section_data.get("title", section_num)
        section_text = section_data.get("text", "").strip()
        canonical_key = section_data.get("canonical_key", section_num)
        
        if not section_text:
            # Skip empty sections
            continue
        
        # Create document hash for deduplication (drug_id + version_hash)
        document_hash = f"{drug_id}_{version_hash}"
        
        # If section is small enough, use as single chunk
        if len(section_text) <= chunk_size:
            chunk_id = f"{drug_id}_{section_num}_0"
            chunks.append(Chunk(
                text=section_text,
                section_title=section_title,
                source_document=source_pdf,
                page_number=1,  # Page number not available from JSON
                medication_name=drug_id,
                metadata={
                    "drug_id": drug_id,
                    "section_number": section_num,
                    "section_title": section_title,
                    "canonical_key": canonical_key,
                    "version_hash": version_hash,
                    "pdf_path": source_pdf,
                    "extracted_at": extracted_at,
                    "document_hash": document_hash,
                    "chunk_id": chunk_id,
                    "section": section_title,
                    "source": source_pdf,
                    "medication_name": drug_id,
                }
            ))
        else:
            # Subdivide large section
            sub_chunks = text_splitter.split_text(section_text)
            
            for i, sub_chunk in enumerate(sub_chunks):
                chunk_id = f"{drug_id}_{section_num}_{i}"
                chunks.append(Chunk(
                    text=sub_chunk,
                    section_title=section_title,
                    source_document=source_pdf,
                    page_number=1,  # Page number not available from JSON
                    medication_name=drug_id,
                    metadata={
                        "drug_id": drug_id,
                        "section_number": section_num,
                        "section_title": section_title,
                        "canonical_key": canonical_key,
                        "version_hash": version_hash,
                        "pdf_path": source_pdf,
                        "extracted_at": extracted_at,
                        "document_hash": document_hash,
                        "chunk_id": chunk_id,
                        "section": section_title,
                        "source": source_pdf,
                        "medication_name": drug_id,
                        "sub_section": i + 1,
                        "atc_codes": atc_codes,  # Add ATC codes to metadata
                    }
                ))
    
    logger.info(f"Created {len(chunks)} chunks from {len(sections)} sections for {drug_id}")
    return chunks
