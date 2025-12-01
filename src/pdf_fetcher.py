"""PDF fetching and text extraction module."""
import logging
from pathlib import Path
from typing import List, Tuple
import requests
import fitz  # PyMuPDF

from config import Config

logger = logging.getLogger(__name__)


def extract_medication_name_from_url(url: str) -> str:
    """
    Extract medication name from URL pattern:
    https://old.serlyfjaskra.is/FileRepos/[uniqueID]/[MEDICATION_NAME].pdf
    
    Args:
        url: PDF URL
        
    Returns:
        Medication name extracted from filename
    """
    # Extract filename from URL
    filename = url.split("/")[-1]
    
    # Remove .pdf extension
    medication_name = filename.replace(".pdf", "")
    
    # Clean up: replace underscores with spaces, strip whitespace
    medication_name = medication_name.replace("_", " ").strip()
    
    logger.info(f"Extracted medication name: {medication_name} from URL: {url}")
    return medication_name


class PDFDocument:
    """Represents a PDF document with extracted text and metadata."""
    
    def __init__(self, text: str, pages: List[Tuple[int, str]], filename: str, medication_name: str = ""):
        """
        Initialize PDF document.
        
        Args:
            text: Full extracted text
            pages: List of (page_number, page_text) tuples
            filename: Source filename
            medication_name: Medication name extracted from filename
        """
        self.text = text
        self.pages = pages
        self.filename = filename
        self.medication_name = medication_name


def download_pdf(url: str, output_dir: Path) -> Path:
    """
    Download PDF from URL and save to output directory.
    
    Args:
        url: URL of the PDF to download
        output_dir: Directory to save the PDF
        
    Returns:
        Path to the downloaded PDF file
        
    Raises:
        requests.RequestException: If download fails
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract filename from URL or use default
    filename = url.split("/")[-1]
    if not filename.endswith(".pdf"):
        filename = "document.pdf"
    
    filepath = output_dir / filename
    
    logger.info(f"Downloading PDF from {url}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    
    with open(filepath, "wb") as f:
        f.write(response.content)
    
    logger.info(f"PDF saved to {filepath}")
    return filepath


def extract_text_from_pdf(pdf_path: Path, medication_name: str = "") -> PDFDocument:
    """
    Extract text from PDF file using PyMuPDF.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        PDFDocument with extracted text and page information
        
    Raises:
        FileNotFoundError: If PDF file doesn't exist
        Exception: If text extraction fails
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    logger.info(f"Extracting text from {pdf_path}")
    
    doc = fitz.open(pdf_path)
    pages: List[Tuple[int, str]] = []
    full_text_parts: List[str] = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text()
        
        # Basic preprocessing: normalize whitespace
        page_text = " ".join(page_text.split())
        
        pages.append((page_num + 1, page_text))
        full_text_parts.append(page_text)
    
    doc.close()
    
    full_text = "\n\n".join(full_text_parts)
    
    logger.info(f"Extracted {len(pages)} pages from PDF")
    
    return PDFDocument(
        text=full_text,
        pages=pages,
        filename=pdf_path.name,
        medication_name=medication_name
    )


def fetch_and_extract_pdf(url: str) -> PDFDocument:
    """
    Download PDF from URL and extract text.
    
    Args:
        url: URL of the PDF to download
        
    Returns:
        PDFDocument with extracted text and medication name
        
    Raises:
        requests.RequestException: If download fails
        Exception: If text extraction fails
    """
    Config.ensure_directories()
    
    # Extract medication name from URL
    medication_name = extract_medication_name_from_url(url)
    
    # Download PDF
    pdf_path = download_pdf(url, Config.PDFS_DIR)
    
    # Extract text with medication name
    document = extract_text_from_pdf(pdf_path, medication_name=medication_name)
    
    return document
