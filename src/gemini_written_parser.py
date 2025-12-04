import fitz  # PyMuPDF
import hashlib
import json
import re
import datetime
import os
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---

# Mapping of Section Numbers to Canonical Keys and expected Icelandic titles
# We use a list of keywords; if ANY match, it's a hit.
SECTION_MAP = {
    "1": {"key": "name", "keywords": ["HEITI LYFS"]},
    "2": {"key": "composition", "keywords": ["INNIHALDSLÝSING", "VIRK EFNI", "VIRKT EFNI"]},
    "3": {"key": "pharmaceutical_form", "keywords": ["LYFJAFORM"]},
    "4": {"key": "clinical_particulars", "keywords": ["KLÍNÍSKAR UPPLÝSINGAR"]},
    "4.1": {"key": "indications", "keywords": ["Ábendingar"]},
    "4.2": {"key": "dosage_and_administration", "keywords": ["Skammtar og lyfjagjöf", "Skammtar"]},
    "4.3": {"key": "contraindications", "keywords": ["Frábendingar"]},
    "4.4": {"key": "warnings_and_precautions", "keywords": ["Sérstök varnaðarorð", "varúðarreglur"]},
    "4.5": {"key": "interactions", "keywords": ["Milliverkanir"]},
    "4.6": {"key": "pregnancy_lactation_fertility", "keywords": ["Frjósemi", "Meðganga", "brjóstagjöf"]},
    "4.7": {"key": "driving_and_machine_use", "keywords": ["Áhrif á hæfni", "akstur"]},
    "4.8": {"key": "adverse_reactions", "keywords": ["Aukaverkanir"]},
    "4.9": {"key": "overdose", "keywords": ["Ofskömmtun"]},
    "5": {"key": "pharmacological_properties", "keywords": ["LYFJAFRÆÐILEGAR UPPLÝSINGAR"]},
    "5.1": {"key": "pharmacodynamic_properties", "keywords": ["Lyfhrif"]},
    "5.2": {"key": "pharmacokinetic_properties", "keywords": ["Lyfjahvörf"]},
    "5.3": {"key": "preclinical_safety_data", "keywords": ["Forklínískar"]},
    "6": {"key": "pharmaceutical_particulars", "keywords": ["LYFJAGERÐARFRÆÐILEGAR"]},
    "6.1": {"key": "excipients", "keywords": ["Hjálparefni"]},
    "6.2": {"key": "incompatibilities", "keywords": ["Ósamrýmanleiki"]},
    "6.3": {"key": "shelf_life", "keywords": ["Geymsluþol"]},
    "6.4": {"key": "special_precautions_for_storage", "keywords": ["Sérstakar varúðarreglur við geymslu"]},
    "6.5": {"key": "container_and_contents", "keywords": ["Gerð íláts", "innihald"]},
    "6.6": {"key": "special_precautions_for_disposal", "keywords": ["Sérstakar varúðarráðstafanir við förgun", "önnur meðhöndlun"]},
    "7": {"key": "marketing_authorisation_holder", "keywords": ["MARKAÐSLEYFISHAFI"]},
    "8": {"key": "marketing_authorisation_number", "keywords": ["MARKAÐSLEYFISNÚMER"]},
    "9": {"key": "first_authorisation_or_renewal_date", "keywords": ["DAGSETNING FYRSTU ÚTGÁFU", "ENDURNÝJUNAR"]},
    "10": {"key": "revision_date", "keywords": ["DAGSETNING ENDURSKOÐUNAR", "DAGSETNING ENDURSKÓÐUNAR"]}
}

def get_md5_hash(file_path: str) -> str:
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logger.error(f"Could not hash file {file_path}: {e}")
        return "error_hashing"

def normalize_string(s: str) -> str:
    """Removes extra whitespace and newlines to make matching easier."""
    return " ".join(s.split()).upper()

def is_valid_smpc(doc: fitz.Document, filename: str) -> bool:
    """
    Checks if the document contains the mandatory SmPC phrase.
    Also strictly rejects Fylgiseðill (Leaflet) or Fylgibréf (Cover Letter).
    """
    full_text_head = ""
    # Scan first 3 pages for metadata
    for i in range(min(3, len(doc))):
        full_text_head += doc[i].get_text("text") + " "

    normalized_text = normalize_string(full_text_head)

    # 1. Check for Negative Keywords (Fail fast)
    if "FYLGISEÐILL" in normalized_text:
        logger.warning(f"[{filename}] REJECTED: Detected 'FYLGISEÐILL' (Patient Leaflet).")
        return False

    if "FYLGIBRÉF" in normalized_text or "FRÆÐSLUEFNI" in normalized_text:
        logger.warning(f"[{filename}] REJECTED: Detected 'FYLGIBRÉF/FRÆÐSLUEFNI' (Educational Material).")
        return False

    # 2. Check for Positive Keyword
    # Note: Sometimes OCR reads "SAMANTEKTÁEIGINLEIKUM" or "SAMANTEKT Á EIGINLEIKUM"
    # We check a few variations or just the core phrase.
    if "SAMANTEKT Á EIGINLEIKUM LYFS" in normalized_text:
        logger.info(f"[{filename}] VALIDATED: Found SmPC Marker.")
        return True

    # Fallback: sometimes OCR misses accents "SAMANTEKT A EIGINLEIKUM"
    if "SAMANTEKT A EIGINLEIKUM" in normalized_text:
        logger.info(f"[{filename}] VALIDATED: Found SmPC Marker (fuzzy match).")
        return True

    logger.warning(f"[{filename}] REJECTED: Could not find 'SAMANTEKT Á EIGINLEIKUM LYFS'. Content snippet: {normalized_text[:200]}...")
    return False

def format_span(span: dict) -> str:
    """
    Applies markdown formatting to a span based on font flags and name.
    
    Args:
        span: A span dictionary from PyMuPDF's get_text("dict") output
        
    Returns:
        Formatted text with markdown (bold, italic, underline)
    """
    text = span['text']
    
    # Simple check for list bullets (often a single character with high font size)
    if len(text.strip()) == 1 and text.strip() in ['•', 'o', '-', '•']:
        return f"\n{text}"  # Treat list items as new lines
    
    # PyMuPDF font flags (0x02=Italic, 0x04=Bold, 0x100=Underlined)
    flags = span.get('flags', 0)
    
    # Check for font name keywords (more reliable for bold/italic in some PDFs)
    font_name = span.get('font', '').lower()
    
    is_bold = (flags & 0x04) or ('bold' in font_name) or ('black' in font_name)
    is_italic = (flags & 0x02) or ('italic' in font_name) or ('oblique' in font_name)
    is_underlined = (flags & 0x100)
    
    # Apply tags (order matters: inner tags first)
    if is_underlined:
        text = f"<u>{text}</u>"
    if is_italic:
        text = f"*{text}*"
    if is_bold:
        text = f"**{text}**"
        
    return text

def clean_text_content(text_list: List[str]) -> str:
    """Joins a list of strings into a clean block."""
    raw = "\n".join(text_list)
    # Fix simple hyphenation (Word- \nbreak -> Wordbreak)
    fixed = re.sub(r'-\s*\n\s*', '', raw)
    # Condense multiple newlines (paragraph separation)
    fixed = re.sub(r'\n\s*\n', '\n\n', fixed)
    return fixed.strip()

def extract_special_considerations_from_section_10(section_10_text: str) -> Optional[str]:
    """
    Checks if section 10 contains information beyond just a date.
    
    Args:
        section_10_text: The text content from section 10 (revision date section)
        
    Returns:
        The additional text content if present, None if only a date is found
    """
    if not section_10_text:
        return None
    
    # Normalize whitespace
    text = " ".join(section_10_text.split())
    
    # Common Icelandic month names
    icelandic_months = [
        "janúar", "febrúar", "mars", "apríl", "maí", "júní",
        "júlí", "ágúst", "september", "október", "nóvember", "desember"
    ]
    
    # Pattern for Icelandic date format: "10. febrúar 2021." or "10. febrúar 2021"
    icelandic_date_pattern = (
        r'\d{1,2}\.\s*(' + '|'.join(icelandic_months) + r')\s+\d{4}\.?'
    )
    
    # Pattern for standard date formats: "2021-02-10", "10/02/2021", "10.02.2021"
    standard_date_patterns = [
        r'\d{4}-\d{1,2}-\d{1,2}',  # ISO format: 2021-02-10
        r'\d{1,2}/\d{1,2}/\d{4}',  # US format: 02/10/2021
        r'\d{1,2}\.\d{1,2}\.\d{4}',  # European format: 10.02.2021
        r'\d{1,2}-\d{1,2}-\d{4}',  # Dash format: 10-02-2021
    ]
    
    # Remove all date patterns from the text
    text_without_dates = text
    text_without_dates = re.sub(icelandic_date_pattern, '', text_without_dates, flags=re.IGNORECASE)
    for pattern in standard_date_patterns:
        text_without_dates = re.sub(pattern, '', text_without_dates)
    
    # Remove common punctuation that might be left after date removal
    text_without_dates = re.sub(r'^[.,;\s]+|[.,;\s]+$', '', text_without_dates)
    text_without_dates = text_without_dates.strip()
    
    # If there's remaining text after removing dates, return it
    if text_without_dates:
        return text_without_dates
    
    return None

def parse_smpc_file(file_path: str) -> Optional[Dict[str, Any]]:
    filename = os.path.basename(file_path)

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None

    try:
        doc = fitz.open(file_path)
    except Exception as e:
        logger.error(f"Error opening PDF {file_path}: {e}")
        return None

    # --- VALIDATION STEP ---
    if not is_valid_smpc(doc, filename):
        total_pages_count = len(doc)
        doc.close()
        return {
            "drug_id": filename.replace(".pdf", ""),
            "source_pdf": file_path,
            "version_hash": get_md5_hash(file_path),
            "extracted_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "sections": {},
            "error": "Document validation failed (Not an SmPC)",
            "special_considerations": None,
            "validation_report": {
                "detection_method": "pymupdf_blocks_validation_failed",
                "num_sections": 0,
                "sections_detected": [],
                "total_pages": total_pages_count
            }
        }

    # Store total_pages before doc.close() is called
    total_pages_count = len(doc)

    # --- EXTRACTION STEP ---

    # We extract blocks with dict format to access font information for markdown formatting
    all_lines = []  # List of tuples: (raw_text, formatted_text)
    all_blocks = []
    
    for page in doc:
        blocks = page.get_text("dict", sort=True)['blocks']
        all_blocks.extend(blocks)
    
    # Process blocks to extract lines with both raw and formatted text
    for block in all_blocks:
        if block['type'] != 0:  # Skip images (type 1)
            continue
        
        for line in block['lines']:
            line_text_formatted = ""
            line_raw_text = ""
            
            for span in line['spans']:
                line_text_formatted += format_span(span)
                line_raw_text += span['text']
            
            line_text_formatted = line_text_formatted.strip()
            line_raw_text = line_raw_text.strip()
            
            if line_raw_text:  # Only add non-empty lines
                all_lines.append((line_raw_text, line_text_formatted))

    sections = {}
    current_section_id = None
    current_text_buffer = []

    # Buffer to handle split headers (e.g. "1." on line A, "HEITI LYFS" on line B)
    potential_header_num = None

    logger.info(f"[{filename}] analyzing {len(all_lines)} text lines...")

    for i, (line_raw, line_formatted) in enumerate(all_lines):

        # Regex to catch "1." or "4.1" or "4.1." at start of line
        # We verify if it is a section by checking the REST of the line OR the NEXT line
        match_num = re.match(r"^(\d{1,2}(\.\d)?)\.?$", line_raw)  # Matches JUST a number like "1." or "4.1"
        match_full = re.match(r"^(\d{1,2}(\.\d)?)\.?\s+(.+)", line_raw)  # Matches "1. HEITI LYFS"

        detected_section = None

        # CASE A: Combined Header (e.g., "1. HEITI LYFS")
        if match_full:
            sec_num = match_full.group(1)
            rest_text = match_full.group(3).strip()

            if sec_num in SECTION_MAP:
                # Check keywords
                keywords = SECTION_MAP[sec_num]["keywords"]
                if any(k.upper() in rest_text.upper() for k in keywords):
                    detected_section = sec_num
                    logger.debug(f"[{filename}] Found Combined Header: {sec_num} -> {rest_text}")

        # CASE B: Split Header (e.g., "1." [newline] "HEITI LYFS")
        elif match_num:
            sec_num = match_num.group(1)
            if sec_num in SECTION_MAP:
                # Look ahead to next line
                if i + 1 < len(all_lines):
                    next_line_raw = all_lines[i+1][0].strip()  # Get raw text from next line
                    keywords = SECTION_MAP[sec_num]["keywords"]
                    if any(k.upper() in next_line_raw.upper() for k in keywords):
                        detected_section = sec_num
                        logger.debug(f"[{filename}] Found Split Header: {sec_num} -> {next_line_raw}")

        # --- PROCESSING STATE MACHINE ---

        if detected_section:
            # 1. Save previous section if exists
            if current_section_id:
                sections[current_section_id]["text"] = clean_text_content(current_text_buffer)

            # 2. Start new section
            current_section_id = detected_section
            current_text_buffer = []

            # Metadata for new section
            sections[current_section_id] = {
                "number": current_section_id,
                "heading": line_raw,  # capture the raw line for heading
                "canonical_key": SECTION_MAP[current_section_id]["key"],
                "title": SECTION_MAP[current_section_id]["keywords"][0].capitalize(),
                "text": ""
            }

        else:
            # Check if this line is actually the second part of a split header
            is_header_title = False
            if current_section_id:
                keywords = SECTION_MAP[current_section_id]["keywords"]
                # If the current line is EXACTLY the keyword (fuzzy match), treat it as header part
                if any(k.upper() == line_raw.upper().strip() for k in keywords):
                     is_header_title = True
                     # Append to heading metadata for clarity
                     sections[current_section_id]["heading"] += " " + line_raw

            if not is_header_title and current_section_id:
                # It is content - use formatted text for markdown formatting
                # Add a space before appending unless it's a newline
                if current_text_buffer and not current_text_buffer[-1].endswith('\n'):
                    current_text_buffer.append(f"\n{line_formatted}")
                else:
                    current_text_buffer.append(line_formatted)

    # Save the very last section
    if current_section_id:
        sections[current_section_id]["text"] = clean_text_content(current_text_buffer)

    doc.close()

    # Validation Report
    detected_keys = list(sections.keys())
    logger.info(f"[{filename}] Parsing complete. Found {len(detected_keys)} sections.")

    # Add parent and children fields to sections
    for section_num, section_data in sections.items():
        # Calculate parent: "4.1" -> "4", "4" -> None
        if "." in section_num:
            parent_num = section_num.rsplit(".", 1)[0]
            section_data["parent"] = parent_num
        else:
            section_data["parent"] = None
        
        # Initialize children list
        section_data["children"] = []
    
    # Build children relationships
    for section_num, section_data in sections.items():
        parent_num = section_data.get("parent")
        if parent_num and parent_num in sections:
            sections[parent_num]["children"].append(section_num)

    # Extract special considerations from section 10 if present
    special_considerations = None
    if "10" in sections:
        section_10_text = sections["10"].get("text", "")
        special_considerations = extract_special_considerations_from_section_10(section_10_text)
        if special_considerations:
            logger.info(f"[{filename}] Found special considerations in section 10: {special_considerations[:100]}...")

    return {
        "drug_id": filename.replace(".pdf", ""),
        "source_pdf": file_path,
        "version_hash": get_md5_hash(file_path),
        "extracted_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sections": sections,
        "special_considerations": special_considerations,
            "validation_report": {
                "detection_method": "pymupdf_spans_markdown_injection",
                "num_sections": len(detected_keys),
                "sections_detected": detected_keys,
                "total_pages": total_pages_count
            }
    }
