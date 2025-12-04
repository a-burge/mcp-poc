"""
SMPC Parser — Extracts structured sections & subsections from Icelandic SmPC PDFs.

Each numbered heading (e.g. "1", "4", "4.1", "5.3", "6.6", "10") becomes a section node:

{
  "drug_id": str,
  "source_pdf": str,
  "version_hash": str,
  "extracted_at": str (ISO8601),
  "sections": {
    "4.3": {
      "number": "4.3",
      "parent": "4",
      "heading": "4.3 Frábendingar",
      "title": "Frábendingar",
      "canonical_key": "contraindications",
      "text": "full text…",
      "children": []
    },
    ...
  }
}
"""

import re
import json
import hashlib
import datetime
import logging
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


# ---------------------------------------
# Canonical Section Key Mapping
# ---------------------------------------

# Top-level sections (1–10) – based on EU SmPC template semantics
CANONICAL_TOP = {
    "1": "name",
    "2": "composition",
    "3": "pharmaceutical_form",
    "4": "clinical_particulars",
    "5": "pharmacological_properties",
    "6": "pharmaceutical_particulars",
    "7": "marketing_authorisation_holder",
    "8": "marketing_authorisation_number",
    "9": "first_authorisation_or_renewal_date",
    "10": "revision_date",
}

# Known standard subsections
CANONICAL_SUB = {
    # 4.x Clinical particulars
    "4.1": "indications",
    "4.2": "dosage_and_administration",
    "4.3": "contraindications",
    "4.4": "warnings_and_precautions",
    "4.5": "interactions",
    "4.6": "pregnancy_lactation_fertility",
    "4.7": "driving_and_machine_use",
    "4.8": "adverse_reactions",
    "4.9": "overdose",
    # 5.x Pharmacological properties
    "5.1": "pharmacodynamic_properties",
    "5.2": "pharmacokinetic_properties",
    "5.3": "preclinical_safety_data",
    # 6.x Pharmaceutical particulars
    "6.1": "excipients",
    "6.2": "incompatibilities",
    "6.3": "shelf_life",
    "6.4": "special_precautions_for_storage",
    "6.5": "container_and_contents",
    "6.6": "special_precautions_for_disposal",
}

UNKNOWN_SECTION_TITLE = "UNKNOWN SECTION"

def canonical_key_for(number: str) -> str:
    """
    Map section/subsection number to a canonical key.

    Args:
        number: Section number (e.g., "4", "4.3", "6.1").

    Returns:
        Canonical key string (e.g., "clinical_particulars", "contraindications").
        Falls back to the number itself if unknown.
    """
    if number in CANONICAL_SUB:
        return CANONICAL_SUB[number]
    root = number.split(".")[0]
    if root in CANONICAL_TOP:
        return CANONICAL_TOP[root]
    return number


# ---------------------------------------
# Regex for headings
# ---------------------------------------

# Example matches:
# "1. HEITI LYFS"
# "4.1 Ábendingar"
# "6.3 Geymsluþol"
# Enhanced to handle optional periods, multiple spaces, case variations
HEADING_REGEX = re.compile(
    r"^\s*(\d{1,2}(?:\.\d{1,2})*)\s*\.?\s+(.+)$"
)

# Common units that should not be detected as headings
UNIT_BLACKLIST = {
    'mg', 'ml', 'g', 'kg', 'μg', 'μl',  # Metric units
    'ár', 'ára', 'mánuðir', 'mánuði', 'mánuðum',  # Icelandic time units
}
MIN_HEADING_TITLE_LENGTH = 3


def is_valid_heading(section_number: str, title: str, next_line: Optional[str] = None) -> bool:
    """
    Validate if a detected heading is actually a section heading.
    
    Rejects false positives like "1 mg" (dosage) or "1 ár" (duration).
    
    Args:
        section_number: Extracted section number (e.g., "1", "4.3")
        title: Title text after the number
        next_line: Optional next line for context-aware detection
    
    Returns:
        True if this appears to be a valid heading, False otherwise
    """
    title_stripped = title.strip()
    
    # Too short - probably a unit
    if len(title_stripped) < MIN_HEADING_TITLE_LENGTH:
        return False
    
    # Check against unit blacklist (case-insensitive)
    if title_stripped.lower() in UNIT_BLACKLIST:
        return False
    
    # Context-aware: if next line is also a short number+unit pattern, likely a table
    if next_line:
        next_match = HEADING_REGEX.match(next_line.rstrip())
        if next_match:
            next_title = next_match.group(2).strip()
            if (len(next_title) < MIN_HEADING_TITLE_LENGTH or 
                next_title.lower() in UNIT_BLACKLIST):
                # Both current and next look like units - likely a table
                return False
    
    return True


# ---------------------------------------
# PDF Extraction
# ---------------------------------------

def extract_pdf_text(path: str) -> str:
    """
    Extract full plain text from PDF using PyMuPDF.

    Args:
        path: Path to the PDF file.

    Returns:
        Full text content of the PDF as a single string with newlines.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        RuntimeError: If the PDF cannot be opened or text extraction fails.
    """
    try:
        doc = fitz.open(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"PDF file not found: {path}")
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF file {path}: {e}") from e
    
    try:
        text = "\n".join(page.get_text("text") for page in doc)
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from PDF {path}: {e}") from e
    finally:
        doc.close()
    return text


def extract_pdf_text_with_structure(
    path: str,
    font_size_threshold: float = 1.2
) -> List[Tuple[str, Optional[str], float, bool]]:
    """
    Extract text from PDF with font structure information for heading detection.
    
    Uses PyMuPDF's structured text extraction to identify headings by font size
    and style (bold). Returns lines with their font properties.
    
    Args:
        path: Path to the PDF file.
        font_size_threshold: Multiplier of median font size to identify headings
            (default: 1.2, meaning headings are 1.2x larger than body text).
    
    Returns:
        List of tuples: (line_text, section_number, font_size, is_bold)
        - line_text: The text content of the line
        - section_number: Extracted section number if line matches heading pattern, None otherwise
        - font_size: Font size in points
        - is_bold: Whether text is bold
    
    Raises:
        FileNotFoundError: If the PDF file does not exist.
        RuntimeError: If the PDF cannot be opened or text extraction fails.
    """
    try:
        doc = fitz.open(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"PDF file not found: {path}")
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF file {path}: {e}") from e
    
    try:
        all_lines: List[Tuple[str, float, bool]] = []
        font_sizes: List[float] = []
        
        # Extract text with structure from all pages
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" not in block:
                    continue
                
                for line in block["lines"]:
                    line_text_parts: List[str] = []
                    line_font_sizes: List[float] = []
                    line_is_bold = False
                    
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if not text:
                            continue
                        
                        font_size = span["size"]
                        flags = span.get("flags", 0)
                        is_bold = bool(flags & 16)  # Bit 4 indicates bold
                        
                        line_text_parts.append(text)
                        line_font_sizes.append(font_size)
                        if is_bold:
                            line_is_bold = True
                        font_sizes.append(font_size)
                    
                    if line_text_parts:
                        # Use average font size for the line
                        avg_font_size = sum(line_font_sizes) / len(line_font_sizes)
                        line_text = " ".join(line_text_parts)
                        all_lines.append((line_text, avg_font_size, line_is_bold))
        
        # Calculate median font size for body text
        if font_sizes:
            sorted_sizes = sorted(font_sizes)
            median_font_size = sorted_sizes[len(sorted_sizes) // 2]
            threshold_size = median_font_size * font_size_threshold
        else:
            median_font_size = 12.0  # Default fallback
            threshold_size = 12.0  # Default fallback
        
        # Identify headings and extract section numbers
        result: List[Tuple[str, Optional[str], float, bool]] = []
        
        for idx, (line_text, font_size, is_bold) in enumerate(all_lines):
            section_number = None
            
            # Check if this line could be a heading based on font properties
            is_potential_heading = (font_size >= threshold_size) or is_bold
            
            if is_potential_heading:
                # Try to extract section number using regex
                m = HEADING_REGEX.match(line_text.rstrip())
                if m:
                    section_number = m.group(1)
                    title = m.group(2).strip()
                    
                    # Get next line for context-aware validation
                    next_line = None
                    if idx + 1 < len(all_lines):
                        next_line = all_lines[idx + 1][0]
                    
                    # Validate heading - reject false positives like "1 mg" or "1 ár"
                    if not is_valid_heading(section_number, title, next_line):
                        section_number = None
                        logger.debug(
                            f"Rejected false positive heading: '{line_text.strip()}' "
                            f"(detected as unit or too short)"
                        )
            
            result.append((line_text, section_number, font_size, is_bold))
        
        logger.debug(
            f"Extracted {len(result)} lines with structure, "
            f"median font size: {median_font_size:.1f}, "
            f"threshold: {threshold_size:.1f}"
        )
        
        return result
        
    except Exception as e:
        raise RuntimeError(f"Failed to extract structured text from PDF {path}: {e}") from e
    finally:
        doc.close()


# ---------------------------------------
# Expected Sections Validator
# ---------------------------------------

def get_expected_sections() -> Dict[str, List[str]]:
    """
    Get expected section numbers based on EU SmPC template.
    
    Returns:
        Dictionary with:
        - "top_level": List of expected top-level section numbers (1-10)
        - "subsections": List of expected subsection numbers
    """
    top_level = [str(i) for i in range(1, 11)]  # Sections 1-10
    subsections = list(CANONICAL_SUB.keys())
    
    return {
        "top_level": top_level,
        "subsections": subsections
    }


def validate_and_fill_sections(
    sections: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Validate detected sections against expected sections and fill missing parents.
    
    Identifies missing sections but doesn't require ALL sections (handles "light touch" docs).
    For missing parent sections: creates placeholder entries if subsections exist.
    For missing subsections: logs warnings but doesn't create placeholders (subsections are optional).
    
    Args:
        sections: Dictionary of detected sections keyed by section number.
    
    Returns:
        Validation report dictionary with:
        - "missing_top_level": List of missing top-level section numbers
        - "missing_subsections": List of missing subsection numbers
        - "is_light_touch": bool indicating if document appears to be light touch (< 5 top-level sections)
        - "detected_top_level": List of detected top-level sections
        - "detected_subsections": List of detected subsections
    """
    expected = get_expected_sections()
    
    # Get detected sections
    detected_top_level = [
        num for num in sections.keys()
        if num.isdigit() and 1 <= int(num) <= 10
    ]
    detected_subsections = [
        num for num in sections.keys()
        if "." in num and num in CANONICAL_SUB
    ]
    
    # Find missing sections
    missing_top_level = [
        num for num in expected["top_level"]
        if num not in sections
    ]
    missing_subsections = [
        num for num in expected["subsections"]
        if num not in sections
    ]
    
    # Check if light touch document (< 5 top-level sections)
    is_light_touch = len(detected_top_level) < 5
    
    # Create placeholder entries for missing parent sections if children exist
    for subsection_num in detected_subsections:
        if "." in subsection_num:
            parent = subsection_num.rsplit(".", 1)[0]
            if parent not in sections:
                # Parent is missing but has children, create placeholder
                sections[parent] = {
                    "number": parent,
                    "parent": None,
                    "heading": f"{parent}. {UNKNOWN_SECTION_TITLE}",
                    "title": UNKNOWN_SECTION_TITLE,
                    "canonical_key": canonical_key_for(parent),
                    "text": "",
                    "children": [],
                }
                logger.warning(
                    f"Created placeholder for missing parent section {parent} "
                    f"(subsection {subsection_num} exists)"
                )
    
    # Log warnings for missing sections
    if missing_top_level and not is_light_touch:
        logger.warning(
            f"Missing top-level sections: {', '.join(missing_top_level)}"
        )
    
    if missing_subsections:
        logger.debug(
            f"Missing subsections: {', '.join(missing_subsections[:10])}"
            + (f" (and {len(missing_subsections) - 10} more)" if len(missing_subsections) > 10 else "")
        )
    
    if is_light_touch:
        logger.info(
            f"Document appears to be 'light touch' type "
            f"(only {len(detected_top_level)} top-level sections detected)"
        )
    
    return {
        "missing_top_level": missing_top_level,
        "missing_subsections": missing_subsections,
        "is_light_touch": is_light_touch,
        "detected_top_level": detected_top_level,
        "detected_subsections": detected_subsections,
    }


# ---------------------------------------
# Parsing Logic
# ---------------------------------------

def parse_sections_from_structured(
    structured_lines: List[Tuple[str, Optional[str], float, bool]]
) -> Dict[str, Dict[str, Any]]:
    """
    Parse sections from structured lines with font information.
    
    Args:
        structured_lines: List of (line_text, section_number, font_size, is_bold) tuples
            from extract_pdf_text_with_structure().
    
    Returns:
        Dictionary keyed by section number with section data.
    """
    sections: Dict[str, Dict] = {}
    current_number: Optional[str] = None

    for line_text, section_number, font_size, is_bold in structured_lines:
        stripped = line_text.rstrip()
        
        # If this line has a detected section number, it's a heading
        if section_number is not None:
            number = section_number
            # Extract title from heading line
            m = HEADING_REGEX.match(stripped)
            if m:
                rest = m.group(2).strip()
            else:
                # Fallback: try to extract title after number
                rest = stripped.replace(number, "", 1).strip()
                if rest.startswith("."):
                    rest = rest[1:].strip()
            
            heading = stripped

            # Determine parent (None for top level)
            parent = None
            if "." in number:
                parent = number.rsplit(".", 1)[0]

                # Ensure parent exists, even if the parent heading line never appeared
                if parent not in sections:
                    sections[parent] = {
                        "number": parent,
                        "parent": None,
                        "heading": f"{parent}. {UNKNOWN_SECTION_TITLE}",
                        "title": UNKNOWN_SECTION_TITLE,
                        "canonical_key": canonical_key_for(parent),
                        "text_lines": [],
                        "children": [],
                    }

            # Initialize or overwrite the section
            sections[number] = {
                "number": number,
                "parent": parent,
                "heading": heading,
                "title": rest,
                "canonical_key": canonical_key_for(number),
                "text_lines": [],
                "children": [],
            }

            current_number = number
            continue

        # Non-heading: append text to current section, if any
        if current_number is not None and stripped:
            sections[current_number]["text_lines"].append(stripped)

    # Build children lists based on parent relationships
    for num, sec in sections.items():
        parent = sec["parent"]
        if parent:
            if parent in sections:
                sections[parent]["children"].append(num)
            else:
                # Should not happen because we create missing parents above,
                # but keep this guard for robustness.
                sections[parent] = {
                    "number": parent,
                    "parent": None,
                    "heading": f"{parent}. {UNKNOWN_SECTION_TITLE}",
                    "title": UNKNOWN_SECTION_TITLE,
                    "canonical_key": canonical_key_for(parent),
                    "text_lines": [],
                    "children": [num],
                }

    # Join text_lines into text field and remove text_lines
    for sec in sections.values():
        sec["text"] = "\n".join(sec["text_lines"])
        del sec["text_lines"]

    return sections


def parse_sections(
    text: str,
    use_font_detection: bool = True,
    pdf_path: Optional[str] = None,
    font_size_threshold: float = 1.2
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Parse plain text of an SmPC into numbered sections using hybrid approach.

    Args:
        text: Plain text content of the SmPC document.
        use_font_detection: If True, try font-based detection first (requires pdf_path).
        pdf_path: Path to PDF file (required if use_font_detection is True).
        font_size_threshold: Font size threshold multiplier for heading detection.

    Returns:
        Tuple of:
        - Dictionary keyed by section number (e.g., "1", "4.1", "6.3"). Each value
          is a dictionary containing:
          - "number": Section number string
          - "parent": Parent section number (None for top-level)
          - "heading": Full heading line
          - "title": Title text without number prefix
          - "canonical_key": Canonical key for the section
          - "text": Full text content of the section
          - "children": List of child section numbers
        - Validation report dictionary with detection method and validation results
    """
    detection_method = "regex"
    sections: Dict[str, Dict[str, Any]] = {}
    
    # Try font-based detection first if enabled and PDF path is available
    if use_font_detection and pdf_path:
        try:
            structured_lines = extract_pdf_text_with_structure(
                pdf_path,
                font_size_threshold=font_size_threshold
            )
            
            # Count how many headings were detected
            headings_detected = sum(
                1 for _, section_num, _, _ in structured_lines
                if section_num is not None
            )
            
            # If font detection found at least 5 headings, use it
            if headings_detected >= 5:
                sections = parse_sections_from_structured(structured_lines)
                detection_method = "font"
                logger.info(
                    f"Used font-based detection: found {headings_detected} headings"
                )
            else:
                logger.warning(
                    f"Font detection found only {headings_detected} headings, "
                    f"falling back to regex"
                )
        except Exception as e:
            logger.warning(
                f"Font-based detection failed: {e}, falling back to regex"
            )
    
    # Fallback to regex-based parsing if font detection wasn't used or failed
    if detection_method == "regex":
        lines = text.splitlines()

        # Normalize lines: merge split headings like "1." + "HEITI LYFS"
        normalized_lines = []
        skip_next = False
        for i in range(len(lines)):
            if skip_next:
                skip_next = False
                continue
            current = lines[i].strip()
            # Detect numeric-only heading number
            if re.fullmatch(r"\d{1,2}(?:\.\d{1,2})*", current):
                if i + 1 < len(lines):
                    nxt = lines[i+1].strip()
                    # If next line looks like a heading title
                    if re.match(r"^[A-Za-zÁÉÍÓÚÝÞÆÖ]", nxt):
                        normalized_lines.append(f"{current} {nxt}")
                        skip_next = True
                        continue
            normalized_lines.append(current)
        lines = normalized_lines

        sections = {}
        current_number: Optional[str] = None

        for idx, line in enumerate(lines):
            stripped = line.rstrip()

            # Check if this line is a heading
            m = HEADING_REGEX.match(stripped)
            if m:
                number = m.group(1)          # e.g. "4.3"
                rest = m.group(2).strip()    # e.g. "Frábendingar" or "KLÍNÍSKAR UPPLÝSINGAR"
                
                # Get next line for context-aware validation
                next_line = None
                if idx + 1 < len(lines):
                    next_line = lines[idx + 1]
                
                # Validate heading - reject false positives like "1 mg" or "1 ár"
                if not is_valid_heading(number, rest, next_line):
                    logger.debug(
                        f"Rejected false positive heading: '{stripped}' "
                        f"(detected as unit or too short)"
                    )
                    # Treat as regular text, not a heading
                    if current_number is not None and stripped:
                        sections[current_number]["text_lines"].append(stripped)
                    continue
                
                heading = stripped

                # Determine parent (None for top level)
                parent = None
                if "." in number:
                    parent = number.rsplit(".", 1)[0]

                    # Ensure parent exists, even if the parent heading line never appeared
                    if parent not in sections:
                        sections[parent] = {
                            "number": parent,
                            "parent": None,
                            "heading": f"{parent}. {UNKNOWN_SECTION_TITLE}",
                            "title": UNKNOWN_SECTION_TITLE,
                            "canonical_key": canonical_key_for(parent),
                            "text_lines": [],
                            "children": [],
                        }

                # Initialize or overwrite the section
                sections[number] = {
                    "number": number,
                    "parent": parent,
                    "heading": heading,
                    "title": rest,
                    "canonical_key": canonical_key_for(number),
                    "text_lines": [],
                    "children": [],
                }

                current_number = number
                continue

            # Non-heading: append text to current section, if any
            if current_number is not None and stripped:
                sections[current_number]["text_lines"].append(stripped)

        # Build children lists based on parent relationships
        for num, sec in sections.items():
            parent = sec["parent"]
            if parent:
                if parent in sections:
                    sections[parent]["children"].append(num)
                else:
                    # Should not happen because we create missing parents above,
                    # but keep this guard for robustness.
                    sections[parent] = {
                        "number": parent,
                        "parent": None,
                        "heading": f"{parent}. {UNKNOWN_SECTION_TITLE}",
                        "title": UNKNOWN_SECTION_TITLE,
                        "canonical_key": canonical_key_for(parent),
                        "text_lines": [],
                        "children": [num],
                    }

        # Join text_lines into text field and remove text_lines
        for sec in sections.values():
            sec["text"] = "\n".join(sec["text_lines"])
            del sec["text_lines"]

    # Always apply validation
    validation_report = validate_and_fill_sections(sections)
    validation_report["detection_method"] = detection_method
    
    return sections, validation_report


# ---------------------------------------
# Hashing Utility
# ---------------------------------------

def file_md5(path: str) -> str:
    """
    Calculate MD5 hash of a file.

    Args:
        path: Path to the file to hash.

    Returns:
        Hexadecimal MD5 hash string.

    Raises:
        FileNotFoundError: If the file does not exist.
        PermissionError: If permission is denied reading the file.
        RuntimeError: If hashing fails for any other reason.
    """
    hash_md5 = hashlib.md5()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found for MD5 calculation: {path}")
    except PermissionError:
        raise PermissionError(f"Permission denied reading file for MD5: {path}")
    except Exception as e:
        raise RuntimeError(f"Failed to calculate MD5 hash for {path}: {e}") from e
    return hash_md5.hexdigest()


# ---------------------------------------
# Ingredients Extraction
# ---------------------------------------

def extract_ingredients_from_text(text: str, is_active: bool = False) -> List[str]:
    """
    Extract ingredient names from section text.
    
    Attempts to identify ingredient names by looking for:
    - Lines that start with ingredient names (often capitalized)
    - Common patterns like "mg", "g", percentages
    - Parenthetical E-numbers for excipients
    
    Args:
        text: Section text to extract from
        is_active: If True, looking for active ingredients; if False, excipients
        
    Returns:
        List of ingredient name strings (without amounts)
    """
    if not text or not text.strip():
        return []
    
    ingredients = []
    lines = text.split('\n')
    
    # Common patterns to identify ingredient lines
    # Active ingredients often have "mg", "g", percentages, or "jafngildir"
    # Excipients may have E-numbers or be listed simply
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Skip header lines and common section text
        if any(skip in line.lower() for skip in [
            'innihaldslýsing', 'virk efni', 'hjálparefni', 
            'sjá lista', 'sjá kafla', 'innihald'
        ]):
            continue
        
        # Try to extract ingredient name (before amounts/descriptions)
        # Pattern: ingredient name, optionally followed by amount or description
        # Remove common prefixes and suffixes
        
        # Remove amount patterns (e.g., "23,2 mg", "50 mg/g", "1,54 ml")
        cleaned = re.sub(r'\d+[.,]\d*\s*(mg|g|ml|%|mg/g|mg/ml)', '', line, flags=re.IGNORECASE)
        cleaned = re.sub(r'\d+\s*(mg|g|ml|%)', '', cleaned, flags=re.IGNORECASE)
        
        # Remove parenthetical content but keep E-numbers for excipients
        if not is_active:
            # Keep E-numbers like (E1520), (E321)
            cleaned = re.sub(r'\([^)]*\)', lambda m: m.group(0) if 'E' in m.group(0) else '', cleaned)
        else:
            cleaned = re.sub(r'\([^)]*\)', '', cleaned)
        
        # Remove common descriptive text
        cleaned = re.sub(r'\s*(jafngildir|í hverju|í hverri|í hverjum|sem|er|innan|með)', '', cleaned, flags=re.IGNORECASE)
        
        # Extract first meaningful word/phrase (usually the ingredient name)
        # Split by common separators
        parts = re.split(r'[,\n]', cleaned)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            # Skip if it's just a number or very short
            if len(part) < 3 or part.isdigit():
                continue
            
            # Skip common non-ingredient words
            if part.lower() in ['og', 'eða', 'sem', 'er', 'innan', 'með', 'í', 'af', 'fyrir']:
                continue
            
            # Clean up the ingredient name
            ingredient = part.split()[0] if part.split() else part
            ingredient = ingredient.strip('.,;:()[]')
            
            if len(ingredient) >= 3 and ingredient not in ingredients:
                ingredients.append(ingredient)
    
    return ingredients


def normalize_ingredient_name(name: str) -> str:
    """
    Normalize ingredient name for deduplication.
    
    Removes spaces, converts to lowercase, removes common suffixes.
    Used to detect if two differently formatted names refer to the same ingredient.
    
    Args:
        name: Ingredient name to normalize
        
    Returns:
        Normalized name string
    """
    # Remove E-numbers for normalization (we'll add them back later)
    normalized = re.sub(r'\s*\(E\d+\)', '', name)
    # Remove spaces and convert to lowercase
    normalized = re.sub(r'\s+', '', normalized.lower())
    # Remove common punctuation
    normalized = normalized.strip('.,;:()[]')
    return normalized


def create_ingredients_summary(sections: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Create a consolidated ingredients summary section from sections 2 and 6.1.
    
    Extracts:
    - Active ingredients from section 2 (composition)
    - Excipients from section 6.1 (complete authoritative list)
    - Excipients mentioned in section 2 (if not already in 6.1)
    
    Handles duplicates by:
    - Normalizing names (case-insensitive, space-insensitive)
    - Preferring section 6.1 names (more complete/authoritative)
    - Merging E-numbers when present
    
    Args:
        sections: Dictionary of sections from parsed SmPC
        
    Returns:
        Dictionary representing the ingredients_summary section, or None if
        section 2 is missing or empty
    """
    section_2 = sections.get("2")
    section_6_1 = sections.get("6.1")
    
    # Need at least section 2 (composition) to create summary
    if not section_2 or not section_2.get("text", "").strip():
        logger.debug("Section 2 (composition) missing or empty, skipping ingredients summary")
        return None
    
    text_2 = section_2.get("text", "")
    text_6_1 = section_6_1.get("text", "") if section_6_1 else ""
    
    active_ingredients = []
    excipients = []  # List of ingredient name strings
    excipients_normalized = {}  # Map: normalized_name -> full_name (for deduplication)
    
    # Extract active ingredients from section 2
    # Active ingredients are usually mentioned before "Hjálparefni" marker
    lines_2 = text_2.split('\n')
    found_excipient_marker = False
    
    for line in lines_2:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        
        line_lower = line_stripped.lower()
        
        # Check for excipient section marker
        if 'hjálparefni' in line_lower or 'sjá lista yfir öll hjálparefni' in line_lower:
            found_excipient_marker = True
            continue
        
        # Before excipient marker: look for active ingredients
        if not found_excipient_marker:
            # Pattern: "X mg af [ingredient name]" or "[ingredient name] (X%)"
            # Remove amounts and extract ingredient name
            # Example: "23,2 mg af díklófenaktvíetýlamíni" -> "díklófenaktvíetýlamíni"
            
            # Try to find pattern: "mg af [ingredient]" or "g af [ingredient]"
            match = re.search(r'(?:mg|g)\s+af\s+([a-záéíóúýþæö]+)', line_lower)
            if match:
                ingredient = match.group(1).strip()
                if len(ingredient) >= 3 and ingredient not in active_ingredients:
                    active_ingredients.append(ingredient)
            else:
                # Alternative: look for ingredient name followed by percentage or "jafngildir"
                # Remove common words and amounts
                cleaned = re.sub(r'\d+[.,]\d*\s*(mg|g|ml|%|mg/g|mg/ml)', '', line_stripped, flags=re.IGNORECASE)
                cleaned = re.sub(r'\([^)]*\)', '', cleaned)  # Remove parentheticals
                cleaned = re.sub(r'\s+(jafngildir|sem|er|innan|í|af|fyrir|gramm|eitt)', '', cleaned, flags=re.IGNORECASE)
                cleaned = cleaned.strip('.,;:()[]')
                
                # Extract first substantial word (likely ingredient name)
                words = cleaned.split()
                for word in words:
                    word_clean = word.strip('.,;:()[]')
                    # Skip if too short, is a number, or common word
                    if (len(word_clean) >= 5 and 
                        not word_clean.isdigit() and
                        word_clean.lower() not in ['innihaldslýsing', 'innihald', 'voltaren'] and
                        word_clean not in active_ingredients):
                        active_ingredients.append(word_clean)
                        break
        
        # After excipient marker: extract excipients mentioned in section 2
        # Only add if not already in section 6.1 (to avoid duplicates)
        else:
            # Pattern: "X mg af [ingredient] (E####)" or similar
            e_number_match = re.search(r'\(E\d+\)', line_stripped)
            e_number = e_number_match.group(0) if e_number_match else ""
            
            # Extract ingredient name
            cleaned = re.sub(r'\d+[.,]\d*\s*(mg|g|ml|%|mg/g)', '', line_stripped, flags=re.IGNORECASE)
            
            # Handle parenthetical content (may contain sub-ingredients)
            paren_match = re.search(r'\(([^)]+)\)', cleaned)
            if paren_match and 'ilmefni' in line_lower:
                # Special case: "ilmefni (ingredient1, ingredient2, ...)"
                sub_ingredients = [s.strip() for s in paren_match.group(1).split(',')]
                for sub_ing in sub_ingredients:
                    if len(sub_ing) >= 3:
                        normalized = normalize_ingredient_name(sub_ing)
                        # Only add if not already in section 6.1 list
                        if normalized not in excipients_normalized:
                            excipients.append(sub_ing)
                            excipients_normalized[normalized] = sub_ing
            
            # Extract main ingredient name
            cleaned = re.sub(r'\([^)]*\)', '', cleaned)
            cleaned = re.sub(r'\s+(í|af|hverju|grammi|hlaupi)', '', cleaned, flags=re.IGNORECASE)
            cleaned = cleaned.strip('.,;:()[]')
            
            words = cleaned.split()
            if words:
                ingredient = words[0].strip('.,;:()[]')
                if len(ingredient) >= 3:
                    if e_number:
                        ingredient = f"{ingredient} {e_number}"
                    
                    # Check for duplicates using normalized name
                    normalized = normalize_ingredient_name(ingredient)
                    if normalized not in excipients_normalized:
                        # Not in section 6.1, add it (section 2 may have additional info)
                        excipients.append(ingredient)
                        excipients_normalized[normalized] = ingredient
                    else:
                        # Already exists from section 6.1, but check if we need to merge E-number
                        existing = excipients_normalized[normalized]
                        if e_number and f"({e_number})" not in existing:
                            # Update existing entry with E-number from section 2
                            idx = excipients.index(existing)
                            excipients[idx] = f"{existing} {e_number}"
                            excipients_normalized[normalized] = excipients[idx]
    
    # Extract excipients from section 6.1 FIRST (authoritative complete list)
    # This ensures we use the most complete/authoritative names
    if text_6_1:
        lines_6_1 = text_6_1.split('\n')
        for line in lines_6_1:
            line = line.strip()
            if not line:
                continue
            
            # Skip header text
            if any(skip in line.lower() for skip in ['hjálparefni', 'innihald']):
                continue
            
            # Extract ingredient name (may include E-number in parentheses)
            e_number_match = re.search(r'\(E\d+\)', line)
            e_number = e_number_match.group(0) if e_number_match else ""
            
            # Handle parenthetical content for complex ingredients
            # Example: "Eukalyptusolía (bensýlalkóhól, sítrónellól, ...)"
            paren_match = re.search(r'\(([^)]+)\)', line)
            if paren_match and not e_number:
                # Extract sub-ingredients from parentheses
                sub_ingredients = [s.strip() for s in paren_match.group(1).split(',')]
                for sub_ing in sub_ingredients:
                    if len(sub_ing) >= 3:
                        normalized = normalize_ingredient_name(sub_ing)
                        if normalized not in excipients_normalized:
                            excipients.append(sub_ing)
                            excipients_normalized[normalized] = sub_ing
            
            # Extract main ingredient name
            cleaned = re.sub(r'\([^)]*\)', '', line)
            cleaned = cleaned.strip('.,;:()[]')
            
            if cleaned:
                # Use full cleaned line as ingredient name (preserves formatting)
                ingredient = cleaned.strip('.,;:()[]')
                
                if len(ingredient) >= 3:
                    # Add E-number if present
                    if e_number:
                        ingredient = f"{ingredient} {e_number}"
                    
                    # Check for duplicates using normalized name
                    normalized = normalize_ingredient_name(ingredient)
                    if normalized not in excipients_normalized:
                        excipients.append(ingredient)
                        excipients_normalized[normalized] = ingredient
                    else:
                        # If duplicate found, prefer the section 6.1 version (already stored)
                        # But merge E-numbers if the new one has one
                        existing = excipients_normalized[normalized]
                        if e_number and f"({e_number})" not in existing:
                            # Update existing entry with E-number
                            idx = excipients.index(existing)
                            excipients[idx] = f"{existing} {e_number}"
                            excipients_normalized[normalized] = excipients[idx]
    
    # Build summary text
    summary_lines = []
    summary_lines.append("Yfirlit innihaldsefna:")
    summary_lines.append("")
    
    if active_ingredients:
        summary_lines.append("Virk efni:")
        for ing in active_ingredients:
            summary_lines.append(f"- {ing}")
        summary_lines.append("")
    
    if excipients:
        summary_lines.append("Hjálparefni:")
        for exc in excipients:
            summary_lines.append(f"- {exc}")
        summary_lines.append("")
    
    summary_lines.append("Sjá nánari upplýsingar í kafla 2 (Innihaldslýsing) og kafla 6.1 (Hjálparefni).")
    
    summary_text = "\n".join(summary_lines)
    
    # Create section dictionary
    ingredients_section = {
        "number": "ingredients_summary",
        "heading": "Yfirlit innihaldsefna",
        "canonical_key": "ingredients_summary",
        "title": "Yfirlit innihaldsefna",
        "text": summary_text,
        "parent": None,
        "children": [],
        "see_sections": ["2", "6.1"]  # Reference to original sections
    }
    
    logger.info(
        f"Created ingredients summary: {len(active_ingredients)} active ingredients, "
        f"{len(excipients)} excipients"
    )
    
    return ingredients_section


# ---------------------------------------
# Top-Level Builder
# ---------------------------------------

def is_valid_smpc(smpc_data: Dict[str, Any]) -> bool:
    """
    Validate that parsed document has expected SmPC structure.
    
    Checks for:
    - At least one top-level section (1-10)
    - Recognizable section numbering pattern
    - Non-empty sections
    
    Args:
        smpc_data: Dictionary returned by build_smpc_json()
        
    Returns:
        True if document appears to be a valid SmPC, False otherwise
    """
    if not isinstance(smpc_data, dict):
        return False
    
    sections = smpc_data.get("sections", {})
    if not sections:
        return False
    
    # Check for at least one top-level section (1-10)
    top_level_sections = [
        num for num in sections.keys() 
        if num.isdigit() and 1 <= int(num) <= 10
    ]
    
    if not top_level_sections:
        return False
    
    # Check that sections have content (not just empty headings)
    sections_with_content = [
        num for num, sec in sections.items()
        if sec.get("text", "").strip()
    ]
    
    # Should have at least a few sections with content
    if len(sections_with_content) < 2:
        return False
    
    return True


def build_smpc_json(
    pdf_path: str,
    drug_id: Optional[str] = None,
    use_font_detection: bool = True,
    font_size_threshold: float = 1.2,
    use_mistral_ocr: bool = False,
    use_gemini_parser: bool = False
) -> Dict[str, Any]:
    """
    High-level function: parse an SmPC PDF and produce structured JSON.

    Args:
        pdf_path: Path to the SmPC PDF file.
        drug_id: Optional drug identifier. If not provided, uses the PDF filename
            stem (without extension).
        use_font_detection: If True, try font-based detection first (default: True).
            Only used when use_mistral_ocr and use_gemini_parser are False.
        font_size_threshold: Font size threshold multiplier for heading detection (default: 1.2).
            Only used when use_mistral_ocr and use_gemini_parser are False.
        use_mistral_ocr: If True, use Mistral OCR API for extraction instead of PyMuPDF
            (default: False). Ignored if use_gemini_parser is True.
        use_gemini_parser: If True, use Gemini-written parser with blocks-based extraction
            (default: False). Takes precedence over other parsers.

    Returns:
        Dictionary containing:
        - "drug_id": Drug identifier string
        - "source_pdf": Path to source PDF as string
        - "version_hash": MD5 hash of the PDF file
        - "extracted_at": ISO8601 timestamp of extraction
        - "sections": Dictionary of parsed sections (see parse_sections)
        - "validation_report": Dictionary with detection method and validation results

    Raises:
        ValueError: If the PDF path does not exist or is not a file.
        FileNotFoundError: If the PDF file cannot be found during processing.
        RuntimeError: If PDF extraction or hashing fails.
    """
    pdf_path_obj = Path(pdf_path)
    if not pdf_path_obj.exists():
        raise ValueError(f"PDF path does not exist: {pdf_path_obj}")
    if not pdf_path_obj.is_file():
        raise ValueError(f"PDF path is not a file: {pdf_path_obj}")
    
    if drug_id is None:
        drug_id = pdf_path_obj.stem

    # Route to Gemini parser if requested (takes precedence)
    if use_gemini_parser:
        from src.gemini_written_parser import parse_smpc_file
        result = parse_smpc_file(str(pdf_path_obj))
        if result is None:
            raise RuntimeError(f"Failed to parse PDF with Gemini parser: {pdf_path_obj}")
        # Ensure drug_id matches (in case parse_smpc_file derived it differently)
        result["drug_id"] = drug_id
        # Add ingredients summary
        ingredients_summary = create_ingredients_summary(result.get("sections", {}))
        if ingredients_summary:
            result["sections"]["ingredients_summary"] = ingredients_summary
        return result

    # Route to Mistral OCR if requested
    if use_mistral_ocr:
        from src.smpc_extractor_mistral import extract_with_mistral_ocr
        result = extract_with_mistral_ocr(str(pdf_path_obj), drug_id)
        # Add ingredients summary
        ingredients_summary = create_ingredients_summary(result.get("sections", {}))
        if ingredients_summary:
            result["sections"]["ingredients_summary"] = ingredients_summary
        return result
    
    # Existing PyMuPDF path
    raw_text = extract_pdf_text(str(pdf_path_obj))
    sections, validation_report = parse_sections(
        raw_text,
        use_font_detection=use_font_detection,
        pdf_path=str(pdf_path_obj),
        font_size_threshold=font_size_threshold
    )

    logger.info(
        f"Parsed {drug_id}: {len(sections)} sections detected using "
        f"{validation_report['detection_method']} method"
    )

    # Add ingredients summary section
    ingredients_summary = create_ingredients_summary(sections)
    if ingredients_summary:
        sections["ingredients_summary"] = ingredients_summary
        logger.info(f"Added ingredients summary section for {drug_id}")

    data = {
        "drug_id": drug_id,
        "source_pdf": str(pdf_path_obj),
        "version_hash": file_md5(str(pdf_path_obj)),
        "extracted_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sections": sections,
        "validation_report": validation_report,
    }

    return data


# ---------------------------------------
# CLI Runner
# ---------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse Icelandic SmPC PDF into structured JSON"
    )
    parser.add_argument("pdf", help="Path to PDF document")
    parser.add_argument("--out", help="Output JSON path", default=None)
    parser.add_argument("--drug-id", help="Drug ID (defaults to filename stem)", default=None)

    args = parser.parse_args()

    result = build_smpc_json(args.pdf, drug_id=args.drug_id)

    out_path = args.out or f"{Path(args.pdf).stem}_smpc.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved structured SmPC JSON to {out_path}")