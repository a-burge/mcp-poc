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
from pathlib import Path
from typing import Dict, Optional, Any

import fitz  # PyMuPDF


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
HEADING_REGEX = re.compile(
    r"^\s*(\d{1,2}(?:\.\d{1,2})*)\s+(.+)$"
)


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


# ---------------------------------------
# Parsing Logic
# ---------------------------------------

def parse_sections(text: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse plain text of an SmPC into numbered sections.

    Args:
        text: Plain text content of the SmPC document.

    Returns:
        Dictionary keyed by section number (e.g., "1", "4.1", "6.3"). Each value
        is a dictionary containing:
        - "number": Section number string
        - "parent": Parent section number (None for top-level)
        - "heading": Full heading line
        - "title": Title text without number prefix
        - "canonical_key": Canonical key for the section
        - "text": Full text content of the section
        - "children": List of child section numbers
    """
    lines = text.splitlines()

    sections: Dict[str, Dict] = {}
    current_number: Optional[str] = None

    for line in lines:
        stripped = line.rstrip()

        # Check if this line is a heading
        m = HEADING_REGEX.match(stripped)
        if m:
            number = m.group(1)          # e.g. "4.3"
            rest = m.group(2).strip()    # e.g. "Frábendingar" or "KLÍNÍSKAR UPPLÝSINGAR"
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
                # strip leading number from title if it is repeated:
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


def build_smpc_json(pdf_path: str, drug_id: Optional[str] = None) -> Dict[str, Any]:
    """
    High-level function: parse an SmPC PDF and produce structured JSON.

    Args:
        pdf_path: Path to the SmPC PDF file.
        drug_id: Optional drug identifier. If not provided, uses the PDF filename
            stem (without extension).

    Returns:
        Dictionary containing:
        - "drug_id": Drug identifier string
        - "source_pdf": Path to source PDF as string
        - "version_hash": MD5 hash of the PDF file
        - "extracted_at": ISO8601 timestamp of extraction
        - "sections": Dictionary of parsed sections (see parse_sections)

    Raises:
        ValueError: If the PDF path does not exist or is not a file.
        FileNotFoundError: If the PDF file cannot be found during processing.
        RuntimeError: If PDF extraction or hashing fails.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise ValueError(f"PDF path does not exist: {pdf_path}")
    if not pdf_path.is_file():
        raise ValueError(f"PDF path is not a file: {pdf_path}")
    
    if drug_id is None:
        drug_id = pdf_path.stem

    raw_text = extract_pdf_text(str(pdf_path))
    sections = parse_sections(raw_text)

    data = {
        "drug_id": drug_id,
        "source_pdf": str(pdf_path),
        "version_hash": file_md5(str(pdf_path)),
        "extracted_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sections": sections,
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