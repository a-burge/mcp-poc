"""
Diagnostic script to understand why a specific PDF fails SmPC validation.

This script runs a PDF through the same parsing and validation logic as
ingest_all_smpcs.py and reports detailed information about why it fails.
Includes font analysis and heading detection method information.
"""
import json
from pathlib import Path
from typing import Dict, Any, List
from collections import Counter

from config import Config
from src.smpc_parser import (
    build_smpc_json,
    is_valid_smpc,
    extract_pdf_text_with_structure,
    get_expected_sections
)


def diagnose_validation(pdf_filename: str) -> None:
    """
    Diagnose why a PDF fails SmPC validation.
    
    Args:
        pdf_filename: Name of the PDF file to diagnose (e.g., 'Íbúfen_600mg_SmPC.pdf')
    """
    print("=" * 80)
    print(f"Diagnosing SmPC validation for: {pdf_filename}")
    print("=" * 80)
    print()
    
    # Find the PDF file
    source_dir = Config.RAW_SOURCE_DOCS_DIR
    pdf_path = source_dir / pdf_filename
    
    if not pdf_path.exists():
        print(f"❌ ERROR: PDF file not found at: {pdf_path}")
        print(f"   Searched in: {source_dir}")
        print(f"   Available PDFs in directory:")
        if source_dir.exists():
            for f in sorted(source_dir.glob("*.pdf")):
                print(f"     - {f.name}")
        return
    
    print(f"✓ Found PDF at: {pdf_path}")
    print()
    
    # Step 1: Parse the PDF
    print("Step 1: Parsing PDF...")
    try:
        smpc_data = build_smpc_json(str(pdf_path))
        print(f"✓ PDF parsed successfully")
        print(f"  Drug ID: {smpc_data.get('drug_id', 'N/A')}")
        print(f"  Source PDF: {smpc_data.get('source_pdf', 'N/A')}")
        
        # Show detection method and validation info
        validation_report = smpc_data.get("validation_report", {})
        detection_method = validation_report.get("detection_method", "unknown")
        is_light_touch = validation_report.get("is_light_touch", False)
        
        print(f"  Detection method: {detection_method.upper()}")
        print(f"  Document type: {'LIGHT TOUCH' if is_light_touch else 'FULL SmPC'}")
        print()
        
        # Font analysis
        print("Step 1a: Font size analysis...")
        try:
            structured_lines = extract_pdf_text_with_structure(str(pdf_path))
            
            font_sizes = [size for _, _, size, _ in structured_lines]
            headings_by_font = [
                (line, num, size, is_bold)
                for line, num, size, is_bold in structured_lines
                if num is not None
            ]
            
            if font_sizes:
                sorted_sizes = sorted(font_sizes)
                median_size = sorted_sizes[len(sorted_sizes) // 2]
                threshold = median_size * 1.2
                
                print(f"  ✓ Font size statistics:")
                print(f"    Median: {median_size:.1f}pt")
                print(f"    Threshold: {threshold:.1f}pt")
                print(f"    Headings detected by font: {len(headings_by_font)}")
                
                if headings_by_font:
                    print(f"    Sample headings (first 5):")
                    for line, num, size, is_bold in headings_by_font[:5]:
                        bold_marker = " (bold)" if is_bold else ""
                        print(f"      {num:8s} ({size:.1f}pt{bold_marker}): {line[:60]}")
            print()
        except Exception as e:
            print(f"  ⚠️  Font analysis failed: {e}")
            print()
        
    except Exception as e:
        print(f"❌ ERROR parsing PDF: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: Check validation step by step
    print("Step 2: Validating SmPC structure...")
    print()
    
    # Show validation report if available
    validation_report = smpc_data.get("validation_report", {})
    if validation_report:
        expected = get_expected_sections()
        detected_top = validation_report.get("detected_top_level", [])
        detected_sub = validation_report.get("detected_subsections", [])
        missing_top = validation_report.get("missing_top_level", [])
        missing_sub = validation_report.get("missing_subsections", [])
        is_light_touch = validation_report.get("is_light_touch", False)
        
        print("Validation Report Summary:")
        print(f"  Detection method: {validation_report.get('detection_method', 'unknown').upper()}")
        print(f"  Document type: {'LIGHT TOUCH' if is_light_touch else 'FULL SmPC'}")
        print()
        print(f"  Expected top-level sections: {len(expected['top_level'])}")
        print(f"  Detected top-level sections: {len(detected_top)}")
        print(f"  Missing top-level sections: {len(missing_top)}")
        if missing_top:
            print(f"    Missing: {', '.join(missing_top)}")
        print()
        print(f"  Expected subsections: {len(expected['subsections'])}")
        print(f"  Detected subsections: {len(detected_sub)}")
        print(f"  Missing subsections: {len(missing_sub)}")
        if missing_sub and len(missing_sub) <= 10:
            print(f"    Missing: {', '.join(missing_sub)}")
        elif missing_sub:
            print(f"    Missing: {', '.join(missing_sub[:10])} ... and {len(missing_sub) - 10} more")
        print()
        if is_light_touch:
            print("  ⚠️  NOTE: This appears to be a 'light touch' document")
            print("     (references another SmPC, has fewer sections)")
        print()
    
    # Check 1: Is it a dict?
    print("Check 1: Is smpc_data a dictionary?")
    is_dict = isinstance(smpc_data, dict)
    print(f"  Result: {'✓ YES' if is_dict else '❌ NO'}")
    if not is_dict:
        print(f"  Type: {type(smpc_data)}")
        print()
        return
    print()
    
    # Check 2: Does it have sections?
    print("Check 2: Does it have a 'sections' key?")
    sections = smpc_data.get("sections", {})
    has_sections = bool(sections)
    print(f"  Result: {'✓ YES' if has_sections else '❌ NO'}")
    if not has_sections:
        print(f"  Available keys: {list(smpc_data.keys())}")
        print()
        return
    
    print(f"  Number of sections found: {len(sections)}")
    print(f"  Section numbers: {sorted(sections.keys())}")
    print()
    
    # Check 3: Does it have top-level sections (1-10)?
    print("Check 3: Does it have at least one top-level section (1-10)?")
    top_level_sections = [
        num for num in sections.keys() 
        if num.isdigit() and 1 <= int(num) <= 10
    ]
    has_top_level = bool(top_level_sections)
    print(f"  Result: {'✓ YES' if has_top_level else '❌ NO'}")
    
    if top_level_sections:
        print(f"  Top-level sections found: {sorted(top_level_sections, key=int)}")
    else:
        print(f"  No top-level sections found!")
        print(f"  All section numbers: {sorted(sections.keys())}")
        print()
        print("  This is likely why validation failed.")
        print("  The parser expects sections numbered 1-10 (e.g., '1', '4', '6').")
        print("  It may have found subsections (e.g., '4.1', '4.2') but no top-level sections.")
    print()
    
    # Check 4: Do sections have content?
    print("Check 4: Do sections have non-empty text content?")
    sections_with_content = [
        (num, sec) for num, sec in sections.items()
        if sec.get("text", "").strip()
    ]
    has_content = len(sections_with_content) >= 2
    print(f"  Result: {'✓ YES' if has_content else '❌ NO'}")
    print(f"  Sections with content: {len(sections_with_content)} (need at least 2)")
    
    if sections_with_content:
        print(f"  Sections with content:")
        for num, sec in sections_with_content[:10]:  # Show first 10
            text_preview = sec.get("text", "")[:50].replace("\n", " ")
            print(f"    - {num}: '{sec.get('title', 'N/A')}' ({len(sec.get('text', ''))} chars)")
            if text_preview:
                print(f"      Preview: {text_preview}...")
    else:
        print(f"  No sections with content found!")
        print()
        print("  This is likely why validation failed.")
        print("  Sections were found but they have no text content.")
    
    # Show sections without content
    sections_without_content = [
        (num, sec) for num, sec in sections.items()
        if not sec.get("text", "").strip()
    ]
    if sections_without_content:
        print()
        print(f"  Sections WITHOUT content ({len(sections_without_content)}):")
        for num, sec in sections_without_content[:10]:
            print(f"    - {num}: '{sec.get('title', 'N/A')}' (heading only, no text)")
    print()
    
    # Final validation result
    print("=" * 80)
    print("Final Validation Result:")
    is_valid = is_valid_smpc(smpc_data)
    print(f"  is_valid_smpc() returns: {'✓ TRUE' if is_valid else '❌ FALSE'}")
    print()
    
    if not is_valid:
        print("Summary of why validation failed:")
        if not has_sections:
            print("  ❌ No sections found in parsed data")
            print()
            print("  Possible causes:")
            print("    1. Heading detection failed (check detection method above)")
            print("    2. PDF text extraction issues")
            print("    3. Document may not be a standard SmPC format")
        elif not has_top_level:
            print("  ❌ No top-level sections (1-10) found")
            print()
            print("  Possible causes:")
            print("    1. Headings may use non-standard numbering")
            print("    2. Font detection may have failed (try regex fallback)")
            print("    3. Document may be a 'light touch' type (check validation report)")
        elif not has_content:
            print("  ❌ Less than 2 sections have text content")
            print()
            print("  Possible causes:")
            print("    1. Sections detected but text extraction failed")
            print("    2. Document may have only headings without content")
        print()
        
        # Show sample of what was parsed
        print("Sample of parsed sections (first 5):")
        for i, (num, sec) in enumerate(list(sections.items())[:5]):
            print(f"\n  Section {num}:")
            print(f"    Title: {sec.get('title', 'N/A')}")
            print(f"    Heading: {sec.get('heading', 'N/A')}")
            print(f"    Parent: {sec.get('parent', 'N/A')}")
            print(f"    Text length: {len(sec.get('text', ''))} chars")
            if sec.get('text'):
                preview = sec.get('text', '')[:100].replace('\n', ' ')
                print(f"    Text preview: {preview}...")
            else:
                print(f"    Text: (empty)")
    
    print("=" * 80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python diagnose_smpc_validation.py <pdf_filename>")
        print()
        print("Example:")
        print("  python diagnose_smpc_validation.py 'Íbúfen_600mg_SmPC.pdf'")
        sys.exit(1)
    
    pdf_filename = sys.argv[1]
    diagnose_validation(pdf_filename)
