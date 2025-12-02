"""
Debug script to understand how PDF text is extracted and how headings are detected.

This shows the raw text extraction and which lines match the heading regex pattern.
Also includes font size analysis and comparison between font-based and regex detection.
"""
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any
from collections import Counter

from config import Config
from src.smpc_parser import (
    extract_pdf_text,
    extract_pdf_text_with_structure,
    HEADING_REGEX,
    get_expected_sections,
    parse_sections,
    build_smpc_json
)


def analyze_pdf_text(
    pdf_filename: str,
    show_lines: int = 200,
    use_font_detection: bool = False
) -> None:
    """
    Analyze how a PDF's text is extracted and which lines match heading patterns.
    
    Args:
        pdf_filename: Name of the PDF file to analyze
        show_lines: Number of lines to show in the output
    """
    print("=" * 80)
    print(f"Analyzing PDF text extraction for: {pdf_filename}")
    print("=" * 80)
    print()
    
    # Find the PDF file
    source_dir = Config.RAW_SOURCE_DOCS_DIR
    pdf_path = source_dir / pdf_filename
    
    if not pdf_path.exists():
        print(f"❌ ERROR: PDF file not found at: {pdf_path}")
        return
    
    print(f"✓ Found PDF at: {pdf_path}")
    print(f"  Font detection: {'ENABLED' if use_font_detection else 'DISABLED'}")
    print()
    
    # Extract raw text
    print("Step 1: Extracting raw text from PDF...")
    try:
        raw_text = extract_pdf_text(str(pdf_path))
        lines = raw_text.splitlines()
        original_line_count = len(lines)
        
        # Normalize lines: merge split headings like "1." + "HEITI LYFS"
        normalized_lines = []
        normalization_examples = []
        skip_next = False
        
        for i in range(len(lines)):
            if skip_next:
                skip_next = False
                continue
            
            current = lines[i].strip()
            # Detect numeric-only heading number (with optional trailing period)
            # Pattern matches: "1", "1.", "6.1", "6.1.", etc.
            if re.fullmatch(r"\d{1,2}(?:\.\d{1,2})*\.?", current):
                if i + 1 < len(lines):
                    nxt = lines[i+1].strip()
                    # If next line looks like a heading title
                    if re.match(r"^[A-Za-zÁÉÍÓÚÝÞÆÖ]", nxt):
                        combined = f"{current} {nxt}"
                        normalized_lines.append(combined)
                        normalization_examples.append((current, nxt, combined))
                        skip_next = True
                        continue
            
            normalized_lines.append(current)
        
        lines = normalized_lines
        normalized_count = len(normalization_examples)
        
        print(f"✓ Extracted {original_line_count} lines from PDF")
        print(f"✓ After normalization: {len(lines)} lines")
        if normalized_count > 0:
            print(f"✓ Normalized {normalized_count} split headings")
            print(f"  Examples:")
            for num_part, title_part, combined in normalization_examples[:5]:
                print(f"    '{num_part}' + '{title_part}' -> '{combined}'")
            if len(normalization_examples) > 5:
                print(f"    ... and {len(normalization_examples) - 5} more")
        print(f"  Total characters: {len(raw_text)}")
        print()
    except Exception as e:
        print(f"❌ ERROR extracting text: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Font size analysis (if font detection enabled)
    font_analysis = None
    if use_font_detection:
        print("Step 2: Font size analysis...")
        print()
        try:
            structured_lines = extract_pdf_text_with_structure(str(pdf_path))
            
            # Collect font size statistics
            font_sizes = [size for _, _, size, _ in structured_lines]
            bold_lines = [line for line, _, _, is_bold in structured_lines if is_bold]
            headings_by_font = [
                (line, num, size, is_bold)
                for line, num, size, is_bold in structured_lines
                if num is not None
            ]
            
            if font_sizes:
                sorted_sizes = sorted(font_sizes)
                median_size = sorted_sizes[len(sorted_sizes) // 2]
                min_size = min(font_sizes)
                max_size = max(font_sizes)
                threshold = median_size * 1.2
                
                # Font size distribution
                size_counter = Counter(int(size) for size in font_sizes)
                
                print(f"✓ Font size statistics:")
                print(f"  Min: {min_size:.1f}pt")
                print(f"  Max: {max_size:.1f}pt")
                print(f"  Median: {median_size:.1f}pt")
                print(f"  Threshold (1.2x median): {threshold:.1f}pt")
                print(f"  Bold lines: {len(bold_lines)}")
                print(f"  Headings detected by font: {len(headings_by_font)}")
                print()
                
                print(f"  Font size distribution (top 10):")
                for size, count in size_counter.most_common(10):
                    marker = ">>>" if size >= threshold else "   "
                    print(f"    {marker} {size:4.0f}pt: {count:4d} lines")
                print()
                
                font_analysis = {
                    "median_size": median_size,
                    "threshold": threshold,
                    "headings_by_font": headings_by_font,
                    "font_sizes": font_sizes
                }
        except Exception as e:
            print(f"⚠️  Font analysis failed: {e}")
            print()
    
    # Analyze heading detection
    print("Step 3: Analyzing heading detection (regex method)...")
    print()
    
    # Find all lines that match the heading regex
    matched_headings: List[Tuple[int, str, str]] = []
    potential_headings: List[Tuple[int, str]] = []
    
    for i, line in enumerate(lines):
        stripped = line.rstrip()
        
        # Check if it matches the regex
        m = HEADING_REGEX.match(stripped)
        if m:
            number = m.group(1)
            title = m.group(2).strip()
            matched_headings.append((i + 1, number, title))
        
        # Also look for lines that start with numbers (potential headings that don't match)
        if stripped.strip():
            # Check if line starts with a number pattern
            number_pattern = re.match(r'^\s*(\d{1,2}(?:\.\d{1,2})*)', stripped)
            if number_pattern and not HEADING_REGEX.match(stripped):
                potential_headings.append((i + 1, stripped[:80]))
    
    print(f"✓ Found {len(matched_headings)} lines that match heading regex")
    print(f"✓ Found {len(potential_headings)} lines that start with numbers but DON'T match regex")
    print()
    
    # Compare font vs regex detection
    if font_analysis:
        print("=" * 80)
        print("COMPARISON: Font Detection vs Regex Detection")
        print("=" * 80)
        
        font_headings = {num: (line, size, is_bold) for line, num, size, is_bold in font_analysis["headings_by_font"] if num}
        regex_headings = {num: (line_num, title) for line_num, num, title in matched_headings}
        
        font_only = set(font_headings.keys()) - set(regex_headings.keys())
        regex_only = set(regex_headings.keys()) - set(font_headings.keys())
        both = set(font_headings.keys()) & set(regex_headings.keys())
        
        print(f"  Headings found by BOTH methods: {len(both)}")
        print(f"  Headings found by FONT only: {len(font_only)}")
        print(f"  Headings found by REGEX only: {len(regex_only)}")
        print()
        
        if font_only:
            print("  Headings detected by font but not regex:")
            for num in sorted(font_only, key=lambda x: (len(x.split('.')), x)):
                line, size, is_bold = font_headings[num]
                print(f"    {num:8s} (font: {size:.1f}pt, bold: {is_bold}) -> {line[:60]}")
            print()
        
        if regex_only:
            print("  Headings detected by regex but not font:")
            for num in sorted(regex_only, key=lambda x: (len(x.split('.')), x)):
                line_num, title = regex_headings[num]
                print(f"    {num:8s} (line {line_num}) -> {title[:60]}")
            print()
    
    # Show matched headings
    print("=" * 80)
    print("HEADINGS THAT MATCH THE REGEX (these will be detected as sections):")
    print("=" * 80)
    if matched_headings:
        for line_num, number, title in matched_headings:
            print(f"  Line {line_num:4d}: {number:8s} -> {title}")
    else:
        print("  ❌ NO HEADINGS MATCHED THE REGEX!")
    print()
    
    # Expected sections validation
    print("=" * 80)
    print("EXPECTED SECTIONS VALIDATION")
    print("=" * 80)
    try:
        smpc_data = build_smpc_json(
            str(pdf_path),
            use_font_detection=use_font_detection
        )
        validation_report = smpc_data.get("validation_report", {})
        expected = get_expected_sections()
        
        detected_top = validation_report.get("detected_top_level", [])
        detected_sub = validation_report.get("detected_subsections", [])
        missing_top = validation_report.get("missing_top_level", [])
        missing_sub = validation_report.get("missing_subsections", [])
        is_light_touch = validation_report.get("is_light_touch", False)
        detection_method = validation_report.get("detection_method", "unknown")
        
        print(f"  Detection method used: {detection_method.upper()}")
        print(f"  Document type: {'LIGHT TOUCH' if is_light_touch else 'FULL SmPC'}")
        print()
        print(f"  Top-level sections:")
        print(f"    Expected: {len(expected['top_level'])}")
        print(f"    Detected: {len(detected_top)}")
        print(f"    Missing: {len(missing_top)}")
        if missing_top:
            print(f"      {', '.join(missing_top)}")
        print()
        print(f"  Subsections:")
        print(f"    Expected: {len(expected['subsections'])}")
        print(f"    Detected: {len(detected_sub)}")
        print(f"    Missing: {len(missing_sub)}")
        if missing_sub and len(missing_sub) <= 20:
            print(f"      {', '.join(missing_sub)}")
        elif missing_sub:
            print(f"      {', '.join(missing_sub[:20])} ... and {len(missing_sub) - 20} more")
        print()
    except Exception as e:
        print(f"  ⚠️  Validation failed: {e}")
        import traceback
        traceback.print_exc()
        print()
    
    # Show potential headings that didn't match
    if potential_headings:
        print("=" * 80)
        print("LINES THAT START WITH NUMBERS BUT DON'T MATCH REGEX:")
        print("=" * 80)
        print("  (These might be headings with formatting issues)")
        for line_num, line_preview in potential_headings[:20]:  # Show first 20
            print(f"  Line {line_num:4d}: {line_preview}")
        if len(potential_headings) > 20:
            print(f"  ... and {len(potential_headings) - 20} more")
        print()
    
    # Show raw text sample
    print("=" * 80)
    print(f"RAW EXTRACTED TEXT (first {show_lines} lines):")
    print("=" * 80)
    print()
    for i, line in enumerate(lines[:show_lines], 1):
        # Highlight lines that match heading regex
        is_heading = HEADING_REGEX.match(line.rstrip()) is not None
        marker = ">>> " if is_heading else "    "
        # Show line with visible whitespace
        line_display = repr(line) if line.strip() else "(empty line)"
        print(f"{marker}Line {i:4d}: {line_display}")
    
    if len(lines) > show_lines:
        print(f"\n  ... ({len(lines) - show_lines} more lines)")
    
    print()
    print("=" * 80)
    print("ANALYSIS:")
    print("=" * 80)
    print()
    
    if not matched_headings:
        print("❌ PROBLEM: No headings matched the regex pattern!")
        print()
        print("The regex pattern is:")
        print(f"  {HEADING_REGEX.pattern}")
        print()
        print("This pattern expects headings like:")
        print("  - '1. HEITI LYFS'")
        print("  - '4.1 Ábendingar'")
        print("  - '6.3 Geymsluþol'")
        print()
        print("Possible issues:")
        print("  1. Headings might have extra spaces or tabs")
        print("  2. Headings might be on multiple lines")
        print("  3. Headings might use different number formats")
        print("  4. PDF text extraction might be splitting headings incorrectly")
        print()
        if potential_headings:
            print("Look at the 'potential headings' above to see what the actual format is.")
    elif len(matched_headings) < 5:
        print(f"⚠️  WARNING: Only {len(matched_headings)} headings detected!")
        print("   A typical SmPC should have 10+ sections.")
        print("   Check if the regex pattern needs adjustment.")
    else:
        print(f"✓ Found {len(matched_headings)} headings - this looks reasonable.")
    
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_pdf_parsing.py <pdf_filename> [num_lines] [--use-font-detection]")
        print()
        print("Example:")
        print("  python debug_pdf_parsing.py 'Íbúfen_600mg_SmPC.pdf'")
        print("  python debug_pdf_parsing.py 'Íbúfen_600mg_SmPC.pdf' 300")
        print("  python debug_pdf_parsing.py 'Íbúfen_600mg_SmPC.pdf' 300 --use-font-detection")
        sys.exit(1)
    
    pdf_filename = sys.argv[1]
    num_lines = 200
    use_font_detection = False
    
    # Parse arguments
    for arg in sys.argv[2:]:
        if arg == "--use-font-detection":
            use_font_detection = True
        elif arg.isdigit():
            num_lines = int(arg)
    
    analyze_pdf_text(
        pdf_filename,
        show_lines=num_lines,
        use_font_detection=use_font_detection
    )