"""Test strict SmPC validation with known files."""
from pathlib import Path
from src.smpc_parser import build_smpc_json, is_valid_smpc

# Test files from sample results
test_cases = [
    # True SmPC files (should PASS)
    ("Ritalin_SmPC.pdf", True, "True SmPC"),
    ("Haldol_5mgml_SmPC.pdf", True, "True SmPC"),
    ("Trilafon-Dekanoat_SmPC.pdf", True, "True SmPC"),
    ("Abilify_Maintena_Lyfjaver_SmPC.pdf", True, "True SmPC"),
    
    # Fylgisedill files (should FAIL)
    ("Hypotron_Fylgiseðill.pdf", False, "Package leaflet"),
    ("Testogel-(Heilsa)_50mg_Fylgisedill.pdf", False, "Package leaflet"),
    ("Lident_Fylgisedill.pdf", False, "Package leaflet"),
    
    # Product-information files (should FAIL)
    ("zessly-epar-product-information_is.pdf", False, "Product information ad"),
    ("arava-epar-product-information_is.pdf", False, "Product information ad"),
    ("tivicay-epar-product-information_is.pdf", False, "Product information ad"),
]

source_dir = Path("data/raw_source_docs")

print("Testing Strict SmPC Validation")
print("=" * 80)
print()

passed = 0
failed = 0

for filename, expected_valid, file_type in test_cases:
    pdf_path = source_dir / filename
    
    if not pdf_path.exists():
        print(f"SKIP: {filename} (file not found)")
        continue
    
    try:
        smpc_data = build_smpc_json(str(pdf_path))
        is_valid = is_valid_smpc(smpc_data, pdf_path=str(pdf_path))
        
        status = "✓ PASS" if is_valid == expected_valid else "✗ FAIL"
        if is_valid == expected_valid:
            passed += 1
        else:
            failed += 1
        
        print(f"{status} | {filename}")
        print(f"      Expected: {'VALID' if expected_valid else 'INVALID'} ({file_type})")
        print(f"      Got: {'VALID' if is_valid else 'INVALID'}")
        
        if is_valid != expected_valid:
            # Show why it failed/passed
            sections = smpc_data.get("sections", {})
            section_1 = sections.get("1", {}).get("heading", "N/A")
            print(f"      Section 1 heading: {section_1[:60]}")
        
        print()
        
    except Exception as e:
        print(f"ERROR: {filename} - {e}")
        print()
        failed += 1

print("=" * 80)
print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
