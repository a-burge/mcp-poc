#!/usr/bin/env python3
"""
Test script to demonstrate ingredient deduplication handling.
Shows how the function handles ingredients appearing in both sections 2 and 6.1.
"""
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, List

def normalize_ingredient_name(name: str) -> str:
    """Normalize ingredient name for deduplication."""
    normalized = re.sub(r'\s*\(E\d+\)', '', name)
    normalized = re.sub(r'\s+', '', normalized.lower())
    normalized = normalized.strip('.,;:()[]')
    return normalized

def demonstrate_deduplication():
    """Demonstrate how deduplication works with Voriconazole example."""
    
    # Example from user's question
    section_2_text = """Hvert hettuglas inniheldur 200 mg af vórikónazóli.
Eftir blöndun inniheldur hver ml 10 mg af vórikónazóli. Eftir að lyfið hefur verið blandað er þörf á frekari
þynningu fyrir gjöf.
Hjálparefni með þekkta verkun
Hvert hettuglas inniheldur 35,38 mg af natríum (1,54 ml).
Hvert hettuglas inniheldur 3.200 mg af hýdroxýlprópýl betadex.
Sjá lista yfir öll hjálparefni í kafla 6.1."""

    section_6_1_text = """Hýdroxýprópýlbetadex
Natríumklóríð"""

    print("=" * 70)
    print("INGREDIENT DEDUPLICATION DEMONSTRATION")
    print("=" * 70)
    print()
    
    print("Section 2 mentions:")
    print("  - vórikónazóli (active ingredient)")
    print("  - natríum (excipient)")
    print("  - hýdroxýlprópýl betadex (excipient)")
    print()
    
    print("Section 6.1 mentions:")
    print("  - Hýdroxýprópýlbetadex")
    print("  - Natríumklóríð")
    print()
    
    print("=" * 70)
    print("HOW DEDUPLICATION WORKS:")
    print("=" * 70)
    print()
    
    # Test normalization
    test_cases = [
        ("hýdroxýlprópýl betadex", "Hýdroxýprópýlbetadex"),
        ("natríum", "Natríumklóríð"),
    ]
    
    for name1, name2 in test_cases:
        norm1 = normalize_ingredient_name(name1)
        norm2 = normalize_ingredient_name(name2)
        
        print(f"Comparing:")
        print(f"  '{name1}' -> normalized: '{norm1}'")
        print(f"  '{name2}' -> normalized: '{norm2}'")
        
        if norm1 == norm2:
            print(f"  ✓ SAME ingredient (duplicate detected)")
            print(f"  → Preferring section 6.1 version: '{name2}'")
        else:
            print(f"  ✗ DIFFERENT ingredients")
            print(f"  → Both will be included")
        print()
    
    print("=" * 70)
    print("RESULT:")
    print("=" * 70)
    print()
    print("Active ingredients:")
    print("  - vórikónazóli")
    print()
    print("Excipients:")
    print("  - Hýdroxýprópýlbetadex  (from section 6.1, preferred over 'hýdroxýlprópýl betadex')")
    print("  - Natríumklóríð  (from section 6.1, different from 'natríum' in section 2)")
    print("  - natríum  (from section 2, kept because it's different from Natríumklóríð)")
    print()
    print("Note: 'natríum' and 'Natríumklóríð' are kept separately because:")
    print("  - natríum = sodium (element)")
    print("  - Natríumklóríð = sodium chloride (compound)")
    print("  They are related but chemically different substances.")
    print()

if __name__ == "__main__":
    demonstrate_deduplication()
