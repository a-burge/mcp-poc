#!/usr/bin/env python3
"""
Test script for ingredients extraction functionality.
"""
import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import just the function we need (avoiding fitz import)
from smpc_parser import create_ingredients_summary


def test_voltaren():
    """Test extraction with Voltaren JSON."""
    json_path = Path("data/structured/Voltaren_Heilsa_SmPC_SmPC.json")
    
    if not json_path.exists():
        print(f"✗ JSON file not found: {json_path}")
        return False
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Testing ingredients extraction for: {data.get('drug_id', 'Unknown')}")
    print("-" * 60)
    
    # Test extraction
    summary = create_ingredients_summary(data.get('sections', {}))
    
    if not summary:
        print("✗ Failed to create ingredients summary")
        return False
    
    print("✓ Ingredients summary created successfully")
    print()
    print("Summary content:")
    print("=" * 60)
    print(summary['text'])
    print("=" * 60)
    print()
    print(f"Metadata:")
    print(f"  - Section number: {summary.get('number')}")
    print(f"  - Canonical key: {summary.get('canonical_key')}")
    print(f"  - See sections: {summary.get('see_sections', [])}")
    
    return True


def test_voriconazole():
    """Test extraction with Voriconazole JSON."""
    json_path = Path("data/structured/Voriconazole_Normon_SmPC_SmPC.json")
    
    if not json_path.exists():
        print(f"✗ JSON file not found: {json_path}")
        return False
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\nTesting ingredients extraction for: {data.get('drug_id', 'Unknown')}")
    print("-" * 60)
    
    # Test extraction
    summary = create_ingredients_summary(data.get('sections', {}))
    
    if not summary:
        print("✗ Failed to create ingredients summary")
        return False
    
    print("✓ Ingredients summary created successfully")
    print()
    print("Summary content:")
    print("=" * 60)
    print(summary['text'])
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    print("Testing Ingredients Extraction")
    print("=" * 60)
    
    success1 = test_voltaren()
    success2 = test_voriconazole()
    
    if success1 and success2:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed")
        sys.exit(1)
