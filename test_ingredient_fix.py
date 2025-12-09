#!/usr/bin/env python3
"""Quick test to verify the ingredient lookup fix for underscore vs space issue."""

from src.ingredients_manager import IngredientsManager

def test_dicloxacillin_lookup():
    """Test that Dicloxacillin_Bluefish_SmPC can find its ingredients."""
    manager = IngredientsManager()
    
    # This was failing before: underscore vs space mismatch
    ingredients = manager.get_ingredients_for_drug("Dicloxacillin_Bluefish_SmPC")
    
    print(f"Looking up: 'Dicloxacillin_Bluefish_SmPC'")
    print(f"Found ingredients: {ingredients}")
    
    if ingredients:
        print("✅ SUCCESS: Found ingredients for Dicloxacillin")
        return True
    else:
        print("❌ FAILED: No ingredients found")
        return False


def test_ibufen_still_works():
    """Verify Íbúfen still works after the fix."""
    manager = IngredientsManager()
    
    ingredients = manager.get_ingredients_for_drug("Íbúfen_200mg_SmPC")
    
    print(f"\nLooking up: 'Íbúfen_200mg_SmPC'")
    print(f"Found ingredients: {ingredients}")
    
    if ingredients:
        print("✅ SUCCESS: Found ingredients for Íbúfen")
        return True
    else:
        print("❌ FAILED: No ingredients found")
        return False


def test_normalization():
    """Test the normalization function directly."""
    manager = IngredientsManager()
    
    test_cases = [
        ("Dicloxacillin_Bluefish_SmPC", "dicloxacillin bluefish"),
        ("Dicloxacillin Bluefish", "dicloxacillin bluefish"),
        ("Íbúfen_200mg_SmPC", "ibufen"),
        ("Íbúfen", "ibufen"),
    ]
    
    print("\n--- Normalization Tests ---")
    all_passed = True
    for input_val, expected in test_cases:
        result = manager._normalize_drug_id(input_val)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{input_val}' → '{result}' (expected: '{expected}')")
        if result != expected:
            all_passed = False
    
    return all_passed


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Ingredient Lookup Fix")
    print("=" * 60)
    
    test_normalization()
    print()
    
    diclox_ok = test_dicloxacillin_lookup()
    ibufen_ok = test_ibufen_still_works()
    
    print("\n" + "=" * 60)
    if diclox_ok and ibufen_ok:
        print("All tests PASSED! ✅")
    else:
        print("Some tests FAILED! ❌")
    print("=" * 60)
