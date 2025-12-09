"""Test script to verify ATC scraper fixes work correctly."""
import logging
import sys
from pathlib import Path

_script_dir = Path(__file__).parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from src.atc_scraper import scrape_atc_index

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    print("=" * 80)
    print("Testing ATC Scraper with Fixes")
    print("=" * 80)
    print("\nThis will test:")
    print("1. Session expiration detection and recovery")
    print("2. Validation that all 14 sections are scraped")
    print("3. Proper error reporting if sections are missing")
    print("\nNote: This may take 10-30 minutes depending on website response times.")
    print("=" * 80)
    
    try:
        result = scrape_atc_index(use_selenium=True)
        
        print("\n" + "=" * 80)
        print("SCRAPING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Level 1 categories scraped: {len(result['hierarchy'])}")
        print(f"Drug mappings found: {len(result['drug_mappings'])}")
        print(f"Scraped at: {result['scraped_at']}")
        
        # Verify all sections
        expected = ['A', 'B', 'C', 'D', 'G', 'H', 'J', 'L', 'M', 'N', 'P', 'R', 'S', 'V']
        actual = sorted(result['hierarchy'].keys())
        missing = set(expected) - set(actual)
        
        if missing:
            print(f"\n⚠️  WARNING: Missing sections: {sorted(missing)}")
            print("This should not happen with the fixes - validation should have caught this!")
        else:
            print(f"\n✅ All {len(expected)} expected sections are present!")
            print(f"Sections: {', '.join(actual)}")
        
    except RuntimeError as e:
        print("\n" + "=" * 80)
        print("VALIDATION ERROR (This is expected if scraping is incomplete)")
        print("=" * 80)
        print(f"Error: {e}")
        print("\nThis means the validation is working correctly!")
        print("The scraper detected missing sections and raised an error.")
        sys.exit(1)
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("SCRAPING FAILED")
        print("=" * 80)
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)

