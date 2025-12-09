# ATC Scraper Documentation

## Purpose

The ATC scraper extracts the complete **Anatomical Therapeutic Chemical (ATC) classification system** from the Icelandic Medicines Agency website (`https://old.serlyfjaskra.is/ATCList.aspx?d=1&a=0`). 

The ATC system is a 5-level hierarchical classification where:
- **Level 1**: Anatomical main group (single letter: A, B, C, D, G, H, J, L, M, N, P, R, S, V)
- **Level 2**: Therapeutic subgroup (2 digits, e.g., "A01")
- **Level 3**: Therapeutic/pharmacological subgroup (1 letter, e.g., "A01A")
- **Level 4**: Chemical/therapeutic/pharmacological subgroup (2 letters, e.g., "A01AA")
- **Level 5**: Chemical substance (2 digits, e.g., "A01AA01")

At Level 5, the scraper also extracts individual **drug products** (brand names) that belong to each ATC code, along with their form/strength and document links (SmPC, Fylgiseðill).

## Files and Dependencies

### Core Files

1. **`src/atc_scraper.py`** (868 lines)
   - Main scraper implementation
   - Contains `ATCScraper` class with all scraping logic
   - Entry point: `scrape_atc_index(use_selenium: bool = False)`

2. **`config.py`**
   - Configuration management
   - Defines `Config.ATC_INDEX_PATH` = `data/atc/atc_index.json`
   - Defines `Config.ATC_DATA_DIR` = `data/atc/`
   - Provides `Config.ensure_directories()` to create output directories

3. **`src/atc_manager.py`**
   - Consumes the scraped data
   - Loads JSON from `Config.ATC_INDEX_PATH`
   - Provides utilities for RAG integration and ATC code lookups

4. **`src/atc_matcher.py`**
   - Uses scraped data to match drugs to ATC codes
   - Referenced by `enrich_with_atc.py` and `enrich_all_with_atc.py`

### Test Files

- **`test_atc_simple.py`**: Simple test scraping just category "A"
- **`test_atc_scraper_debug.py`**: Debug script for troubleshooting

### Dependencies

From `requirements.txt`:
- `requests>=2.32.5,<3.0` - HTTP requests
- `beautifulsoup4` (via `bs4` import) - HTML parsing (not in requirements.txt, but used)
- `selenium` - Browser automation (required for actual scraping, not in requirements.txt)

**Note**: The scraper requires Selenium because the website uses ASP.NET with JavaScript/AJAX for dynamic content loading. The basic `scrape()` method (without Selenium) is incomplete and not functional.

## Execution Flow

### Entry Point

```python
# From command line:
python src/atc_scraper.py --selenium

# Or programmatically:
from src.atc_scraper import scrape_atc_index
data = scrape_atc_index(use_selenium=True)
```

### Main Execution Steps

1. **Initialization** (`ATCScraper.__init__`)
   - Creates `requests.Session` with User-Agent header
   - Initializes `self.hierarchy: Dict[str, Any] = {}`
   - Initializes `self.drug_mappings: Dict[str, List[str]] = {}`

2. **Selenium Setup** (`scrape_with_selenium`)
   - Configures Chrome in headless mode
   - Navigates to `https://old.serlyfjaskra.is/ATCList.aspx?d=1&a=0`
   - Waits for page load

3. **Level 1 Navigation** (lines 177-248)
   - Iterates through 14 level 1 codes: `['A', 'B', 'C', 'D', 'G', 'H', 'J', 'L', 'M', 'N', 'P', 'R', 'S', 'V']`
   - For each code:
     - Finds and clicks the letter link (e.g., "A")
     - Waits for result grid table to appear
     - Extracts the full hierarchy starting from level 2

4. **Recursive Hierarchy Extraction** (`_extract_level_recursive`, lines 263-601)
   - **Depth-first traversal** through all 5 levels
   - For each level:
     - Finds all data rows in the current table (XPath: `.//tbody/tr[(contains(@class, 'rgRow') or contains(@class, 'rgAltRow')) and td[4]]`)
     - Extracts ATC code from column 4 (`td[4]`)
     - Extracts name from column 5 (`td[5]`)
     - Checks if row has children (column 3 contains count > 0, or expand button exists)
     - If has children:
       - Clicks expand button (`.//td[1]//input[contains(@class, 'rgExpand')]`)
       - Waits for detail table to appear (following sibling `<tr>` with `rgDetailTable`)
       - Recursively processes children at next level
     - If level 5: extracts drugs using `_extract_drugs_from_level5`

5. **Drug Extraction** (`_extract_drugs_from_level5`, lines 603-808)
   - Expands level 5 row if not already expanded
   - Finds detail table containing drug rows
   - For each drug row:
     - Extracts drug name from column 3 (link with class `productlink`)
     - Extracts form/strength from column 4
     - Extracts document links from column 5 (SmPC, Fylgiseðill PDFs)
   - Stores in `self.drug_mappings` dictionary

6. **Data Persistence** (`save`, lines 810-831)
   - Saves to JSON: `data/atc/atc_index.json`
   - Structure:
     ```json
     {
       "hierarchy": { ... },
       "drug_mappings": { ... },
       "scraped_at": "2024-12-04T23:19:31"
     }
     ```

## Technical Details

### Website Structure

The website uses **ASP.NET with Telerik RadGrid** controls:
- Main result grid: `table[contains(@id, 'resultGrid') and contains(@class, 'rgMasterTable')]`
- Detail tables (nested): `table[contains(@class, 'rgDetailTable')]`
- Rows: `tr[contains(@class, 'rgRow') or contains(@class, 'rgAltRow')]`
- Expand/collapse buttons: `input[contains(@class, 'rgExpand') or contains(@class, 'rgCollapse')]`

### Table Column Structure

- **Column 1** (`td[1]`): Expand/collapse button
- **Column 2** (`td[2]`): Hidden/empty
- **Column 3** (`td[3]`): Child count (for hierarchy rows) or drug name link (for drug rows)
- **Column 4** (`td[4]`): ATC code (for hierarchy rows) or form/strength (for drug rows)
- **Column 5** (`td[5]`): Name (for hierarchy rows) or document links (for drug rows)

### Key Challenges Handled

1. **Stale Element References**: The DOM changes when rows are expanded, so elements become stale. The code:
   - Re-finds parent elements when they become stale
   - Uses row IDs to re-locate elements after DOM changes
   - Implements multiple fallback methods for finding detail tables

2. **AJAX Loading**: Expansion triggers AJAX requests. The code:
   - Uses `WebDriverWait` with expected conditions
   - Falls back to `time.sleep()` if waits timeout
   - Waits for detail tables to appear before processing

3. **Depth-First Traversal**: The scraper processes each branch completely before moving to the next:
   - Processes one row at a time
   - Fully expands and processes all children before moving to next sibling
   - Maintains `processed_codes` set to avoid duplicates

4. **Level 5 Drug Extraction**: Critical to find the correct detail table:
   - Uses row ID to find following sibling with detail table
   - Multiple fallback methods (3 different XPath strategies)
   - **Important**: Removed fallback that searched all detail tables (would assign wrong drugs to wrong ATC codes)

### Data Structure

#### Hierarchy Structure
```python
{
  "A": {
    "code": "A",
    "name": "Alimentary tract and metabolism",
    "level2": {
      "A01": {
        "code": "A01",
        "name": "MUNN- OG TANNLYF",
        "level3": {
          "A01A": {
            "code": "A01A",
            "name": "MUNN- OG TANNLYF",
            "level4": {
              "A01AA": {
                "code": "A01AA",
                "name": "Lyf til varnar gegn tannskemmdum",
                "level5": {
                  "A01AA01": {
                    "code": "A01AA01",
                    "name": "Natrii fluoridum",
                    "drugs": {
                      "Duraphat": {
                        "atc_code": "A01AA01",
                        "form_strength": "Tannpasta / 5 mg/g",
                        "documents": [
                          {
                            "type": "Fylgiseðill",
                            "url": "https://...",
                            "date": ""
                          }
                        ]
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
```

#### Drug Mappings Structure
```python
{
  "Duraphat": ["A01AA01"],
  "Corsodyl": ["A01AB03"],
  ...
}
```

## Key Methods Reference

### `scrape_with_selenium() -> Dict[str, Any]`
- Main scraping method using Selenium
- Returns: `{"hierarchy": {...}, "drug_mappings": {...}, "scraped_at": "..."}`

### `_extract_level_recursive(driver, parent_element, parent_code, level=2, max_level=5) -> Dict[str, Any]`
- Recursively extracts hierarchy from level 2 to 5
- Handles stale elements, DOM changes, and AJAX loading
- Returns dictionary mapping ATC codes to their data

### `_extract_drugs_from_level5(driver, row, atc_code) -> Dict[str, Dict[str, Any]]`
- Extracts drug products from a level 5 ATC row
- Returns dictionary mapping drug names to their information

### `_get_level_name(code: str) -> str`
- Maps level 1 codes to human-readable names (English)
- Used for level 1 categories only

### `save(output_path: Optional[Path] = None) -> None`
- Saves scraped data to JSON file
- Default path: `Config.ATC_INDEX_PATH`

## Usage in Codebase

1. **Scraping**: Run `python src/atc_scraper.py --selenium` to generate `data/atc/atc_index.json`

2. **Consumption**: `ATCManager` loads the JSON and provides:
   - `get_atc_codes_for_drug(drug_id)` - Get ATC codes for a drug
   - `get_drugs_by_atc(atc_code)` - Get drugs with a specific ATC code
   - `get_atc_hierarchy_path(atc_code)` - Get full hierarchy path
   - `format_atc_context_for_rag(drug_id)` - Format for RAG prompts

3. **Enrichment**: `enrich_with_atc.py` and `enrich_all_with_atc.py` use `ATCMatcher` to add ATC codes to structured drug data

## Important Notes for Ingredients Scraper

When building the Ingredients scraper (`https://old.serlyfjaskra.is/Ingredients.aspx?d=1&a=0`):

1. **Similar Structure**: The Ingredients page likely uses the same ASP.NET/Telerik RadGrid structure
2. **Alphabet Navigation**: Instead of ATC codes, it uses alphabet letters (A-Z) to navigate
3. **Different Data Model**: 
   - Top level: Alphabet letters (A, B, C, ..., Z)
   - Second level: Active ingredient names (INN names)
   - Third level: Drug products containing that ingredient
4. **Reusable Patterns**:
   - Selenium setup and navigation
   - Stale element handling
   - Recursive extraction logic
   - Detail table finding strategies
   - AJAX wait patterns
5. **Key Differences**:
   - No 5-level hierarchy (likely 2-3 levels: letter → ingredient → drugs)
   - Different column structure (need to inspect)
   - Different data extraction (ingredient names vs ATC codes)

## Configuration

- **Output Path**: `data/atc/atc_index.json` (via `Config.ATC_INDEX_PATH`)
- **Base URL**: `https://old.serlyfjaskra.is`
- **ATC List URL**: `{BASE_URL}/ATCList.aspx?d=1&a=0`
- **Ingredients URL**: `{BASE_URL}/Ingredients.aspx?d=1&a=0` (for new scraper)

