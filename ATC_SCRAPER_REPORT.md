# ATC Scraper - Technical Report and Issue Analysis

**Date:** 2025-01-05  
**Status:** Partially Working - Critical Issues with Stale Element Recovery  
**Last Updated:** After implementing expand→process→collapse pattern and parent_code filtering

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture and Design](#architecture-and-design)
3. [How It Works](#how-it-works)
4. [Current Issues](#current-issues)
5. [Attempted Fixes](#attempted-fixes)
6. [Root Cause Analysis](#root-cause-analysis)
7. [Recommendations](#recommendations)

---

## Overview

The ATC (Anatomical Therapeutic Chemical) scraper is designed to extract the complete drug classification hierarchy from the Icelandic Medicines Agency website (`old.serlyfjaskra.is`). The website uses a complex nested table structure with JavaScript-driven expansion/collapse functionality, requiring Selenium WebDriver for proper rendering.

### Key Requirements

- Extract all 14 top-level ATC categories (A, B, C, D, G, H, J, L, M, N, P, R, S, V)
- Navigate through 5 levels of hierarchy (Level 1 → Level 5)
- Extract drug information at Level 5 (drug names, formulations, documents)
- Handle multiple formulations for the same drug name
- Maintain data integrity (correct ATC code assignments)

### Current Status

- ✅ Successfully extracts most of the hierarchy
- ✅ Handles multiple formulations per drug
- ✅ Implements expand→process→collapse pattern
- ❌ **Fails to recover from stale element exceptions at deeper levels**
- ❌ **Skips entire sections when parent detail tables become stale**
- ❌ **Jumps to wrong parent sections when recovery fails**

---

## Architecture and Design

### Class Structure

```python
class ATCScraper:
    - hierarchy: Dict[str, Any]  # Stores the complete ATC hierarchy
    - drug_mappings: Dict[str, List[str]]  # Maps drug names to ATC codes
    
    Methods:
    - scrape_with_selenium()  # Main entry point using Selenium
    - _extract_level_recursive()  # Recursive depth-first traversal
    - _extract_drugs_from_level5()  # Extracts drug data from level 5 rows
    - save()  # Persists data to JSON
```

### Data Structure

The scraper builds a nested dictionary structure:

```json
{
  "hierarchy": {
    "A": {
      "code": "A",
      "name": "...",
      "level2": {
        "A01": {
          "code": "A01",
          "name": "...",
          "level3": { ... },
          "drugs": { ... }  // Only at level 5
        }
      }
    }
  },
  "drug_mappings": {
    "Duraphat": ["A01AA01"],
    "Afipran": ["A03FA01"]
  }
}
```

---

## How It Works

### 1. Initialization

The scraper starts by:
1. Creating a Chrome WebDriver instance
2. Navigating to the ATC list URL
3. Waiting for the main result grid to load
4. Extracting all Level 1 categories (A, B, C, etc.)

### 2. Recursive Depth-First Traversal

The core algorithm uses `_extract_level_recursive()` which:

1. **Finds all rows** at the current level in the parent detail table
2. **For each row:**
   - Extracts ATC code and name
   - Checks if row has children (via `td[3]` HasChild column or expand button)
   - If Level 5: Extracts drugs directly
   - If has children: Expands row → Processes children recursively → Collapses row
3. **Returns** dictionary of processed codes and their data

### 3. Expand → Process → Collapse Pattern

**Intended Behavior:**
```
For each row with children:
  1. Store row_id as entry_point
  2. Expand row (if not already expanded)
  3. Find detail table (following sibling <tr>)
  4. Recursively process all children
  5. Re-find row by entry_point row_id
  6. Collapse row (if we expanded it)
  7. Move to next sibling
```

**Why This Pattern:**
- Keeps DOM simple (only one branch expanded at a time)
- Reduces stale element issues
- Matches the nested table structure

### 4. Drug Extraction (Level 5)

When processing Level 5 rows:
1. Expands the row to reveal drug detail table
2. Finds drug detail table using multiple methods:
   - Method 1: Following sibling using row ID
   - Method 2: From parent using preceding-sibling XPath
   - Method 3: Direct following sibling from row element
3. Extracts drug information:
   - Drug name (column 3)
   - Form/strength (column 4)
   - Documents (column 5)
4. Stores formulations as array to handle multiple formulations per drug name

### 5. Stale Element Handling

The scraper attempts to handle stale elements through:

1. **Proactive Re-finding:** Before using `parent_element`, checks if it's stale
2. **Multiple Recovery Strategies:**
   - Strategy 1: Find detail table containing `parent_code`
   - Strategy 2: Navigate via parent's parent
   - Strategy 3: Fallback search through all detail tables
3. **Row Filtering:** When using fallback methods, filters rows by `parent_code` prefix

---

## Current Issues

### Issue 1: Stale Element Recovery Fails at Deeper Levels

**Symptom:**
```
WARNING: Could not re-find parent detail table at level 3 for A02
WARNING: Could not recover parent element at level 3 for A02, will try to continue
INFO: Processing A02B at level 3 (depth-first)  # WRONG! Should continue with A02A children
```

**What Happens:**
- After processing A02AF02 (Level 5, no children), the parent detail table becomes stale
- Recovery strategies fail to re-find the correct detail table
- The scraper jumps to A02B (Level 3) instead of continuing with A02AG, A02AH, A02AX (Level 4)

**Affected Sections:**
- A02A → Missing: A02AG, A02AH, A02AX
- A03F → Missing: A03FA01 (Metoclopramidum with Afipran)
- B06A → Missing: B06AC01 (C1-hemill with Berinert and Cinryze)

**Impact:** Entire branches of the hierarchy are skipped, leading to incomplete data.

### Issue 2: Parent Code Filtering Not Working Correctly

**Problem:**
When recovery strategies fail, the fallback row finding uses pattern matching:
```python
rows = driver.find_elements(
    By.XPATH,
    f"//table[contains(@class, 'rgDetailTable')]//tbody/tr[td[4] and string-length(translate(td[4], ' ', ''))={expected_length}]"
)
```

Even with `parent_code` prefix filtering, the scraper still finds rows from wrong sections because:
- Multiple detail tables exist in the DOM simultaneously
- The filter checks `code_clean.startswith(parent_code)`, but A02AG, A02AH start with "A02A", and A02B also starts with "A02" - causing confusion
- The XPath finds rows from ALL detail tables, not just the one we need

### Issue 3: Row Re-finding Logic Breaks

**Problem:**
When `parent_element` becomes stale and we try to re-find a row:
```python
row = parent_element.find_element(By.XPATH, f".//tbody/tr[td[4]='{code_clean}']")
```

This fails because `parent_element` is stale. The fallback tries:
```python
row = driver.find_element(By.XPATH, f"//tr[td[4]='{code_clean}']")
```

But this finds the FIRST row with that code in the entire document, which might be from a different section or already processed.

### Issue 4: No Tracking of Processing State

**Problem:**
The scraper uses `processed_codes` set to track what's been processed, but:
- When recovery happens, it doesn't know which rows have already been processed
- It might re-process rows or skip unprocessed ones
- No way to resume from a known good state

---

## Attempted Fixes

### Fix 1: Multiple Recovery Strategies (Lines 483-681)

**What Was Tried:**
- Strategy 1: Search all detail tables for one containing `parent_code`
- Strategy 2: Navigate via parent's parent (e.g., A02A → A02 → find A02A's detail table)
- Strategy 3: Fallback search with ancestor verification

**Result:** ❌ Still fails. The strategies can't reliably find the correct detail table because:
- Multiple detail tables exist in DOM
- Ancestor checks fail when tables are nested
- Parent's parent navigation breaks when intermediate levels are collapsed

### Fix 2: Parent Code Prefix Filtering (Lines 698-745)

**What Was Tried:**
- Filter rows by checking `code_clean.startswith(parent_code)`
- Applied to both level 2 and deeper level recovery

**Result:** ❌ Partial success. Still jumps to wrong sections because:
- A02AG, A02AH start with "A02A" ✓
- But A02B also starts with "A02" - the filter is too broad
- When parent_code is "A02", both A02A children and A02B children match

### Fix 3: Expand → Process → Collapse Pattern (Lines 884-980)

**What Was Tried:**
- Store `entry_point_row_id` before expanding
- After processing children, re-find row by ID and collapse it
- Keep DOM simple with only one branch expanded

**Result:** ✅ Helps reduce stale elements, but ❌ doesn't solve the recovery problem. When parent becomes stale, we can't find the entry point row either.

### Fix 4: Direct Row Finding with Parent Code Validation (Lines 789-820)

**What Was Tried:**
- When `parent_element` is stale, find row directly by code
- Validate that row's code starts with `parent_code`
- Try to update `parent_element` from the found row's ancestor table

**Result:** ❌ Fails because:
- Finding row by code alone finds the first match in entire document
- Might find a row from a different section or already processed
- Ancestor table finding fails when structure is complex

### Fix 5: Enhanced Stale Element Detection (Lines 457-681)

**What Was Tried:**
- Check if `parent_element` is stale before using it
- Multiple recovery attempts before giving up
- Continue processing even if recovery fails

**Result:** ❌ Continues processing but with wrong context. Jumps to wrong sections instead of breaking.

---

## Root Cause Analysis

### The Fundamental Problem

**The website's DOM structure is highly dynamic:**
1. When a row is expanded, a new `<tr>` is inserted with a nested `<table class="rgDetailTable">`
2. When a row is collapsed, the detail table `<tr>` is removed
3. Multiple detail tables can exist simultaneously (if multiple branches are expanded)
4. Selenium WebDriver references become stale when DOM changes

**The scraper's approach has a fundamental flaw:**
- It tries to maintain references to `parent_element` (a detail table)
- When that table's parent row is collapsed (by our collapse pattern or by other operations), the reference becomes stale
- Recovery strategies try to re-find the table, but can't reliably identify which table is the correct one

### Why Recovery Strategies Fail

1. **Strategy 1 (Search all detail tables):** Finds multiple tables, can't determine which is correct
2. **Strategy 2 (Navigate via parent's parent):** Breaks when intermediate levels are collapsed or stale
3. **Strategy 3 (Fallback with ancestor check):** Ancestor checks fail with complex nesting
4. **Parent code filtering:** Too broad (A02 matches both A02A and A02B children)

### The Core Issue

**We're trying to maintain state (which detail table we're processing) in a dynamic DOM where that state can disappear at any time.**

When we collapse a row after processing its children, we're removing the detail table from the DOM. If we then need to continue processing siblings, we need to re-find the parent detail table - but it might have been removed or become stale.

---

## Recommendations

### Option 1: Complete Rewrite with State Machine Approach

**Concept:** Don't maintain references to DOM elements. Instead:

1. **Build a complete map first:** Before processing, expand everything and build a complete map of row IDs and their relationships
2. **Process using row IDs only:** Use row IDs to navigate, always re-finding elements by ID
3. **Track processing state separately:** Maintain a separate data structure tracking what's been processed

**Pros:**
- No stale element issues (always re-find by ID)
- Can resume from any point
- Clear separation of concerns

**Cons:**
- Requires significant rewrite
- Might be slower (expanding everything first)

### Option 2: Row ID-Based Navigation

**Concept:** Never maintain `parent_element` references. Always navigate by row ID:

1. Store row IDs in a queue/stack
2. For each row ID, re-find the row, find its detail table, process children
3. Get child row IDs and add to queue
4. Never maintain references across operations

**Pros:**
- Eliminates stale element issues
- Simpler logic
- More robust

**Cons:**
- More DOM queries (but more reliable)
- Need to restructure the recursive approach

### Option 3: Process All Rows at Level Before Moving Down

**Concept:** Change from depth-first to level-by-level:

1. Process all Level 2 rows, store their row IDs
2. For each Level 2 row ID, expand and process all Level 3 rows
3. Continue level by level

**Pros:**
- Simpler state management
- Easier to track progress
- Can resume from any level

**Cons:**
- Different traversal order
- Might need to handle expanded state differently

### Option 4: Use Explicit Waits and Retry Logic

**Concept:** Instead of trying to re-find elements, use explicit waits:

1. When parent becomes stale, wait for it to become available again
2. Use WebDriverWait with custom conditions
3. Retry operations with exponential backoff

**Pros:**
- Minimal code changes
- Handles timing issues

**Cons:**
- Doesn't solve the fundamental problem
- If element is truly gone (collapsed), wait will timeout

### Recommended Approach: Option 2 (Row ID-Based Navigation)

**Why:**
- Solves the stale element problem at its root
- Maintains the depth-first approach (which is efficient)
- Can be implemented incrementally
- Most robust solution

**Implementation Sketch:**
```python
def _extract_level_by_row_ids(self, driver, parent_row_id, level, max_level):
    # Re-find parent row by ID
    parent_row = driver.find_element(By.XPATH, f"//tr[@id='{parent_row_id}']")
    
    # Expand if needed
    # Find detail table
    # Get all child row IDs
    child_row_ids = [row.get_attribute("id") for row in child_rows]
    
    # Process each child
    for child_row_id in child_row_ids:
        # Re-find row by ID
        child_row = driver.find_element(By.XPATH, f"//tr[@id='{child_row_id}']")
        # Extract data
        # If has children, recurse with child_row_id
        # Collapse if we expanded
```

---

## Conclusion

The current scraper works for most cases but fails when DOM elements become stale at deeper levels. The recovery strategies are insufficient because they can't reliably identify the correct detail table in a dynamic DOM with multiple nested tables.

**The fundamental issue is maintaining references to DOM elements that can disappear.** The solution is to stop maintaining references and instead always navigate by row IDs, re-finding elements as needed.

**Next Steps:**
1. Implement Option 2 (Row ID-Based Navigation)
2. Test thoroughly with problematic sections (A02A, A03F, B06A)
3. Add comprehensive logging to track which sections are processed
4. Consider adding a resume capability for partial scrapes

---

## Appendix: Known Missing Sections

Based on testing, the following sections are known to be missing or incomplete:

- **A02A:** Missing A02AG, A02AH, A02AX
- **A03F:** Missing A03FA01 (Metoclopramidum with Afipran formulations)
- **B06A:** Missing B06AC01 (C1-hemill with Berinert and Cinryze)

These sections fail at the same point: after processing a level 5 row with no children, the parent detail table becomes stale and recovery fails, causing the scraper to jump to the next level 3 section instead of continuing with remaining level 4 rows.

