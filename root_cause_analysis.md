# Root Cause Analysis: Repevax_SmPC.pdf and Risperidon_Krka_SmPC.pdf

## Problem Summary

Both PDFs fail validation because sections are detected but marked as empty. The root cause is **false positive heading detection** - the parser incorrectly identifies numbers followed by units (like "1 mg" or "1 ár") as section headings.

## Detailed Analysis

### Repevax_SmPC.pdf

**What's happening:**
1. The parser correctly detects real section headings like:
   - `3. skammtinum af barnaveiki- eða dT bóluefni 6 mánuðum...`
   - `6 ára. Hæsta tíðnin úr hvorri rannsókn er sýnd...`
   - `5 til 6 ára.`

2. **But then it also detects false positives:**
   - `1 ár` (line 704, 707, 711) - "1 year" in Icelandic
   - `3 ár` (line 705, 708, 712) - "3 years"
   - `5 ár` (line 706, 709, 713) - "5 years"
   - `10 ár` (line 710, 714) - "10 years"

3. These false headings **overwrite** the real sections because:
   - The parser uses the section number as the key (e.g., "1", "3", "5", "10")
   - When it encounters "1 ár" later in the document, it overwrites section "1" with a new entry
   - Since "1 ár" appears in a table or isolated context, there's no content between it and the next heading
   - Result: Section "1" ends up with heading "1 ár" and empty text

**Evidence from analysis:**
- Section 1: Heading is "1 ár" (should be something like "1. NAME OF THE MEDICINAL PRODUCT")
- Section 3: Heading is "3 ár" (should be "3. PHARMACEUTICAL FORM")
- Section 5: Heading is "5 ár" (should be "5. PHARMACOLOGICAL PROPERTIES")
- Section 10: Heading is "10 ár" but this one has content (12768 chars) - likely because it appears later and captures content from tables

**Why this happens:**
The heading regex pattern `^\s*(\d{1,2}(?:\.\d{1,2})*)\s+(.+)$` matches:
- `1` (number) + `ár` (text) = ✅ Match!
- `3` (number) + `ár` (text) = ✅ Match!
- `5` (number) + `ár` (text) = ✅ Match!

These appear in study duration references within tables (e.g., "3 ára eftirfylgni" = "3 year follow-up").

### Risperidon_Krka_SmPC.pdf

**What's happening:**
1. The parser detects false headings:
   - `1 mg` (line 30) - dosage amount
   - `2 mg` (line 31) - dosage amount
   - `3 mg` (line 32) - dosage amount
   - `4 mg` (line 33) - dosage amount
   - `6 mg` (line 34) - dosage amount

2. These are **dosage specifications**, not section headings:
   - They appear in composition tables listing different tablet strengths
   - Example: "1 mg tafla: Hvít, sporöskjulaga..." = "1 mg tablet: White, oval..."

3. The regex matches:
   - `1` (number) + `mg` (text) = ✅ Match!
   - `2` (number) + `mg` (text) = ✅ Match!

4. These false headings overwrite the real sections 1, 2, 3, 4, 6:
   - Section 1 should be "1. NAME OF THE MEDICINAL PRODUCT"
   - But it's overwritten with "1 mg" (dosage)
   - Since "1 mg" appears in a table, there's no content between it and "2 mg"
   - Result: All sections 1-4 end up empty

**Evidence from analysis:**
- Section 1: Heading is "1 mg" (should be section name)
- Section 2: Heading is "2 mg" (should be section name)
- Section 3: Heading is "3 mg" (should be section name)
- Section 4: Heading is "4 mg" (should be section name)
- Section 6: Heading is "6 mg" but has content (57740 chars) - likely because it appears later and captures table content

## Root Cause

The `HEADING_REGEX` pattern is **too permissive**. It matches:
```regex
^\s*(\d{1,2}(?:\.\d{1,2})*)\s+(.+)$
```

This pattern will match:
- ✅ Valid headings: `1. NAME OF MEDICINAL PRODUCT`
- ✅ Valid headings: `4.1 Ábendingar`
- ❌ **False positives**: `1 mg` (dosage)
- ❌ **False positives**: `1 ár` (duration)
- ❌ **False positives**: `3 til 5 ára` (age range)
- ❌ **False positives**: Any number followed by short text in tables

## Why Sections End Up Empty

1. **Overwriting behavior**: When the parser encounters a heading, it creates/overwrites a section with that number as the key
2. **False headings appear later**: Real sections are created first, then false headings overwrite them
3. **No content between false headings**: Since "1 mg" and "2 mg" appear consecutively in a table, there's no text between them
4. **Result**: Sections have headings but no text content

## Solutions

### Option 1: Improve Heading Regex (Recommended)
Add validation to ensure the text part is substantial and doesn't look like a unit:

```python
HEADING_REGEX = re.compile(
    r'^\s*(\d{1,2}(?:\.\d{1,2})*)\s+(.+)$'
)

# After matching, validate:
if len(title.strip()) < 3:  # Too short, probably a unit
    continue
if title.strip().lower() in ['mg', 'ml', 'ár', 'ára', 'mánuðir', 'mánuði']:
    continue  # Common units, not headings
```

### Option 2: Context-Aware Detection
Check if the "heading" appears in a table or list context:
- If the next line is also a number + unit pattern, it's likely a table
- If the line is very short (< 5 chars after the number), it's likely not a heading

### Option 3: Require Period After Number
Many valid headings have a period: `1. NAME`, `4.1 Ábendingar`
But this might miss some valid formats.

### Option 4: Font-Based Detection Priority
Since font-based detection uses font size and bold, it's less likely to match table entries. However, the current implementation falls back to regex when font detection finds < 5 headings.

## Recommended Fix

Implement **Option 1** with a whitelist/blacklist approach:

1. After regex match, check if the title is a known unit (blacklist)
2. Check if title is too short (< 3 characters)
3. Check if it's followed by another number+unit pattern (table context)
4. Only accept if it passes all checks

This would prevent "1 mg", "1 ár", etc. from being detected as headings while preserving valid section detection.
