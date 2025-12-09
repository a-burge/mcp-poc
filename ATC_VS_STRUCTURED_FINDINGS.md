# ATC Index vs Structured Data: Key Findings

## The 1,366 vs 447 Difference Explained

### Summary Statistics

| Source | Count | Description |
|--------|-------|-------------|
| **ATC Index** | 447 | Human drugs with therapeutic classification |
| **Structured Data** | 1,366 | All processed SmPC documents (human + veterinary) |
| **Difference** | +919 | More drugs in structured data |

### Matching Results

With improved normalization (stripping `_SmPC` suffixes), we found:

- **198 drugs match** between ATC and structured data
  - 44.1% of ATC drugs have matches
  - 14.5% of structured drugs have matches

- **250 drugs only in ATC** (not in structured data)
  - 55.9% of ATC drugs are missing from structured data

- **1,168 drugs only in structured data** (not in ATC)
  - 85.5% of structured drugs are not in ATC index

## Why the Difference?

### 1. Veterinary Drugs (61 drugs) ✅ Expected

**ATC Index:** Only human drugs (ATC classification is for human medicines)

**Structured Data:** Includes veterinary drugs (e.g., `AQUI-S_vet_SmPC`, `Baytril_vet_SmPC`)

**Impact:** This is expected - veterinary drugs shouldn't be in ATC.

### 2. Human Drugs Not in ATC (1,107 drugs) ⚠️ Main Discrepancy

These are human medications in structured data but not in ATC index.

**Patterns observed:**

1. **Different Formulations/Strengths:**
   - `Albuman` (ATC) vs `Albuman_200_SmPC`, `Albuman_50_SmPC` (structured)
   - ATC has base name, structured has specific strengths

2. **Distributor-Specific Names:**
   - `Advagraf (Heilsa)` (ATC) vs `Advagraf_Heilsa_SmPC` (structured) ✅ Matched
   - But many distributor-specific drugs in structured data aren't in ATC
   - Common distributors: Alvogen, Lyfjaver, Abacus, Mylan, STADA, Krka, Teva

3. **Newer/Long-Acting Formulations:**
   - `Abilify_Maintena_Lyfjaver_SmPC` (long-acting injection)
   - `Alvofen_Express_SmPC` (fast-acting formulation)
   - These may be newer products not yet in ATC

4. **Naming Conventions:**
   - ATC: Simple names with spaces (`Actilyse`, `Amlodipin Bluefish`)
   - Structured: Filename-based with underscores (`Actilyse_SmPC`, `Amlodipin_Bluefish_SmPC`)

**Possible reasons:**
- Newer drugs not yet added to ATC classification
- Different formulations not separately classified in ATC
- Discontinued drugs removed from ATC but still in structured data
- Specialty/rare drugs not in main ATC classification
- Naming differences that couldn't be matched

### 3. Drugs Only in ATC (250 drugs) ⚠️ Missing from Structured

These are drugs in ATC index but not in structured data.

**Patterns observed:**

1. **Simple Names:**
   - `ALBUTEIN`, `Accofil`, `Actrapid`, `Aklief`
   - No underscores, no `_SmPC` suffix
   - Base drug names

2. **Distributor in Parentheses:**
   - `Eliquis (Abacus Medicine)`, `Eliquis (Lyfjaver)`
   - `Fucidin (Heilsa)`, `Fungoral (Heilsa)`

3. **Spaces in Names:**
   - `Adalat Oros`, `AmBisome liposomal`
   - `Centyl með kaliumklorid`

**Possible reasons:**
- PDFs not yet ingested into structured data
- Different source (ATC may include drugs from different registry)
- Naming mismatches preventing matching
- Discontinued and removed from structured data

## Key Insights

### 1. Data Completeness

**Structured Data:**
- **0 drugs** have ATC codes in their JSON files
- ATC enrichment hasn't been run or failed
- **Recommendation:** Run `enrich_all_with_atc.py`

**ATC Index:**
- Only 44.1% have corresponding structured data
- Many ATC drugs are missing from structured data

### 2. Naming Differences

**ATC Index:**
- Human-readable: `Actilyse`, `Advagraf (Heilsa)`, `Amlodipin Bluefish`
- Uses spaces and parentheses

**Structured Data:**
- Filename-based: `Actilyse_SmPC`, `Advagraf_Heilsa_SmPC`, `Amlodipin_Bluefish_SmPC`
- Uses underscores and `_SmPC` suffix

**Matching:** With suffix stripping, we can match 198 drugs (44.1% of ATC).

### 3. Coverage Gaps

**ATC → Structured:**
- 250 ATC drugs missing from structured data (55.9%)
- Need to ingest these PDFs

**Structured → ATC:**
- 1,107 human drugs not in ATC (81.0% of unmatched structured drugs)
- May be newer drugs, different formulations, or naming differences

## Recommendations

### Immediate Actions

1. **Run ATC Enrichment:**
   ```bash
   python enrich_all_with_atc.py
   ```
   This will add ATC codes to structured data files.

2. **Ingest Missing ATC Drugs:**
   - Check if PDFs exist for the 250 ATC-only drugs
   - Ingest them into structured data

3. **Investigate Unmatched Structured Drugs:**
   - Review the 1,107 human drugs not in ATC
   - Determine if they should be added to ATC index
   - Check if they're newer/discontinued drugs

### Data Quality Improvements

1. **Create Drug Name Mapping:**
   - Map ATC names to structured data names
   - Handle base names vs formulations (e.g., `Albuman` → `Albuman_200_SmPC`)

2. **Separate Veterinary Drugs:**
   - Tag veterinary drugs in structured data
   - Exclude from ATC matching

3. **Handle Multiple Formulations:**
   - Group by base name
   - Link formulations to base drug in ATC

### Long-term Strategy

1. **Bidirectional Sync:**
   - When new drugs added to ATC → check structured data
   - When new structured data created → check ATC code

2. **Automated Matching:**
   - Fuzzy matching algorithm
   - Active ingredient matching as fallback
   - Confidence scores for matches

3. **Regular Validation:**
   - Automated comparison reports
   - Alerts for large discrepancies
   - Track changes over time

## Conclusion

The **919 more drugs in structured data** is primarily due to:

1. ✅ **61 veterinary drugs** (expected - not in ATC)
2. ⚠️ **1,107 human drugs** not in ATC (newer drugs, formulations, naming differences)
3. ⚠️ **Minus 250 drugs** in ATC not in structured (not yet ingested)

**Structured data is more comprehensive:**
- Includes veterinary drugs
- Has more formulations (different strengths)
- Contains newer/recent drugs
- More complete coverage

**ATC index is more curated:**
- Only human drugs
- Standardized naming
- Therapeutic classifications
- May be more up-to-date for some drugs

**Both sources are valuable and complementary.** The recommendation is to:
1. Enrich structured data with ATC codes
2. Use ATC for therapeutic classification
3. Use structured data for comprehensive drug information
4. Create mappings between the two sources

