# Deep Analysis: ATC Index vs Structured Data

## Executive Summary

**The 1,366 vs 447 difference (919 more drugs in structured data) is explained by:**

1. **198 drugs match** between ATC and structured data (44.1% of ATC drugs)
2. **61 veterinary drugs** in structured data (not in ATC - expected, as ATC is for human drugs)
3. **1,107 human drugs** in structured data that are **not** in the ATC index
4. **250 drugs** in ATC index that are **not** in structured data

**Net difference: 1,366 - 447 = 919 more in structured data**

## Detailed Breakdown

### Matching Results (with improved normalization)

| Metric | Count | Percentage |
|--------|-------|------------|
| **ATC Index Total** | 447 | 100% |
| **Structured Data Total** | 1,366 | 100% |
| **Matched Drugs** | 198 | 44.1% of ATC, 14.5% of Structured |
| **Only in ATC** | 250 | 55.9% of ATC |
| **Only in Structured** | 1,168 | 85.5% of Structured |

### Why Drugs Are Only in Structured Data

#### 1. Veterinary Drugs (61 drugs)
These are veterinary medications, which are **not** included in the ATC classification system (ATC is for human drugs only).

**Examples:**
- `AQUI-S_vet_SmPC`
- `Addimag_vet_SmPC`
- `Baytril_vet_SmPC`
- `Cefabactin-vet_SmPC`

**Impact:** Expected - veterinary drugs shouldn't be in ATC index.

#### 2. Human Drugs Not in ATC (1,107 drugs)
These are human medications that exist in structured data but are **not** found in the ATC index.

**Possible reasons:**
- **Newer drugs** not yet added to ATC classification
- **Discontinued drugs** removed from ATC but still in structured data
- **Different naming conventions** that couldn't be matched (despite normalization)
- **Specialty drugs** that may not be in the main ATC classification
- **Combination products** with different naming
- **Different formulations** (e.g., `Albuman_200_SmPC` vs `Albuman`)

**Examples:**
- `Abilify_Maintena_Lyfjaver_SmPC` (long-acting formulation)
- `Albuman_200_SmPC`, `Albuman_50_SmPC` (different strengths)
- `Aimovig_Lyfjaver_SmPC` (may be newer drug)
- `Activelle_SmPC` (combination product)

#### 3. Drugs Only in ATC (250 drugs)
These are drugs in the ATC index but **not** in structured data.

**Possible reasons:**
- **Not yet processed** - PDFs not yet ingested into structured data
- **Different source** - ATC index may include drugs from different sources
- **Naming mismatches** - Couldn't be matched despite normalization
- **Discontinued** - May have been removed from structured data

**Examples:**
- `ALBUTEIN`
- `Accofil`
- `Aciclovir Accord` (vs `Aciclovir_Alvogen_SmPC` in structured)
- `Actrapid`
- `Aklief`
- `Amgevita`

## Key Insights

### 1. Naming Conventions

**ATC Index:**
- Human-readable names: `Actilyse`, `Advagraf (Heilsa)`, `Amlodipin Bluefish`
- Uses parentheses for distributors: `Advagraf (Heilsa)`
- Spaces between words

**Structured Data:**
- Filename-based IDs with suffixes: `Actilyse_SmPC`, `Advagraf_Heilsa_SmPC`
- Uses underscores: `Amlodipin_Bluefish_SmPC`
- Always includes `_SmPC` suffix

**Matching:** With improved normalization (stripping `_SmPC` suffix), we can match 198 drugs (44.1% of ATC).

### 2. Coverage Gaps

**ATC Index Coverage:**
- Only 44.1% of ATC drugs have corresponding structured data
- 250 ATC drugs are missing from structured data

**Structured Data Coverage:**
- Only 14.5% of structured drugs are in ATC index
- 1,168 structured drugs are not in ATC (including 61 veterinary)

### 3. Data Completeness

**Structured Data:**
- **0 drugs** have ATC codes stored in their JSON files
- This suggests ATC enrichment hasn't been run, or enrichment failed

**Recommendation:** Run ATC enrichment (`enrich_all_with_atc.py`) to add ATC codes to structured data files.

## Recommendations

### 1. Immediate Actions

1. **Run ATC Enrichment:**
   ```bash
   python enrich_all_with_atc.py
   ```
   This will add ATC codes to structured data files, improving matching.

2. **Investigate Missing ATC Drugs:**
   - Check if the 250 ATC drugs have corresponding PDFs
   - Verify if they need to be ingested into structured data

3. **Investigate Unmatched Structured Drugs:**
   - Review the 1,107 human drugs not in ATC
   - Determine if they should be added to ATC index
   - Check if they're newer/discontinued drugs

### 2. Data Quality Improvements

1. **Normalize Drug Names:**
   - Create a mapping between ATC names and structured data names
   - Use this for better matching and enrichment

2. **Track Veterinary Drugs:**
   - Separate veterinary drugs from human drugs in analysis
   - Consider creating a separate veterinary classification system

3. **Handle Multiple Formulations:**
   - Group drugs by base name (e.g., `Albuman`, `Albuman_200`, `Albuman_50`)
   - Link formulations to base drug in ATC

### 3. Long-term Strategy

1. **Bidirectional Sync:**
   - When new drugs are added to ATC, check if structured data exists
   - When new structured data is created, check if ATC code exists

2. **Automated Matching:**
   - Implement fuzzy matching algorithm
   - Use active ingredient matching as fallback
   - Create confidence scores for matches

3. **Data Validation:**
   - Regular comparison reports (like this one)
   - Automated alerts for large discrepancies
   - Track changes over time

## Conclusion

The difference of **919 more drugs in structured data** is primarily due to:

1. **61 veterinary drugs** (expected - not in ATC)
2. **1,107 human drugs** not in ATC index (newer drugs, different formulations, naming differences)
3. **Minus 250 drugs** in ATC not in structured data (not yet ingested)

**The structured data appears to be more comprehensive** than the ATC index, containing:
- More recent drugs
- More formulations (different strengths)
- Veterinary drugs
- Drugs that may have been removed from ATC

**The ATC index appears to be more curated**, containing:
- Only human drugs
- Standardized naming
- Therapeutic classifications

Both sources are valuable and complementary. The recommendation is to:
1. Enrich structured data with ATC codes where possible
2. Use ATC index for therapeutic classification
3. Use structured data for comprehensive drug information

