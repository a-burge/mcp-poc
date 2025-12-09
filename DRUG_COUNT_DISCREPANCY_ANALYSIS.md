# Drug Count Discrepancy Analysis

## Summary

Comparison of drug counts across three data sources:

| Source | Count | Status |
|--------|-------|--------|
| **ATC Index** (`data/atc/atc_index.json`) | **447** | ✅ Complete |
| **Vector Store** (`data/vector_store/`) | **3** | ⚠️ **CRITICAL: Almost empty** |
| **Structured Data** (`data/structured/`) | **1,366** | ✅ Complete |

## Key Findings

### 1. Vector Store is Almost Empty

**The vector store contains only 3 drugs out of 1,366 available in structured data.**

The 3 indexed drugs are:
- `ALPHA_JECT_micro_SmPC`
- `APROKAM_Smpc`
- `AQUI-S_vet_SmPC`

### 2. Major Discrepancies

#### ATC Index vs Vector Store
- **447 drugs in ATC but not indexed** (100% of ATC drugs are missing)
- **3 drugs indexed but not in ATC** (veterinary drugs or naming differences)

#### Structured Data vs Vector Store
- **1,363 drugs in structured data but not in vector store** (99.8% missing)
- **0 drugs in vector store but not in structured data** (vector store is subset of structured)

#### ATC Index vs Structured Data
- ATC index has 447 drugs
- Structured data has 1,366 drugs
- This is expected: structured data includes more drugs than what's in the ATC classification

## Root Cause Analysis

The vector store is almost completely empty, which suggests:

1. **Ingestion process not run**: The `ingest_all_smpcs.py` script may not have been executed for most files
2. **Ingestion failures**: The ingestion process may have failed silently for most files
3. **Vector store reset**: The vector store may have been cleared after initial ingestion
4. **Processing limitations**: Only a small subset of files may have been processed (e.g., test files)

## Recommendations

### Immediate Actions

1. **Run full ingestion**: Execute `ingest_all_smpcs.py` to index all 1,366 drugs from structured data
   ```bash
   python ingest_all_smpcs.py
   ```

2. **Verify ingestion**: After ingestion, re-run the comparison script to verify all drugs are indexed
   ```bash
   python compare_drug_counts.py
   ```

3. **Check for errors**: Review ingestion logs to identify any files that failed to process

### Long-term Improvements

1. **Monitor ingestion progress**: Add progress tracking and error reporting to ingestion script
2. **Validate completeness**: Add automated checks to ensure all structured data files are indexed
3. **Name normalization**: Consider normalizing drug names between ATC index and structured data for better matching

## Drug Name Format Differences

The comparison revealed naming format differences:

- **ATC Index**: Uses human-readable names (e.g., "Actilyse", "Advagraf (Heilsa)")
- **Structured Data**: Uses filename-based IDs with suffixes (e.g., "Actilyse_SmPC", "Advagraf_Heilsa_SmPC")
- **Vector Store**: Uses drug_id from JSON files (e.g., "Actilyse_SmPC", "ALPHA_JECT_micro_SmPC")

These differences are expected and handled by the normalization functions in `src/drug_utils.py`.

## Next Steps

1. ✅ **Analysis complete** - Discrepancies identified
2. ⏳ **Run ingestion** - Index all structured data files
3. ⏳ **Re-verify** - Confirm all drugs are indexed after ingestion
4. ⏳ **Investigate ATC matching** - Understand why ATC drugs aren't matching structured data names

