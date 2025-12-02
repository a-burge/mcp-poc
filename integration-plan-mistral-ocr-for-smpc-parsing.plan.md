<!-- 27891605-9b09-4850-98d6-119051188a44 3681da45-e476-45da-9763-016f684eef7b -->
# Integration Plan: Mistral OCR for SmPC Parsing

## Current Project Structure

```
ingest_all_smpcs.py
  └─> scan_pdf_files()           # Scans RAW_SOURCE_DOCS_DIR for PDFs
  └─> process_smpc_pdf()        # Main processing function
       └─> build_smpc_json()    # From smpc_parser.py
            └─> extract_pdf_text()           # PyMuPDF text extraction
            └─> parse_sections()             # Regex/font-based heading detection
       └─> is_valid_smpc()       # Validation
       └─> chunk_smpc_json()     # Chunking
       └─> vector_store.add_chunks()  # Indexing
```

## Integration Points

### **Two Options to Consider:**

#### **Option A: OCR + ML Solution (Original Proposal)**

**Advantages:**

- **Solves your core problem**: ML classifier will dramatically reduce false positives (like "1 mg", "1 ár")
- **Works on any PDF**: OCR handles scanned PDFs and image-based documents
- **Layout-aware**: OCR preserves spatial relationships (blocks, lines)
- **Trainable accuracy**: Once trained, should achieve 90%+ heading detection accuracy
- **No API costs**: Runs entirely locally
- **Full control**: You own the model and can fine-tune it

**Disadvantages:**

- **Complex setup**: Requires Poppler, Tesseract, training data collection, model training
- **Performance**: OCR is 10-50x slower than PyMuPDF (seconds vs milliseconds per page)
- **Training overhead**: Need to manually label 200-500 blocks
- **Maintenance**: You maintain the model and training pipeline
- **OCR quality**: Tesseract may introduce errors with Icelandic text

#### **Option B: Mistral OCR API (Simpler Alternative)** ⭐

Based on [Mistral OCR](https://mistral.ai/news/mistral-ocr), this is a commercial API that:

**Advantages:**

- **Much simpler**: No system dependencies, no training data, no model training
- **High accuracy**: 94.89% overall, 99.02% fuzzy match (beats Google, Azure, Gemini)
- **Native Icelandic support**: Multilingual with excellent results (99.20% for similar languages)
- **Structured output**: Can extract to JSON format directly
- **Fast**: 2000 pages/minute (faster than Tesseract)
- **No maintenance**: Mistral maintains the model
- **Document understanding**: Handles tables, math, images, complex layouts
- **Doc-as-prompt**: Can extract specific structured information

**Disadvantages:**

- **API costs**: $1 per 1000 pages (or ~$0.50 per 1000 with batch)
- **API dependency**: Requires internet connection, subject to rate limits
- **Data privacy**: Documents sent to external API (though self-hosting available for enterprise)
- **Less control**: Can't fine-tune the model yourself

**Cost estimate for your use case:**

- If you have ~50 PDFs averaging 20 pages each = 1000 pages
- Cost: ~$1-2 total (or $0.50-1 with batch)
- Ongoing: Depends on how often you re-process

### **Recommendation: Start with Mistral OCR**

**Why:**

1. **Simplicity**: No training, no system dependencies, just API calls
2. **Speed to value**: Can test and integrate in hours, not days
3. **Better accuracy**: 99%+ vs ~90% for trained classifier
4. **Icelandic support**: Proven multilingual capabilities
5. **Structured extraction**: Can request JSON output matching your format
6. **Low cost**: For your scale, API costs are negligible

**Fallback strategy:**

- If API costs become prohibitive at scale → switch to OCR+ML
- If data privacy becomes critical → Mistral offers self-hosting (enterprise)
- If you need offline processing → implement OCR+ML as backup

---

## 2. What Is Needed to Implement This Solution?

### 1. File Filtering (Filename Check)

**Location:** `ingest_all_smpcs.py` → `scan_pdf_files()`

**Change:**

```python
def scan_pdf_files(source_dir: Path) -> List[Path]:
    pdf_files = []
    for file_path in source_dir.iterdir():
        if (file_path.is_file() and 
            file_path.suffix.lower() == '.pdf' and
            'smpc' in file_path.name.lower()):  # NEW: Filter by filename
            pdf_files.append(file_path)
    return pdf_files
```

**Result:** Only processes files with "SMPC" (case-insensitive) in filename

### 2. New Module: Mistral OCR Extractor

**New file:** `src/smpc_extractor_mistral.py`

**Functions:**

- `extract_with_mistral_ocr(pdf_path: str) -> Dict[str, Any]`
  - Calls Mistral OCR API
  - Uses doc-as-prompt to request structured JSON
  - Returns sections in your existing format

**Key design:**

- Use Mistral's structured output feature
- Prompt: "Extract all numbered sections (1, 2, 3, 4.1, 4.2, etc.) with headings and content. Output as JSON with this structure: {...}"
- Parse response and convert to match `build_smpc_json()` output format

### 3. Integration into Parser

**Modify:** `src/smpc_parser.py` → `build_smpc_json()`

**Add parameter:**

```python
def build_smpc_json(
    pdf_path: str,
    drug_id: Optional[str] = None,
    use_font_detection: bool = True,
    font_size_threshold: float = 1.2,
    use_mistral_ocr: bool = False  # NEW
) -> Dict[str, Any]:
```

**Logic:**

```python
if use_mistral_ocr:
    from src.smpc_extractor_mistral import extract_with_mistral_ocr
    return extract_with_mistral_ocr(pdf_path, drug_id)
else:
    # Existing PyMuPDF path
    raw_text = extract_pdf_text(str(pdf_path_obj))
    sections, validation_report = parse_sections(...)
    # ... rest of existing code
```

### 4. Auto-Detection in Ingestion

**Modify:** `ingest_all_smpcs.py` → `process_smpc_pdf()`

**Change:**

```python
# Step 1: Parse PDF to structured JSON
# Auto-use Mistral OCR for files with "SMPC" in name
use_mistral = 'smpc' in pdf_path.name.lower()
smpc_data = build_smpc_json(
    str(pdf_path),
    use_mistral_ocr=use_mistral
)
```

**Result:** Files with "SMPC" in name automatically use Mistral OCR

### 5. Configuration

**Modify:** `config.py`

**Add:**

```python
MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY", "")
```

**Modify:** `requirements.txt`

**Add:**

```
mistralai>=1.0.0
```

### 6. Format Conversion

**Challenge:** Mistral OCR returns different structure than your current format

**Solution:** Create adapter function in `smpc_extractor_mistral.py`:

```python
def convert_mistral_to_smpc_format(
    mistral_response: Dict,
    pdf_path: str,
    drug_id: str
) -> Dict[str, Any]:
    """
    Convert Mistral OCR output to match build_smpc_json() format.
    
    Your format:
    {
      "drug_id": str,
      "source_pdf": str,
      "version_hash": str,
      "extracted_at": str,
      "sections": {
        "4.3": {
          "number": "4.3",
          "parent": "4",
          "heading": "4.3 Frábendingar",
          "title": "Frábendingar",
          "canonical_key": "contraindications",
          "text": "...",
          "children": []
        }
      },
      "validation_report": {...}
    }
    """
    # Parse Mistral response
    # Map sections to your structure
    # Preserve canonical_key mapping
    # Build parent/children relationships
    # Return in exact same format
```

**Key:** Output format must be identical to current `build_smpc_json()` output so downstream code (chunking, validation) works unchanged.

### E. Integration Points

**Option 1: Replace current parser entirely**

- Modify `build_smpc_json()` to use OCR extractor
- Pros: Single code path
- Cons: All PDFs become slower

**Option 2: Hybrid routing (recommended)**

- Add `use_ocr_extractor: bool = False` parameter to `build_smpc_json()`
- Route based on:
  - User preference
  - Previous failure (retry with OCR)
  - Document type detection
- Pros: Best of both worlds
- Cons: More complex logic

**Option 3: Parallel processing**

- Try PyMuPDF first, validate results
- If validation fails or detects issues, retry with OCR+ML
- Pros: Fast for good PDFs, accurate for problematic ones
- Cons: Some PDFs processed twice

### F. Output Format Compatibility

**Mistral OCR advantages:**

- Can request structured JSON output matching your exact schema
- Handles section detection automatically (no regex needed)
- Returns hierarchical structure (sections, subsections)
- Can extract canonical keys if you provide examples in prompt

**Integration approach:**

1. Call Mistral OCR with PDF
2. Request JSON output with your schema
3. Parse response into existing format
4. Minimal transformation needed

**Current format** (from `build_smpc_json()`):

```python
{
  "drug_id": str,
  "source_pdf": str,
  "version_hash": str,
  "extracted_at": str,
  "sections": {
    "4.3": {
      "number": "4.3",
      "parent": "4",
      "heading": "4.3 Frábendingar",
      "title": "Frábendingar",
      "canonical_key": "contraindications",
      "text": "...",
      "children": []
    }
  }
}
```

**Proposed format** (from `process_pdf_smpc()`):

```python
{
  "pdf_path": str,
  "document_type": "smpc" | "other",
  "num_blocks": int,
  "sections": [
    {
      "level": int,
      "heading": str,
      "page": int,
      "blocks": [{"page": int, "text": str, "x": int, "y": int, "w": int, "h": int}]
    }
  ]
}
```

**Required adaptation:**

- Convert OCR output format to match your existing structure
- Preserve `canonical_key` mapping
- Maintain parent/children relationships
- Keep `validation_report` structure

### G. Testing & Validation

**Test plan:**

1. Test Mistral OCR on 3-5 problematic PDFs (those with false positives)
2. Compare results with current PyMuPDF parser
3. Verify section detection accuracy
4. Check output format compatibility
5. Measure API response time and costs

**Success criteria:**

- No false positives (no "1 mg" detected as heading)
- All real sections detected correctly
- Output format matches current structure
- API response time acceptable (< 10s per PDF)
- Cost acceptable for your volume

**Quick validation:**

- Test on `Repevax_SmPC.pdf` and `Risperidon_Krka_SmPC.pdf` (known problematic files)
- Should correctly identify sections without false positives

---

## Implementation Phases

### Phase 1: Setup & Testing (1-2 hours)

- ✅ API key already in `.env` (you mentioned)
- Add `mistralai` to `requirements.txt` and install
- Create `src/smpc_extractor_mistral.py` skeleton
- Test Mistral OCR API call on one sample PDF
- Verify response format and Icelandic text quality

### Phase 2: Core Implementation (2-3 hours)

- Implement `extract_with_mistral_ocr()` function
- Design prompt for structured JSON extraction
- Create `convert_mistral_to_smpc_format()` adapter
- Test format conversion on sample PDF
- Verify output matches `build_smpc_json()` structure exactly

### Phase 3: Integration (1-2 hours)

- Add `use_mistral_ocr` parameter to `build_smpc_json()` in `smpc_parser.py`
- Add filename filter to `scan_pdf_files()` in `ingest_all_smpcs.py`
- Update `process_smpc_pdf()` to auto-detect SMPC files
- Add `MISTRAL_API_KEY` to `config.py`
- Test end-to-end: scan → filter → Mistral OCR → JSON → validation

### Phase 4: Validation (1-2 hours)

- Test on problematic PDFs (Repevax_SmPC.pdf, Risperidon_Krka_SmPC.pdf)
- Compare Mistral results vs current PyMuPDF parser
- Verify no false positives ("1 mg", "1 ár" not detected as headings)
- Check all sections detected correctly
- Verify downstream compatibility (chunking, indexing work unchanged)

### Phase 5: Error Handling & Polish (1 hour)

- Add retry logic for API calls
- Add error handling (API failures, rate limits)
- Add logging for Mistral OCR usage
- Update documentation
- Test on full corpus of SMPC files

**Total time estimate: 6-10 hours**

---

## How It Fits: Visual Flow

### Before (Current):

```
PDF files in RAW_SOURCE_DOCS_DIR
  ↓
scan_pdf_files() → All PDFs
  ↓
process_smpc_pdf()
  ↓
build_smpc_json() → PyMuPDF extraction
  ↓
parse_sections() → Regex/font detection
  ↓
is_valid_smpc() → Validation
  ↓
chunk_smpc_json() → Chunking
  ↓
vector_store → Indexing
```

### After (With Mistral OCR):

```
PDF files in RAW_SOURCE_DOCS_DIR
  ↓
scan_pdf_files() → Only PDFs with "SMPC" in filename
  ↓
process_smpc_pdf()
  ↓
build_smpc_json(use_mistral_ocr=True) → Auto-detected for SMPC files
  ↓
extract_with_mistral_ocr() → Mistral API call
  ↓
convert_mistral_to_smpc_format() → Format adapter
  ↓
[Same format as before]
  ↓
is_valid_smpc() → Validation (unchanged)
  ↓
chunk_smpc_json() → Chunking (unchanged)
  ↓
vector_store → Indexing (unchanged)
```

**Key Point:** Everything after `build_smpc_json()` remains unchanged because the output format is identical.

---

## Risks & Mitigations

**Risk 1: Poor OCR quality on Icelandic text**

- Mitigation: Test on sample PDFs first, consider alternative OCR engines if needed

**Risk 2: Training data insufficient**

- Mitigation: Start with 200 blocks, expand if accuracy < 90%

**Risk 3: Performance too slow for batch processing**

- Mitigation: Implement parallel processing, cache OCR results, use hybrid approach

**Risk 4: Output format incompatibility**

- Mitigation: Create adapter layer early, test format conversion thoroughly

### To-dos

- [ ] Add `mistralai>=1.0.0` to `requirements.txt` and install dependencies
- [ ] Add `MISTRAL_API_KEY` configuration to `config.py` (read from environment)
- [ ] Create `src/smpc_extractor_mistral.py` module with Mistral OCR integration
- [ ] Implement `extract_with_mistral_ocr()` function to call Mistral OCR API
- [ ] Design and implement prompt for structured JSON extraction (section detection, canonical keys)
- [ ] Create `convert_mistral_to_smpc_format()` adapter function to match existing `build_smpc_json()` output format
- [ ] Add `use_mistral_ocr: bool = False` parameter to `build_smpc_json()` in `smpc_parser.py`
- [ ] Implement routing logic in `build_smpc_json()` to use Mistral OCR when flag is True
- [ ] Add filename filter to `scan_pdf_files()` in `ingest_all_smpcs.py` to only process files with "SMPC" in name
- [ ] Update `process_smpc_pdf()` in `ingest_all_smpcs.py` to auto-detect SMPC files and set `use_mistral_ocr=True`
- [ ] Test Mistral OCR API call on one sample PDF to verify response format and Icelandic text quality
- [ ] Test format conversion on sample PDF to verify output matches `build_smpc_json()` structure exactly
- [ ] Test end-to-end: scan → filter → Mistral OCR → JSON → validation → chunking → indexing
- [ ] Test on problematic PDFs (Repevax_SmPC.pdf, Risperidon_Krka_SmPC.pdf) and compare with current PyMuPDF parser
- [ ] Verify no false positives ("1 mg", "1 ár" not detected as headings)
- [ ] Verify all real sections detected correctly
- [ ] Measure API response time and costs for your volume
- [ ] Add retry logic and error handling for API calls (rate limits, failures)
- [ ] Add logging for Mistral OCR usage and API costs
- [ ] Test on full corpus of SMPC files
- [ ] Update documentation with Mistral OCR integration details
