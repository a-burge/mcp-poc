# How Íbúfen Records Would Appear in ATC Index

## Current Status

**Íbúfen is NOT in the ATC index** because the **M section (Musculoskeletal system) is missing** from the scraped data.

The ATC scraper attempted to scrape sections: A, B, C, D, G, H, J, L, M, N, P, R, S, V
But only these sections were successfully scraped: A, B, C, D, G, H, J, L

**M section (Musculoskeletal system) is missing**, which is where Ibuprofen (M01AE01) would be found.

## How the Three Records Would Appear

Based on the structure of other drugs in the ATC index, here's how the three Íbúfen records would appear:

### In the Hierarchy Structure

```json
{
  "hierarchy": {
    "M": {
      "code": "M",
      "name": "Musculo-skeletal system",
      "level2": {
        "M01": {
          "code": "M01",
          "name": "ANTIINFLAMMATORY AND ANTIRHEUMATIC PRODUCTS",
          "level3": {
            "M01A": {
              "code": "M01A",
              "name": "Antiinflammatory and antirheumatic products, non-steroids",
              "level4": {
                "M01AE": {
                  "code": "M01AE",
                  "name": "Propionic acid derivatives",
                  "level5": {
                    "M01AE01": {
                      "code": "M01AE01",
                      "name": "Ibuprofenum",
                      "drugs": {
                        "Íbúfen": {
                          "atc_code": "M01AE01",
                          "form_strength": "Filmuhúðuð tafla / 200 mg",
                          "documents": [
                            {
                              "type": "Fylgiseðill",
                              "url": "https://old.serlyfjaskra.is/FileRepos/...",
                              "date": "25.3.2025"
                            },
                            {
                              "type": "SmPC",
                              "url": "https://old.serlyfjaskra.is/FileRepos/...",
                              "date": "25.3.2025"
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
}
```

### Important Note: Multiple Strengths

**The ATC scraper structure stores ONE drug entry per drug name per ATC code.**

Looking at the scraper code (line 793-799 in `src/atc_scraper.py`), it uses:
```python
drugs[drug_name] = {
    "atc_code": atc_code,
    "form_strength": form_strength,
    "documents": documents
}
```

**If there are multiple rows with the same drug name "Íbúfen" but different strengths**, the scraper processes each row sequentially, and **only the LAST one would be stored** because Python dictionaries can't have duplicate keys - each assignment overwrites the previous one.

So if the website shows three rows:
- Row 1: Íbúfen - Filmuhúðuð tafla / 200 mg
- Row 2: Íbúfen - Filmuhúðuð tafla / 400 mg  
- Row 3: Íbúfen - Filmuhúðuð tafla / 600 mg

The scraper would process all three rows, but only store:
```json
"drugs": {
  "Íbúfen": {
    "atc_code": "M01AE01",
    "form_strength": "Filmuhúðuð tafla / 600 mg",  // Only the last one!
    "documents": [
      {
        "type": "Fylgiseðill",
        "url": "...",
        "date": "25.3.2025"
      },
      {
        "type": "SmPC",
        "url": "...",
        "date": "25.3.2025"
      }
    ]
  }
}
```

**The 200 mg and 400 mg entries would be lost.**

### In the drug_mappings Structure

The `drug_mappings` would contain:

```json
{
  "drug_mappings": {
    "Íbúfen": ["M01AE01"]
  }
}
```

Again, if there are multiple entries with the same name, only one would appear.

## The Problem

The current ATC scraper structure **doesn't handle multiple strengths/formulations of the same drug name well**. If the website shows:

- Íbúfen - 200 mg
- Íbúfen - 400 mg  
- Íbúfen - 600 mg

The scraper would likely only capture the **last one** (600 mg) because they all have the same drug name "Íbúfen" and Python dictionaries can't have duplicate keys.

## Recommendation

To properly capture multiple strengths, the scraper should either:

1. **Use a composite key** like `"Íbúfen_200mg"`, `"Íbúfen_400mg"`, `"Íbúfen_600mg"`
2. **Store as an array** of formulations under a single drug name
3. **Include strength in the drug name** when extracting from the table

## Why M Section is Missing

The M section scraping likely failed during the initial scrape. Possible reasons:
- JavaScript rendering issues
- Timeout during scraping
- Website structure changes
- Network issues

**To fix this, you would need to re-run the ATC scraper and ensure the M section is successfully scraped.**

