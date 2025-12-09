# ATC Classification Scraper

Scrapes the 5-level Anatomical Therapeutic Chemical (ATC) classification from the Icelandic Medicines Agency website (https://old.serlyfjaskra.is/).

## Features

- Scrapes complete 5-level ATC hierarchy
- Extracts drug names, formulations, and associated documents
- Outputs structured JSON matching the specified format
- Supports headless mode for server environments

## Requirements

- Python 3.7+
- Chrome browser and ChromeDriver
- Selenium

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install ChromeDriver:
   - **macOS**: `brew install chromedriver`
   - **Linux**: Download from https://chromedriver.chromium.org/
   - **Windows**: Download from https://chromedriver.chromium.org/

## Usage

Basic usage:
```bash
python scrape_atc.py
```

Specify output file:
```bash
python scrape_atc.py -o output.json
```

Run in headless mode (no browser window):
```bash
python scrape_atc.py --headless
```

## Output Structure

The script generates a JSON file with the following structure:

```json
{
  "hierarchy": {
    "A": {
      "code": "A",
      "name": "MELTINGARFÆRA- OG EFNASKIPTALYF",
      "level2": {
        "A01": {
          "code": "A01",
          "name": "MUNN- OG TANNLYF",
          "level3": {
            ...
          }
        }
      }
    }
  }
}
```

## Notes

- The website uses Telerik RadGrid controls with server-side AJAX loading
- Scraping the complete hierarchy can take significant time (30+ minutes)
- The script includes delays to avoid overwhelming the server
- Some drug/formulation details may require manual verification

## Troubleshooting

**ChromeDriver version mismatch:**
Ensure ChromeDriver version matches your Chrome browser version.

**Timeout errors:**
Increase wait times in the script or check your internet connection.

**Missing data:**
The Telerik controls may require specific selectors - inspect the page and adjust CSS selectors as needed.
