#!/usr/bin/env python3
"""
ATC Classification Scraper v2 for old.serlyfjaskra.is
Improved version with better Telerik RadGrid handling
"""

import json
import time
import re
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
from typing import Dict, List, Optional
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ATCScraperV2:
    def __init__(self, headless: bool = False, debug: bool = False):
        """Initialize the scraper with Selenium WebDriver"""
        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')

        self.driver = webdriver.Chrome(options=options)
        self.driver.implicitly_wait(5)
        self.base_url = "https://old.serlyfjaskra.is/ATCList.aspx?d=1&a=0"
        self.wait = WebDriverWait(self.driver, 20)
        self.hierarchy = {}
        self.debug = debug

        if debug:
            self.debug_dir = Path("debug_html")
            self.debug_dir.mkdir(exist_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.driver.quit()

    def save_debug_html(self, name: str):
        """Save current page HTML for debugging"""
        if self.debug:
            file_path = self.debug_dir / f"{name}.html"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(self.driver.page_source)
            logger.debug(f"Saved debug HTML to {file_path}")

    def scrape_all(self) -> Dict:
        """Scrape the entire ATC classification tree"""
        logger.info("Starting ATC classification scrape (v2)")

        # Navigate to main page
        self.driver.get(self.base_url)
        time.sleep(2)
        self.save_debug_html("main_page")

        # Get all letter categories
        letters = self._get_letter_categories()

        # For testing, you can limit to specific letters
        # letters = ['A']  # Uncomment to test with just 'A'

        for letter in letters:
            logger.info(f"=" * 60)
            logger.info(f"Processing category: {letter}")
            logger.info(f"=" * 60)
            self._scrape_category(letter)

        return {"hierarchy": self.hierarchy}

    def _get_letter_categories(self) -> List[str]:
        """Get all available letter categories from the main page"""
        try:
            # Look for links with FirstLetter parameter
            links = self.driver.find_elements(By.CSS_SELECTOR, "a[href*='FirstLetter=']")
            letters = set()

            for link in links:
                href = link.get_attribute('href')
                if href and 'FirstLetter=' in href:
                    match = re.search(r'FirstLetter=([A-Z])', href)
                    if match:
                        letters.add(match.group(1))

            sorted_letters = sorted(list(letters))
            logger.info(f"Found {len(sorted_letters)} categories: {', '.join(sorted_letters)}")
            return sorted_letters

        except Exception as e:
            logger.error(f"Error getting letter categories: {e}")
            return []

    def _scrape_category(self, letter: str):
        """Scrape a complete category (Level 1)"""
        url = f"https://old.serlyfjaskra.is/ATCList.aspx?FirstLetter={letter}&d=1"
        logger.info(f"Loading URL: {url}")

        self.driver.get(url)
        time.sleep(3)
        self.save_debug_html(f"category_{letter}")

        try:
            # Wait for the RadGrid to load
            grid = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.RadGrid, table.rgMasterTable"))
            )
            logger.info(f"Grid loaded for category {letter}")

            # Get the category name from the page
            category_name = self._extract_category_name()
            logger.info(f"Category name: {category_name}")

            # Initialize level 1 entry
            self.hierarchy[letter] = {
                "code": letter,
                "name": category_name,
                "level2": {}
            }

            # Extract all level 2 entries
            self._extract_level2_entries(letter)

        except TimeoutException:
            logger.warning(f"Timeout waiting for grid in category {letter}")
        except Exception as e:
            logger.error(f"Error scraping category {letter}: {e}", exc_info=True)

    def _extract_category_name(self) -> str:
        """Extract the category name from the page"""
        try:
            # Try different selectors for the category name
            selectors = [
                "h1",
                "h2",
                ".PageTitle",
                "span.title"
            ]

            for selector in selectors:
                try:
                    element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    text = element.text.strip()
                    if text:
                        return text
                except NoSuchElementException:
                    continue

            return ""

        except Exception as e:
            logger.warning(f"Could not extract category name: {e}")
            return ""

    def _extract_level2_entries(self, letter: str):
        """Extract Level 2 entries (e.g., A01, A02)"""
        try:
            # Find all master table rows
            rows = self.driver.find_elements(By.CSS_SELECTOR, "table.rgMasterTable tbody tr.rgRow, table.rgMasterTable tbody tr.rgAltRow")
            logger.info(f"Found {len(rows)} level 2 rows")

            for idx, row in enumerate(rows):
                try:
                    self._process_grid_row(row, letter, 2)
                except StaleElementReferenceException:
                    logger.warning(f"Stale element at row {idx}, re-finding...")
                    # Re-find the rows and continue
                    rows = self.driver.find_elements(By.CSS_SELECTOR, "table.rgMasterTable tbody tr.rgRow, table.rgMasterTable tbody tr.rgAltRow")
                    if idx < len(rows):
                        self._process_grid_row(rows[idx], letter, 2)
                except Exception as e:
                    logger.warning(f"Error processing level 2 row {idx}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error extracting level 2 entries: {e}", exc_info=True)

    def _process_grid_row(self, row, parent_code: str, level: int):
        """Process a single grid row and extract its data"""
        try:
            cells = row.find_elements(By.TAG_NAME, "td")

            if len(cells) < 2:
                return

            # Find expand button (if any)
            has_children = False
            expand_button = None

            try:
                # Look for expand icon/button in first cell
                expand_button = cells[0].find_element(By.CSS_SELECTOR, "input[type='image'], a")
                expand_src = expand_button.get_attribute('src') if expand_button.tag_name == 'input' else None

                # Check if it's an expand icon (not collapse)
                if expand_src and 'expand' in expand_src.lower():
                    has_children = True
                elif expand_button.tag_name == 'a':
                    has_children = True
            except NoSuchElementException:
                pass

            # Extract code and name from cells
            # Usually: Cell 0 = expand button, Cell 1 = code, Cell 2 = name
            code = ""
            name = ""

            for cell_idx, cell in enumerate(cells[1:], 1):  # Skip first cell (expand button)
                text = cell.text.strip()
                if not text:
                    continue

                if not code:
                    code = text
                elif not name:
                    name = text
                    break

            if not code:
                return

            logger.info(f"  Level {level}: {code} - {name[:50]}... (has_children: {has_children})")

            # Create entry structure
            entry = {
                "code": code,
                "name": name
            }

            # Add to hierarchy
            if level == 2:
                self.hierarchy[parent_code]["level2"][code] = entry
                entry["level3"] = {}
            elif level == 3:
                # Navigate to parent and add
                if parent_code in self.hierarchy:
                    # Find the level2 parent
                    for l2_code, l2_data in self.hierarchy[parent_code]["level2"].items():
                        if code.startswith(l2_code):
                            l2_data["level3"][code] = entry
                            entry["level4"] = {}
                            break
            # Similar logic for levels 4 and 5...

            # If has children, expand and process
            if has_children and expand_button and level < 5:
                try:
                    logger.info(f"    Expanding {code}...")
                    expand_button.click()
                    time.sleep(1.5)

                    # Find the detail table that appears
                    detail_rows = self._find_detail_rows(row)
                    logger.info(f"    Found {len(detail_rows)} child rows")

                    for detail_row in detail_rows:
                        try:
                            self._process_grid_row(detail_row, code, level + 1)
                        except Exception as e:
                            logger.warning(f"    Error processing child row: {e}")

                except Exception as e:
                    logger.warning(f"    Could not expand {code}: {e}")

        except Exception as e:
            logger.warning(f"Error processing row: {e}")

    def _find_detail_rows(self, parent_row):
        """Find detail table rows that appear after expanding a row"""
        try:
            # Telerik RadGrid detail tables usually appear as a nested table
            # Try to find the next sibling tr that contains a detail table
            detail_table = parent_row.find_element(By.XPATH, "./following-sibling::tr[1]//table[@class='rgDetailTable']")
            if detail_table:
                return detail_table.find_elements(By.CSS_SELECTOR, "tbody tr.rgRow, tbody tr.rgAltRow")
        except NoSuchElementException:
            pass

        return []

    def _extract_drugs(self, row_element) -> Optional[Dict]:
        """Extract drug information from a level 5 entry"""
        drugs = {}

        try:
            # Look for product links
            links = row_element.find_elements(By.CSS_SELECTOR, "a[href*='ProductView.aspx']")

            for link in links:
                drug_name = link.text.strip()
                if not drug_name:
                    continue

                product_id = self._extract_product_id(link.get_attribute('href'))

                drugs[drug_name] = {
                    "atc_code": "",
                    "formulations": self._extract_formulations(row_element, product_id)
                }

        except Exception as e:
            logger.warning(f"Error extracting drugs: {e}")

        return drugs if drugs else None

    def _extract_product_id(self, url: str) -> str:
        """Extract product ID from URL"""
        match = re.search(r'ProductId=(\d+)', url)
        return match.group(1) if match else ""

    def _extract_formulations(self, row_element, product_id: str) -> List[Dict]:
        """Extract formulation and document information"""
        formulations = []

        try:
            # Find document links in the row
            doc_links = row_element.find_elements(By.CSS_SELECTOR, "a")

            documents = []
            for link in doc_links:
                text = link.text.strip().lower()
                url = link.get_attribute('href')

                if not url:
                    continue

                doc_type = None
                if 'fylgiseðill' in text or 'patient' in text:
                    doc_type = "Fylgiseðill"
                elif 'smpc' in text or 'summary' in text:
                    doc_type = "SmPC"

                if doc_type:
                    documents.append({
                        "type": doc_type,
                        "url": url,
                        "date": ""
                    })

            if documents:
                formulations.append({
                    "form_strength": "",
                    "documents": documents
                })

        except Exception as e:
            logger.warning(f"Error extracting formulations: {e}")

        return formulations

    def save_to_file(self, output_file: str):
        """Save the scraped hierarchy to a JSON file"""
        data = {"hierarchy": self.hierarchy}

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Data saved to {output_file}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Scrape ATC classification from serlyfjaskra.is')
    parser.add_argument('-o', '--output', default='atc_classification.json',
                        help='Output JSON file (default: atc_classification.json)')
    parser.add_argument('--headless', action='store_true',
                        help='Run browser in headless mode')
    parser.add_argument('--debug', action='store_true',
                        help='Save HTML files for debugging')

    args = parser.parse_args()

    try:
        with ATCScraperV2(headless=args.headless, debug=args.debug) as scraper:
            logger.info("Starting scrape...")
            scraper.scrape_all()
            scraper.save_to_file(args.output)
            logger.info("Scraping completed successfully!")

    except KeyboardInterrupt:
        logger.info("Scraping interrupted by user")
    except Exception as e:
        logger.error(f"Scraping failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
