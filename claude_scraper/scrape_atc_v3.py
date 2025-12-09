#!/usr/bin/env python3
"""
ATC Classification Scraper v3 for old.serlyfjaskra.is
Handles ASP.NET __doPostBack AJAX calls properly
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
from typing import Dict, List, Optional, Tuple
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ATCScraperV3:
    def __init__(self, headless: bool = False, debug: bool = False):
        """Initialize the scraper with Selenium WebDriver"""
        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')

        self.driver = webdriver.Chrome(options=options)
        self.driver.implicitly_wait(3)
        self.base_url = "https://old.serlyfjaskra.is/ATCList.aspx?d=1&a=0"
        self.wait = WebDriverWait(self.driver, 15)
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

    def wait_for_ajax(self, timeout: int = 10):
        """Wait for ASP.NET AJAX postback to complete"""
        try:
            # Wait for any loading indicators to appear and disappear
            time.sleep(0.5)  # Brief pause for AJAX to start

            # Wait for the page to be ready
            WebDriverWait(self.driver, timeout).until(
                lambda d: d.execute_script('return document.readyState') == 'complete'
            )

            # Additional wait for Telerik AJAX
            time.sleep(0.8)

        except TimeoutException:
            logger.warning("Timeout waiting for AJAX")

    def scrape_all(self) -> Dict:
        """Scrape the entire ATC classification tree"""
        logger.info("Starting ATC classification scrape (v3)")

        # Navigate to main page
        self.driver.get(self.base_url)
        time.sleep(2)
        self.save_debug_html("main_page")

        # Get all letter categories
        letters = self._get_letter_categories()

        # For testing, limit to specific letters
        # letters = ['A']  # Uncomment to test with just 'A'

        for letter in letters:
            logger.info(f"=" * 60)
            logger.info(f"Processing category: {letter}")
            logger.info(f"=" * 60)
            try:
                self._scrape_category(letter)
            except Exception as e:
                logger.error(f"Failed to scrape category {letter}: {e}", exc_info=True)
                continue

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
        self.wait_for_ajax(timeout=15)
        self.save_debug_html(f"category_{letter}")

        try:
            # Wait for the grid to load
            self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "table.rgMasterTable"))
            )
            logger.info(f"Grid loaded for category {letter}")

            # Get the category name
            category_name = self._extract_category_name()
            logger.info(f"Category name: {category_name}")

            # Initialize level 1 entry
            self.hierarchy[letter] = {
                "code": letter,
                "name": category_name,
                "level2": {}
            }

            # Process all level 2 entries
            self._process_level(letter, 2, self.hierarchy[letter]["level2"])

        except TimeoutException:
            logger.warning(f"Timeout waiting for grid in category {letter}")
        except Exception as e:
            logger.error(f"Error scraping category {letter}: {e}", exc_info=True)

    def _extract_category_name(self) -> str:
        """Extract the category name from the page"""
        try:
            # Try to find the category description in the page
            selectors = [
                "span[id*='DescriptionLabel']",
                "span[id*='HeaderLabel']",
                "h1",
                "h2"
            ]

            for selector in selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for element in elements:
                        text = element.text.strip()
                        if text and len(text) > 3:
                            return text
                except NoSuchElementException:
                    continue

            return ""

        except Exception as e:
            logger.warning(f"Could not extract category name: {e}")
            return ""

    def _process_level(self, parent_code: str, level: int, parent_dict: Dict):
        """Process all rows at a given level"""
        try:
            # Find all rows at current level
            rows = self.driver.find_elements(By.CSS_SELECTOR,
                "table.rgMasterTable tbody tr.rgRow, table.rgMasterTable tbody tr.rgAltRow")

            logger.info(f"Level {level}: Found {len(rows)} rows under {parent_code}")

            for idx, row in enumerate(rows):
                try:
                    # Extract row data
                    code, name, expand_btn = self._extract_row_data(row, level)

                    if not code:
                        continue

                    logger.info(f"  [{idx+1}/{len(rows)}] {code}: {name[:60]}...")

                    # Create entry
                    entry = {
                        "code": code,
                        "name": name
                    }

                    # Add to parent dictionary
                    parent_dict[code] = entry

                    # If this is level 5, look for drugs
                    if level == 5:
                        drugs = self._extract_drugs_from_row(row, code)
                        if drugs:
                            entry["drugs"] = drugs
                    else:
                        # Add next level container
                        next_level_key = f"level{level + 1}"
                        entry[next_level_key] = {}

                        # If there's an expand button, expand and recurse
                        if expand_btn:
                            try:
                                logger.info(f"    Expanding {code}...")

                                # Click the expand button
                                expand_btn.click()
                                self.wait_for_ajax()

                                # Check if it expanded by looking at the title attribute
                                title = expand_btn.get_attribute('title')
                                if title and 'collapse' in title.lower():
                                    logger.info(f"    Successfully expanded {code}")

                                    # Find the detail table rows
                                    detail_rows = self._find_detail_rows_after_expand(row)

                                    if detail_rows:
                                        logger.info(f"    Found {len(detail_rows)} child rows")
                                        self._process_detail_rows(detail_rows, code, level + 1, entry[next_level_key])
                                    else:
                                        logger.warning(f"    No detail rows found after expanding {code}")

                                    # Collapse it back to keep page manageable
                                    expand_btn.click()
                                    self.wait_for_ajax()
                                else:
                                    logger.warning(f"    Failed to expand {code} (title: {title})")

                            except StaleElementReferenceException:
                                logger.warning(f"    Stale element when expanding {code}, skipping children")
                            except Exception as e:
                                logger.warning(f"    Error expanding {code}: {e}")

                except StaleElementReferenceException:
                    logger.warning(f"Stale element at index {idx}, continuing...")
                    continue
                except Exception as e:
                    logger.warning(f"Error processing row {idx} at level {level}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error processing level {level}: {e}", exc_info=True)

    def _process_detail_rows(self, rows: List, parent_code: str, level: int, parent_dict: Dict):
        """Process detail table rows (children)"""
        logger.info(f"Level {level}: Processing {len(rows)} detail rows under {parent_code}")

        for idx, row in enumerate(rows):
            try:
                code, name, expand_btn = self._extract_row_data(row, level)

                if not code:
                    continue

                logger.info(f"    [{idx+1}/{len(rows)}] {code}: {name[:60]}...")

                entry = {
                    "code": code,
                    "name": name
                }

                parent_dict[code] = entry

                # If this is level 5, look for drugs
                if level == 5:
                    drugs = self._extract_drugs_from_row(row, code)
                    if drugs:
                        entry["drugs"] = drugs
                else:
                    # Add next level container
                    next_level_key = f"level{level + 1}"
                    entry[next_level_key] = {}

                    # If there's an expand button, expand and recurse
                    if expand_btn:
                        try:
                            logger.info(f"      Expanding {code}...")
                            expand_btn.click()
                            self.wait_for_ajax()

                            title = expand_btn.get_attribute('title')
                            if title and 'collapse' in title.lower():
                                # Find nested detail rows
                                nested_rows = self._find_detail_rows_after_expand(row)
                                if nested_rows:
                                    logger.info(f"      Found {len(nested_rows)} nested rows")
                                    self._process_detail_rows(nested_rows, code, level + 1, entry[next_level_key])

                                # Collapse back
                                expand_btn.click()
                                self.wait_for_ajax()

                        except Exception as e:
                            logger.warning(f"      Error expanding {code}: {e}")

            except Exception as e:
                logger.warning(f"Error processing detail row {idx}: {e}")
                continue

    def _extract_row_data(self, row, level: int) -> Tuple[str, str, Optional[any]]:
        """Extract code, name, and expand button from a row"""
        code = ""
        name = ""
        expand_btn = None

        try:
            cells = row.find_elements(By.TAG_NAME, "td")

            if len(cells) < 2:
                return "", "", None

            # Look for expand button (class rgExpand or rgCollapse)
            try:
                expand_btn = row.find_element(By.CSS_SELECTOR, "input.rgExpand, input.rgCollapse")
            except NoSuchElementException:
                pass

            # Extract code and name from cells
            # Typically: cell 0 = expand button, cell 1 = code, cell 2 = name
            for cell_idx, cell in enumerate(cells):
                text = cell.text.strip()

                # Skip empty cells and cells with just buttons
                if not text or len(text) < 2:
                    continue

                # First non-empty text is usually the code
                if not code:
                    code = text
                # Second non-empty text is the name
                elif not name:
                    name = text
                    break

        except Exception as e:
            logger.warning(f"Error extracting row data: {e}")

        return code, name, expand_btn

    def _find_detail_rows_after_expand(self, parent_row) -> List:
        """Find detail table rows that appear after expanding"""
        try:
            # Method 1: Look for following sibling with nested table
            try:
                detail_row = parent_row.find_element(By.XPATH,
                    "./following-sibling::tr[contains(@class, 'rgDetailRow') or .//table[contains(@class, 'rgDetailTable')]][1]")

                # Find the detail table within
                detail_table = detail_row.find_element(By.CSS_SELECTOR, "table.rgDetailTable")
                rows = detail_table.find_elements(By.CSS_SELECTOR, "tbody tr.rgRow, tbody tr.rgAltRow")

                if rows:
                    return rows
            except NoSuchElementException:
                pass

            # Method 2: Look in nested table structure
            try:
                nested_table = parent_row.find_element(By.XPATH,
                    "./following-sibling::tr[1]//table[@class='rgDetailTable']")
                rows = nested_table.find_elements(By.CSS_SELECTOR, "tbody tr.rgRow, tbody tr.rgAltRow")

                if rows:
                    return rows
            except NoSuchElementException:
                pass

            # Method 3: Look for any detail table in the vicinity
            try:
                detail_tables = self.driver.find_elements(By.CSS_SELECTOR, "table.rgDetailTable")
                for table in detail_tables:
                    # Check if this table is visible
                    if table.is_displayed():
                        rows = table.find_elements(By.CSS_SELECTOR, "tbody tr.rgRow, tbody tr.rgAltRow")
                        if rows:
                            return rows
            except:
                pass

        except Exception as e:
            logger.warning(f"Error finding detail rows: {e}")

        return []

    def _extract_drugs_from_row(self, row, atc_code: str) -> Optional[Dict]:
        """Extract drug information from a level 5 entry"""
        drugs = {}

        try:
            # Look for product view links
            links = row.find_elements(By.CSS_SELECTOR, "a[href*='ProductView.aspx']")

            for link in links:
                drug_name = link.text.strip()
                if not drug_name:
                    continue

                # Extract product ID
                href = link.get_attribute('href')
                product_id = self._extract_product_id(href) if href else ""

                # Look for document links in the same row
                formulations = self._extract_formulations_from_row(row)

                drugs[drug_name] = {
                    "atc_code": atc_code,
                    "formulations": formulations if formulations else []
                }

        except Exception as e:
            logger.warning(f"Error extracting drugs: {e}")

        return drugs if drugs else None

    def _extract_product_id(self, url: str) -> str:
        """Extract product ID from URL"""
        match = re.search(r'ProductId=(\d+)', url)
        return match.group(1) if match else ""

    def _extract_formulations_from_row(self, row) -> List[Dict]:
        """Extract formulation and document information from a row"""
        formulations = []

        try:
            # Find all links in the row
            links = row.find_elements(By.TAG_NAME, "a")

            documents = []
            form_strength = ""

            for link in links:
                text = link.text.strip().lower()
                href = link.get_attribute('href')

                if not href:
                    continue

                # Check for document types
                if 'fylgiseðill' in text or 'patient' in text:
                    documents.append({
                        "type": "Fylgiseðill",
                        "url": href,
                        "date": ""
                    })
                elif 'smpc' in text or 'summary' in text:
                    documents.append({
                        "type": "SmPC",
                        "url": href,
                        "date": ""
                    })

            # Look for formulation strength in cell text
            cells = row.find_elements(By.TAG_NAME, "td")
            for cell in cells:
                text = cell.text.strip()
                # Look for patterns like "10 mg", "500mg/ml", etc.
                if re.search(r'\d+\s*(mg|ml|g|%|mcg)', text, re.IGNORECASE):
                    form_strength = text
                    break

            if documents:
                formulations.append({
                    "form_strength": form_strength,
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

        # Print summary
        level1_count = len(data["hierarchy"])
        level2_count = sum(len(v.get("level2", {})) for v in data["hierarchy"].values())
        logger.info(f"Summary: {level1_count} level 1 categories, {level2_count} level 2 categories")


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
    parser.add_argument('--test-letter', type=str,
                        help='Test with a single letter (e.g., A)')

    args = parser.parse_args()

    try:
        with ATCScraperV3(headless=args.headless, debug=args.debug) as scraper:
            logger.info("Starting scrape...")

            # If testing with single letter
            if args.test_letter:
                logger.info(f"Testing with letter: {args.test_letter}")
                scraper._scrape_category(args.test_letter.upper())
            else:
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
