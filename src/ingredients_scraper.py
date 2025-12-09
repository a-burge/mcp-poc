"""Ingredients index scraper for Icelandic Medicines Agency website."""
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add parent directory to path for imports when running as script
# This must happen before importing config
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import requests
from bs4 import BeautifulSoup

from config import Config

logger = logging.getLogger(__name__)


class IngredientsScraper:
    """Scraper for Ingredients index from Icelandic Medicines Agency website."""
    
    BASE_URL = "https://old.serlyfjaskra.is"
    INGREDIENTS_URL = f"{BASE_URL}/Ingredients.aspx?d=1&a=0"
    
    def __init__(self):
        """Initialize the scraper."""
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        })
        self.ingredients: Dict[str, Any] = {}
        self.drug_to_ingredients: Dict[str, List[str]] = {}
    
    def scrape_with_selenium(self) -> Dict[str, Any]:
        """
        Scrape using Selenium for JavaScript-heavy pages.
        
        This method requires selenium and a webdriver.
        
        Returns:
            Dictionary with 'ingredients' and 'drug_to_ingredients' keys
        """
        try:
            from selenium import webdriver
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            from selenium.webdriver.chrome.options import Options
            from selenium.common.exceptions import TimeoutException, NoSuchElementException
        except ImportError:
            logger.error("Selenium not installed. Install with: pip install selenium")
            raise ImportError("Selenium is required for JavaScript-heavy scraping")
        
        logger.info("Starting Ingredients scraping with Selenium...")
        
        # Setup Chrome options
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        
        driver = None
        try:
            driver = webdriver.Chrome(options=chrome_options)
            driver.get(self.INGREDIENTS_URL)
            
            # Wait for page to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Extract ingredients by clicking through alphabet letters
            # Based on the website, alphabet includes: A-Z and numbers (2)
            alphabet_letters = [
                '2', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 
                'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Z'
            ]
            
            for letter in alphabet_letters:
                logger.info(f"Processing letter: {letter}")
                try:
                    # Find and click the link for this letter
                    link = driver.find_element(By.LINK_TEXT, letter)
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", link)
                    time.sleep(0.3)
                    link.click()
                    
                    # Wait for result grid to appear
                    try:
                        WebDriverWait(driver, 10).until(
                            EC.presence_of_element_located(
                                (By.XPATH, "//table[contains(@id, 'resultGrid') and contains(@class, 'rgMasterTable')]")
                            )
                        )
                    except TimeoutException:
                        logger.warning(f"Timeout waiting for result grid after clicking {letter}")
                        time.sleep(2)
                    
                    # Find the main result grid table
                    try:
                        result_grid = driver.find_element(
                            By.XPATH,
                            "//table[contains(@id, 'resultGrid') and contains(@class, 'rgMasterTable')]"
                        )
                        
                        # Extract ingredients for this letter
                        letter_ingredients = self._scrape_letter(driver, result_grid, letter)
                        
                        # Merge into main ingredients dictionary
                        self.ingredients.update(letter_ingredients)
                        logger.info(f"  Extracted {len(letter_ingredients)} ingredients for letter {letter}")
                        
                    except Exception as e:
                        logger.warning(f"Could not find result grid for {letter}: {e}", exc_info=True)
                        # Try alternative selector
                        try:
                            result_grid = driver.find_element(
                                By.XPATH,
                                "//table[@class='rgMasterTable']"
                            )
                            letter_ingredients = self._scrape_letter(driver, result_grid, letter)
                            self.ingredients.update(letter_ingredients)
                            logger.info(f"  Extracted {len(letter_ingredients)} ingredients using alternative selector")
                        except Exception as e2:
                            logger.warning(f"Alternative selector also failed for {letter}: {e2}")
                    
                    # Go back to main page
                    driver.get(self.INGREDIENTS_URL)
                    time.sleep(1.5)
                    
                except Exception as e:
                    logger.warning(f"Error processing letter {letter}: {e}", exc_info=True)
                    # Try to recover by going back to main page
                    try:
                        driver.get(self.INGREDIENTS_URL)
                        time.sleep(1)
                    except:
                        pass
                    continue
            
            # Build reverse mapping: drug_to_ingredients
            logger.info("Building drug_to_ingredients mapping...")
            for ingredient_key, ingredient_data in self.ingredients.items():
                drugs = ingredient_data.get("drugs", {})
                for drug_name in drugs.keys():
                    if drug_name not in self.drug_to_ingredients:
                        self.drug_to_ingredients[drug_name] = []
                    if ingredient_key not in self.drug_to_ingredients[drug_name]:
                        self.drug_to_ingredients[drug_name].append(ingredient_key)
            
            logger.info(f"Scraped {len(self.ingredients)} ingredients")
            logger.info(f"Found {len(self.drug_to_ingredients)} drug-to-ingredient mappings")
            
            return {
                "ingredients": self.ingredients,
                "drug_to_ingredients": self.drug_to_ingredients,
                "scraped_at": time.strftime("%Y-%m-%dT%H:%M:%S")
            }
            
        finally:
            if driver:
                driver.quit()
    
    def _scrape_letter(
        self,
        driver,
        result_grid,
        letter: str
    ) -> Dict[str, Any]:
        """
        Scrape all ingredients starting with a specific letter, handling pagination.
        
        Args:
            driver: Selenium WebDriver instance
            result_grid: The main result grid table element
            letter: The alphabet letter being processed
            
        Returns:
            Dictionary mapping ingredient keys to their data
        """
        try:
            from selenium.webdriver.common.by import By
        except ImportError:
            logger.error("Selenium not available in _scrape_letter")
            return {}
        
        all_ingredients: Dict[str, Any] = {}
        processed_ingredients: set = set()
        
        # Scrape first page
        page_ingredients = self._scrape_current_page(
            driver, result_grid, letter, processed_ingredients
        )
        all_ingredients.update(page_ingredients)
        
        # Check for pagination
        total_pages = self._get_total_pages(driver)
        
        if total_pages > 1:
            logger.info(f"  Letter {letter} has {total_pages} pages")
            
            for page_num in range(2, total_pages + 1):
                # Navigate to the page
                if not self._navigate_to_page(driver, page_num):
                    logger.warning(f"  Failed to navigate to page {page_num} for letter {letter}")
                    continue
                
                # Wait for grid to update
                time.sleep(1.0)
                
                # Re-find result grid (DOM may have changed after AJAX update)
                try:
                    result_grid = driver.find_element(
                        By.XPATH,
                        "//table[contains(@id, 'resultGrid') and contains(@class, 'rgMasterTable')]"
                    )
                except Exception as e:
                    logger.warning(f"  Could not find result grid after navigating to page {page_num}: {e}")
                    continue
                
                # Scrape this page
                page_ingredients = self._scrape_current_page(
                    driver, result_grid, letter, processed_ingredients
                )
                all_ingredients.update(page_ingredients)
                logger.info(f"  Page {page_num}: extracted {len(page_ingredients)} ingredients")
        
        return all_ingredients
    
    def _scrape_current_page(
        self,
        driver,
        result_grid,
        letter: str,
        processed_ingredients: set
    ) -> Dict[str, Any]:
        """
        Scrape ingredients from the current page of results.
        
        Args:
            driver: Selenium WebDriver instance
            result_grid: The main result grid table element
            letter: The alphabet letter being processed
            processed_ingredients: Set of already processed ingredient keys (modified in-place)
            
        Returns:
            Dictionary mapping ingredient keys to their data
        """
        try:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            from selenium.common.exceptions import NoSuchElementException, TimeoutException, StaleElementReferenceException
        except ImportError:
            logger.error("Selenium not available in _scrape_current_page")
            return {}
        
        ingredients = {}
        
        try:
            # Find all ingredient rows in the result grid
            # Rows have classes rgRow or rgAltRow, and are in tbody
            # Exclude detail rows (rows that are inside detail tables)
            row_xpath = ".//tbody/tr[(contains(@class, 'rgRow') or contains(@class, 'rgAltRow')) and not(ancestor::table[contains(@class, 'rgDetailTable')])]"
            rows = result_grid.find_elements(By.XPATH, row_xpath)
            
            logger.debug(f"Found {len(rows)} potential ingredient rows for letter {letter}")
            
            # Step 1: Collect all ingredient names first (without expanding to avoid stale elements)
            ingredient_names = []
            for row in rows:
                try:
                    cell = row.find_element(By.XPATH, ".//td[3]")
                    cell_text = cell.text.strip()
                    if cell_text:
                        # Extract INN name and code
                        if " INN" in cell_text:
                            parts = cell_text.split(" INN")
                            inn_name = parts[0].strip()
                            inn_code = "INN"
                        else:
                            inn_name = cell_text
                            inn_code = ""
                        
                        ingredient_names.append({
                            "key": cell_text,
                            "inn_name": inn_name,
                            "inn_code": inn_code
                        })
                except (StaleElementReferenceException, NoSuchElementException):
                    continue
                except Exception as e:
                    logger.debug(f"Error collecting ingredient name: {e}")
                    continue
            
            logger.debug(f"Collected {len(ingredient_names)} ingredient names")
            
            # Step 2: Process each ingredient individually, re-finding rows after each expansion
            for ingredient_info in ingredient_names:
                ingredient_key = ingredient_info["key"]
                
                if ingredient_key in processed_ingredients:
                    continue
                
                processed_ingredients.add(ingredient_key)
                logger.debug(f"Processing ingredient: {ingredient_key}")
                
                ingredient_data = {
                    "inn_name": ingredient_info["inn_name"],
                    "inn_code": ingredient_info["inn_code"],
                    "drugs": {}
                }
                
                # Re-find the row by ingredient name (to avoid stale element)
                try:
                    # Re-find result grid and find row with this ingredient name
                    result_grid = driver.find_element(
                        By.XPATH,
                        "//table[contains(@id, 'resultGrid') and contains(@class, 'rgMasterTable')]"
                    )
                    row = result_grid.find_element(
                        By.XPATH,
                        f".//tbody/tr[(contains(@class, 'rgRow') or contains(@class, 'rgAltRow')) and td[3]='{ingredient_key}']"
                    )
                    
                    # Get row ID before any DOM changes
                    row_id = row.get_attribute("id")
                    
                    # Check if row has expand button (has drugs)
                    has_expand_btn = False
                    try:
                        expand_btn = row.find_element(
                            By.XPATH,
                            ".//td[1]//input[contains(@class, 'rgExpand')]"
                        )
                        has_expand_btn = True
                        
                        # Expand the row
                        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", expand_btn)
                        time.sleep(0.3)
                        expand_btn.click()
                        
                        # Wait for detail table
                        if row_id:
                            WebDriverWait(driver, 5).until(
                                EC.presence_of_element_located(
                                    (By.XPATH, f"//tr[@id='{row_id}']/following-sibling::tr[1]//table[contains(@class, 'rgDetailTable')]")
                                )
                            )
                        time.sleep(0.8)
                        
                    except NoSuchElementException:
                        # Check if already expanded (has collapse button)
                        try:
                            row.find_element(By.XPATH, ".//td[1]//input[contains(@class, 'rgCollapse')]")
                            # Already expanded
                            has_expand_btn = True
                        except:
                            # No expand or collapse button - might not have drugs
                            pass
                    
                    # Extract drugs if row was expandable (has drugs)
                    # Re-find row by ID to avoid stale element reference
                    if has_expand_btn and row_id:
                        try:
                            # Re-find the row by ID after expansion
                            fresh_row = driver.find_element(By.XPATH, f"//tr[@id='{row_id}']")
                            drugs = self._extract_ingredient_and_drugs(driver, fresh_row, ingredient_key, row_id)
                            if drugs:
                                ingredient_data["drugs"] = drugs
                                logger.debug(f"  Extracted {len(drugs)} drugs for {ingredient_key}")
                        except Exception as e:
                            logger.debug(f"  Error extracting drugs for {ingredient_key}: {e}")
                        
                except Exception as e:
                    logger.debug(f"Error processing ingredient {ingredient_key}: {e}", exc_info=True)
                    continue
                
                ingredients[ingredient_key] = ingredient_data
                logger.debug(f"  Added ingredient: {ingredient_key} with {len(ingredient_data.get('drugs', {}))} drugs")
            
            return ingredients
            
        except Exception as e:
            logger.warning(f"Error scraping current page for letter {letter}: {e}", exc_info=True)
            return ingredients
    
    def _get_total_pages(self, driver) -> int:
        """
        Get total number of pages from pagination info.
        
        Parses the pagination text (e.g., "71 Innihaldsefni á 2 síðum") to extract
        the total page count, with fallback to counting page number links.
        
        Args:
            driver: Selenium WebDriver instance
            
        Returns:
            Total number of pages (minimum 1)
        """
        try:
            from selenium.webdriver.common.by import By
        except ImportError:
            return 1
        
        try:
            # Method 1: Parse "X Innihaldsefni á Y síðum" from .rgInfoPart
            info_divs = driver.find_elements(By.CSS_SELECTOR, ".rgInfoPart")
            for info_div in info_divs:
                text = info_div.text.strip()
                # Match patterns like "á 2 síðum" or "á 10 síðum"
                match = re.search(r'á\s+(\d+)\s+síðum', text)
                if match:
                    total_pages = int(match.group(1))
                    logger.debug(f"Found {total_pages} pages from pagination text: {text}")
                    return total_pages
            
            # Method 2: Count page number links in .rgNumPart
            page_links = driver.find_elements(By.CSS_SELECTOR, ".rgNumPart a")
            if page_links:
                # Count unique page numbers
                page_count = len(page_links)
                logger.debug(f"Found {page_count} pages from page links")
                return page_count
            
        except Exception as e:
            logger.debug(f"Could not determine page count: {e}")
        
        return 1  # Default to 1 page if can't determine
    
    def _navigate_to_page(self, driver, page_num: int) -> bool:
        """
        Navigate to a specific page number in the pagination.
        
        Args:
            driver: Selenium WebDriver instance
            page_num: The page number to navigate to (1-based)
            
        Returns:
            True if navigation succeeded, False otherwise
        """
        try:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
        except ImportError:
            return False
        
        try:
            # Dismiss cookie banner if present (it can block clicks)
            try:
                cookie_banner = driver.find_element(By.CSS_SELECTOR, ".cookie-disclaimer")
                if cookie_banner.is_displayed():
                    # Try to find and click accept/close button, or hide via JS
                    driver.execute_script("arguments[0].style.display = 'none';", cookie_banner)
                    time.sleep(0.2)
            except Exception:
                pass  # No cookie banner or already dismissed
            
            # Find the page number link in .rgNumPart
            # The link contains a span with the page number text
            page_link = driver.find_element(
                By.XPATH,
                f"//div[contains(@class, 'rgNumPart')]//a[span[text()='{page_num}']]"
            )
            
            # Scroll into view and click
            driver.execute_script(
                "arguments[0].scrollIntoView({block: 'center'});", 
                page_link
            )
            time.sleep(0.3)
            page_link.click()
            
            # Wait for the grid to update (the page link should become current)
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((
                        By.XPATH,
                        f"//div[contains(@class, 'rgNumPart')]//a[contains(@class, 'rgCurrentPage')][span[text()='{page_num}']]"
                    ))
                )
            except Exception:
                # Fallback: just wait for grid to be present
                time.sleep(1.5)
            
            logger.debug(f"Successfully navigated to page {page_num}")
            return True
            
        except Exception as e:
            logger.warning(f"Could not navigate to page {page_num}: {e}")
            return False
    
    def _extract_ingredient_and_drugs(
        self,
        driver,
        row,
        ingredient_key: str,
        row_id: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract drug information from an ingredient row.
        
        Args:
            driver: Selenium WebDriver instance
            row: The table row element for the ingredient
            ingredient_key: The ingredient key (e.g., "Ibuprofenum INN")
            
        Returns:
            Dictionary mapping drug names to their information
        """
        try:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            from selenium.common.exceptions import NoSuchElementException, TimeoutException
        except ImportError:
            logger.error("Selenium not available in _extract_ingredient_and_drugs")
            return {}
        
        drugs = {}
        
        try:
            # Use provided row_id or get it from row
            if not row_id:
                try:
                    row_id = row.get_attribute("id")
                except:
                    pass
            
            # First, ensure the ingredient row is expanded
            try:
                expand_btn = row.find_element(
                    By.XPATH,
                    ".//td[1]//input[contains(@class, 'rgExpand')]"
                )
                # Check if already expanded
                btn_class = expand_btn.get_attribute("class") or ""
                is_expanded = "rgCollapse" in btn_class
                
                if not is_expanded:
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", expand_btn)
                    time.sleep(0.3)
                    expand_btn.click()
                    # Wait for AJAX expansion
                    try:
                        if row_id:
                            WebDriverWait(driver, 5).until(
                                EC.presence_of_element_located(
                                    (By.XPATH, f"//tr[@id='{row_id}']/following-sibling::tr[1]//table[contains(@class, 'rgDetailTable')]")
                                )
                            )
                        else:
                            time.sleep(2)
                    except TimeoutException:
                        time.sleep(2)
                    except:
                        time.sleep(2)
            except NoSuchElementException:
                # Already expanded or no expand button - check if detail table already exists
                pass
            
            # Find the nested Detail table that contains drug information
            time.sleep(0.8)
            
            detail_table = None
            
            # Method 1: Find following sibling tr with Detail table using row ID
            if row_id:
                try:
                    detail_table = driver.find_element(
                        By.XPATH,
                        f"//tr[@id='{row_id}']/following-sibling::tr[1]//table[contains(@class, 'rgDetailTable')]"
                    )
                    logger.debug(f"  Found drug detail table for {ingredient_key} using Method 1")
                except Exception as e:
                    logger.debug(f"  Method 1 failed for {ingredient_key}: {e}")
                    pass
            
            # Method 2: Find from parent using XPath with preceding-sibling
            if not detail_table and row_id:
                try:
                    parent_table = row.find_element(By.XPATH, "./ancestor::table[contains(@class, 'rgDetailTable')][1]")
                    detail_table = parent_table.find_element(
                        By.XPATH,
                        f".//tr[preceding-sibling::tr[@id='{row_id}']][1]//table[contains(@class, 'rgDetailTable')]"
                    )
                    logger.debug(f"  Found drug detail table for {ingredient_key} using Method 2")
                except Exception as e:
                    logger.debug(f"  Method 2 failed for {ingredient_key}: {e}")
                    pass
            
            # Method 3: Try finding by following sibling from row element
            if not detail_table:
                try:
                    next_row = row.find_element(By.XPATH, "./following-sibling::tr[1]")
                    detail_table = next_row.find_element(
                        By.XPATH,
                        ".//table[contains(@class, 'rgDetailTable')]"
                    )
                    logger.debug(f"  Found drug detail table for {ingredient_key} using Method 3")
                except Exception as e:
                    logger.debug(f"  Method 3 failed for {ingredient_key}: {e}")
                    pass
            
            if not detail_table:
                logger.warning(f"No detail table found for ingredient {ingredient_key} (row_id={row_id}) - skipping drug extraction")
                return drugs
            
            # Find all drug rows in the detail table
            drug_rows = detail_table.find_elements(
                By.XPATH,
                ".//tbody/tr[contains(@class, 'rgRow') or contains(@class, 'rgAltRow')]"
            )
            
            logger.debug(f"Found {len(drug_rows)} drug rows for {ingredient_key}")
            
            for drug_row in drug_rows:
                try:
                    # Drug name is in column 1 (td[1]) with class "productlink"
                    # Based on the HTML structure provided
                    try:
                        drug_name_link = drug_row.find_element(
                            By.XPATH,
                            ".//td[1]//a[contains(@class, 'productlink')]"
                        )
                        drug_name = drug_name_link.text.strip()
                    except:
                        # Fallback: get text from column 1
                        try:
                            drug_name_cell = drug_row.find_element(By.XPATH, ".//td[1]")
                            drug_name = drug_name_cell.text.strip()
                        except:
                            continue
                    
                    if not drug_name or len(drug_name) < 2:
                        continue
                    
                    # Form and strength in column 2 (td[2])
                    form_strength = ""
                    try:
                        form_cell = drug_row.find_element(By.XPATH, ".//td[2]")
                        form_strength = form_cell.text.strip()
                    except:
                        pass
                    
                    # Document links in column 3 (td[3]) with class "doc-list"
                    documents = []
                    try:
                        doc_cell = drug_row.find_element(By.XPATH, ".//td[3]")
                        # Look for links in the document cell (in ul.doc-list)
                        links = doc_cell.find_elements(By.XPATH, ".//ul[contains(@class, 'doc-list')]//a")
                        for link in links:
                            doc_text = link.text.strip()
                            doc_url = link.get_attribute("href")
                            if doc_text or (doc_url and (".pdf" in doc_url.lower() or "FileRepos" in doc_url or "ema.europa.eu" in doc_url)):
                                # Get date if available (in em tag with class "date")
                                date_text = ""
                                try:
                                    date_elem = link.find_element(By.XPATH, "./following-sibling::em[contains(@class, 'date')]")
                                    date_text = date_elem.text.strip()
                                except:
                                    # Try parent li then em
                                    try:
                                        parent_li = link.find_element(By.XPATH, "./ancestor::li[1]")
                                        date_elem = parent_li.find_element(By.XPATH, ".//em[contains(@class, 'date')]")
                                        date_text = date_elem.text.strip()
                                    except:
                                        pass
                                
                                documents.append({
                                    "type": doc_text,
                                    "url": doc_url,
                                    "date": date_text
                                })
                    except:
                        pass
                    
                    if drug_name:
                        # Use drug name as key, but store full info
                        drugs[drug_name] = {
                            "form_strength": form_strength,
                            "documents": documents
                        }
                
                except Exception as e:
                    logger.debug(f"Error extracting drug from row: {e}")
                    continue
        
        except Exception as e:
            logger.debug(f"Error extracting drugs for {ingredient_key}: {e}", exc_info=True)
        
        return drugs
    
    def save(self, output_path: Optional[Path] = None) -> None:
        """
        Save scraped data to JSON file.
        
        Args:
            output_path: Optional path to save file (defaults to Config.INGREDIENTS_INDEX_PATH)
        """
        if output_path is None:
            output_path = Config.INGREDIENTS_INDEX_PATH
        
        Config.ensure_directories()
        
        data = {
            "ingredients": self.ingredients,
            "drug_to_ingredients": self.drug_to_ingredients,
            "scraped_at": time.strftime("%Y-%m-%dT%H:%M:%S")
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved Ingredients index to {output_path}")


def scrape_ingredients_index(use_selenium: bool = True) -> Dict[str, Any]:
    """
    Scrape Ingredients index from Icelandic website.
    
    Args:
        use_selenium: If True, use Selenium for JavaScript rendering (required)
        
    Returns:
        Dictionary with scraped ingredients data
    """
    scraper = IngredientsScraper()
    
    if use_selenium:
        data = scraper.scrape_with_selenium()
    else:
        raise ValueError("Ingredients scraper requires Selenium")
    
    scraper.save()
    return data


if __name__ == "__main__":
    import logging as log_module
    
    log_module.basicConfig(level=log_module.INFO)
    
    use_selenium = "--selenium" in sys.argv or True  # Always use Selenium for ingredients
    
    try:
        data = scrape_ingredients_index(use_selenium=use_selenium)
        print(f"Successfully scraped Ingredients index")
        print(f"  - Ingredients: {len(data['ingredients'])}")
        print(f"  - Drug mappings: {len(data['drug_to_ingredients'])}")
    except Exception as e:
        logger.error(f"Failed to scrape Ingredients index: {e}", exc_info=True)
        sys.exit(1)

