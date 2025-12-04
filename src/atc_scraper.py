"""ATC index scraper for Icelandic Medicines Agency website."""
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin, urlparse, parse_qs

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


class ATCScraper:
    """Scraper for ATC index from Icelandic Medicines Agency website."""
    
    BASE_URL = "https://old.serlyfjaskra.is"
    ATC_LIST_URL = f"{BASE_URL}/ATCList.aspx?d=1&a=0"
    
    def __init__(self):
        """Initialize the scraper."""
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        })
        self.hierarchy: Dict[str, Any] = {}
        self.drug_mappings: Dict[str, List[str]] = {}
    
    def scrape(self) -> Dict[str, Any]:
        """
        Scrape the complete ATC index and drug mappings.
        
        Returns:
            Dictionary with 'hierarchy' and 'drug_mappings' keys
        """
        logger.info("Starting ATC index scraping...")
        
        try:
            # Get the main page
            response = self.session.get(self.ATC_LIST_URL, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract ATC hierarchy by navigating through the tree
            # The site uses ASP.NET with hierarchical navigation
            # We'll need to click through each level
            
            # Start with level 1 (A, B, C, etc.)
            level1_codes = ['A', 'B', 'C', 'D', 'G', 'H', 'J', 'L', 'M', 'N', 'P', 'R', 'S', 'V']
            
            for code in level1_codes:
                logger.info(f"Processing level 1: {code}")
                level1_data = self._scrape_level1(code, soup)
                if level1_data:
                    self.hierarchy[code] = level1_data
                time.sleep(0.5)  # Be polite to the server
            
            logger.info(f"Scraped {len(self.hierarchy)} level 1 categories")
            logger.info(f"Found {len(self.drug_mappings)} drug mappings")
            
            return {
                "hierarchy": self.hierarchy,
                "drug_mappings": self.drug_mappings,
                "scraped_at": time.strftime("%Y-%m-%dT%H:%M:%S")
            }
            
        except Exception as e:
            logger.error(f"Error scraping ATC index: {e}", exc_info=True)
            raise
    
    def _scrape_level1(self, code: str, soup: BeautifulSoup) -> Optional[Dict[str, Any]]:
        """
        Scrape level 1 ATC category.
        
        Args:
            code: Level 1 code (e.g., 'A', 'G')
            soup: BeautifulSoup object of current page
            
        Returns:
            Dictionary with level 1 data and nested levels
        """
        # Try to find the link for this code
        # The site structure may require POST requests with viewstate
        # For now, we'll try to extract from the current page structure
        
        # Look for links or elements containing the ATC code
        # This is a simplified approach - may need selenium for full functionality
        
        level_data = {
            "code": code,
            "name": self._get_level_name(code),
            "level2": {}
        }
        
        # Try to navigate to level 2 by constructing URLs or making POST requests
        # Since the site uses ASP.NET, we may need to handle viewstate
        # For now, return basic structure - can be enhanced with selenium
        
        return level_data
    
    def _get_level_name(self, code: str) -> str:
        """Get human-readable name for level 1 ATC code."""
        names = {
            'A': 'Alimentary tract and metabolism',
            'B': 'Blood and blood forming organs',
            'C': 'Cardiovascular system',
            'D': 'Dermatologicals',
            'G': 'Genito urinary system and sex hormones',
            'H': 'Systemic hormonal preparations, excluding sex hormones and insulins',
            'J': 'Antiinfectives for systemic use',
            'L': 'Antineoplastic and immunomodulating agents',
            'M': 'Musculo-skeletal system',
            'N': 'Nervous system',
            'P': 'Antiparasitic products, insecticides and repellents',
            'R': 'Respiratory system',
            'S': 'Sensory organs',
            'V': 'Various'
        }
        return names.get(code, f"Unknown category {code}")
    
    def scrape_with_selenium(self) -> Dict[str, Any]:
        """
        Scrape using Selenium for JavaScript-heavy pages.
        
        This method requires selenium and a webdriver.
        Install: pip install selenium
        Download chromedriver or use webdriver-manager
        
        Returns:
            Dictionary with 'hierarchy' and 'drug_mappings' keys
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
        
        logger.info("Starting ATC scraping with Selenium...")
        
        # Setup Chrome options
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        
        driver = None
        try:
            driver = webdriver.Chrome(options=chrome_options)
            driver.get(self.ATC_LIST_URL)
            
            # Wait for page to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Extract ATC hierarchy by clicking through each level
            level1_codes = ['A', 'B', 'C', 'D', 'G', 'H', 'J', 'L', 'M', 'N', 'P', 'R', 'S', 'V']
            
            for code in level1_codes:
                logger.info(f"Processing level 1: {code}")
                try:
                    # Find and click the link for this code
                    # The link is in the alphabet list
                    link = driver.find_element(By.LINK_TEXT, code)
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", link)
                    time.sleep(0.3)
                    link.click()
                    time.sleep(2.5)  # Wait for page load and AJAX
                    
                    # Extract full hierarchy starting from level 1
                    level1_data = {
                        "code": code,
                        "name": self._get_level_name(code),
                        "level2": {}
                    }
                    
                    # Find the main result grid table
                    # The table has id like: ctl00_mainContentPlaceHolder_aTCListCtrl_nestedATCList_resultGrid_ctl00
                    try:
                        result_grid = driver.find_element(
                            By.XPATH,
                            "//table[contains(@id, 'resultGrid') and contains(@class, 'rgMasterTable')]"
                        )
                        
                        # Extract level 2 and below recursively
                        level1_data["level2"] = self._extract_level_recursive(
                            driver, result_grid, code, level=2
                        )
                    except Exception as e:
                        logger.warning(f"Could not find result grid for {code}: {e}", exc_info=True)
                        # Try alternative selector
                        try:
                            result_grid = driver.find_element(
                                By.XPATH,
                                "//table[@class='rgMasterTable']"
                            )
                            level1_data["level2"] = self._extract_level_recursive(
                                driver, result_grid, code, level=2
                            )
                        except Exception as e2:
                            logger.warning(f"Alternative selector also failed for {code}: {e2}")
                    
                    if level1_data.get("level2"):
                        self.hierarchy[code] = level1_data
                        logger.info(f"  Extracted {len(level1_data['level2'])} level 2 categories for {code}")
                    else:
                        logger.warning(f"  No level 2 data extracted for {code}")
                    
                    # Go back to main page
                    driver.get(self.ATC_LIST_URL)
                    time.sleep(1.5)
                    
                except Exception as e:
                    logger.warning(f"Error processing {code}: {e}", exc_info=True)
                    # Try to recover by going back to main page
                    try:
                        driver.get(self.ATC_LIST_URL)
                        time.sleep(1)
                    except:
                        pass
                    continue
            
            logger.info(f"Scraped {len(self.hierarchy)} level 1 categories")
            logger.info(f"Found {len(self.drug_mappings)} drug mappings")
            
            return {
                "hierarchy": self.hierarchy,
                "drug_mappings": self.drug_mappings,
                "scraped_at": time.strftime("%Y-%m-%dT%H:%M:%S")
            }
            
        finally:
            if driver:
                driver.quit()
    
    def _extract_level_recursive(
        self,
        driver,
        parent_element,
        parent_code: str,
        level: int = 2,
        max_level: int = 5
    ) -> Dict[str, Any]:
        """
        Recursively extract ATC hierarchy by expanding each level.
        
        Args:
            driver: Selenium WebDriver instance
            parent_element: Parent table or element containing child rows
            parent_code: ATC code of the parent level (e.g., "A" for level 1)
            level: Current level (2-5)
            max_level: Maximum level to extract (5 for full ATC)
            
        Returns:
            Dictionary mapping ATC codes to their data
        """
        try:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            from selenium.common.exceptions import NoSuchElementException, TimeoutException
        except ImportError:
            logger.error("Selenium not available in _extract_level_recursive")
            return {}
        
        result = {}
        
        if level > max_level:
            return result
        
        try:
            # Find all data rows in the current table
            # Rows have classes rgRow or rgAltRow, and are in tbody
            # Skip header rows and detail rows (which have nested tables)
            # Also skip rows that don't have an ATC code in column 4
            row_xpath = ".//tbody/tr[(contains(@class, 'rgRow') or contains(@class, 'rgAltRow')) and td[4]]"
            rows = parent_element.find_elements(By.XPATH, row_xpath)
            
            logger.debug(f"Found {len(rows)} potential rows at level {level} for {parent_code}")
            
            # Store row IDs and codes first, then process (to avoid stale element issues)
            # We'll re-find each row by ID when processing to avoid DOM staleness
            row_data_list = []
            for i, row in enumerate(rows):
                try:
                    # Check if this row has a nested detail table (already expanded)
                    # If so, skip it as it's a container row
                    try:
                        row.find_element(By.XPATH, ".//table[contains(@class, 'rgDetailTable')]")
                        # This row contains a nested table, skip it
                        logger.debug(f"  Row {i}: Skipping (contains nested table)")
                        continue
                    except NoSuchElementException:
                        pass
                    
                    # Get ATC code and row ID early to validate
                    try:
                        code_cell = row.find_element(By.XPATH, ".//td[4]")
                        code_text = code_cell.text.strip()
                        row_id = row.get_attribute("id")
                    except:
                        logger.debug(f"  Row {i}: No code cell found")
                        continue
                    
                    if not code_text:
                        logger.debug(f"  Row {i}: Empty code text")
                        continue
                    
                    code_clean = code_text.replace(" ", "").upper()
                    expected_lengths = {2: 3, 3: 4, 4: 5, 5: 7}
                    if level in expected_lengths and len(code_clean) != expected_lengths[level]:
                        logger.debug(f"  Row {i}: Code '{code_clean}' length {len(code_clean)} doesn't match expected {expected_lengths[level]} for level {level}")
                        continue
                    
                    # Store row ID and code for later processing (not the row element itself)
                    row_data_list.append((row_id, code_clean))
                    logger.debug(f"  Row {i}: Added '{code_clean}' (id={row_id}) to processing list")
                
                except Exception as e:
                    logger.debug(f"Error pre-processing row {i}: {e}")
                    continue
            
            logger.debug(f"Processing {len(row_data_list)} valid rows at level {level}")
            
            # Now process each row - re-find it fresh each time to avoid stale elements
            for stored_row_id, code_clean in row_data_list:
                try:
                    # Re-find the row element by ID (critical because DOM changes after each expansion)
                    row = None
                    if stored_row_id:
                        try:
                            row = driver.find_element(By.ID, stored_row_id)
                        except:
                            # If can't find by ID, try to find by code in the table
                            try:
                                row = parent_element.find_element(
                                    By.XPATH,
                                    f".//tbody/tr[td[4]='{code_clean}']"
                                )
                            except:
                                logger.debug(f"Could not re-find row for {code_clean} by ID or code, skipping")
                                continue
                    else:
                        # No ID, try to find by code
                        try:
                            row = parent_element.find_element(
                                By.XPATH,
                                f".//tbody/tr[td[4]='{code_clean}']"
                            )
                        except:
                            logger.debug(f"Could not find row for {code_clean} by code, skipping")
                            continue
                    
                    # Get current row ID for later use
                    current_row_id = row.get_attribute("id")
                    
                    # Get name from 5th column (td[5])
                    try:
                        name_cell = row.find_element(By.XPATH, ".//td[5]")
                        name_text = name_cell.text.strip()
                    except:
                        name_text = ""
                    
                    level_data = {
                        "code": code_clean,
                        "name": name_text
                    }
                    
                    # Check if this row has children
                    # In nested tables, td[3] contains the COUNT of children (e.g., "46"), not a boolean "1"
                    # At top level, it might be "1" for has children, "0" for no children
                    has_child = False
                    try:
                        has_child_cell = row.find_element(By.XPATH, ".//td[3]")
                        has_child_value = has_child_cell.get_attribute("textContent") or ""
                        has_child_str = has_child_value.strip()
                        
                        # Check if value is numeric and > 0, or equals "1"
                        try:
                            child_count = int(has_child_str)
                            has_child = child_count > 0
                        except ValueError:
                            # Not numeric, check if it's "1" (boolean true)
                            has_child = has_child_str == "1"
                        
                        logger.debug(f"  {code_clean}: has_child from td[3] = {has_child} (value='{has_child_str}')")
                    except Exception as e:
                        logger.debug(f"  {code_clean}: Could not read has_child from td[3]: {e}")
                        # Try to find expand button as indicator (most reliable)
                        try:
                            expand_btn = row.find_element(
                                By.XPATH, 
                                ".//td[1]//input[contains(@class, 'rgExpand') or contains(@class, 'rgCollapse')]"
                            )
                            has_child = True
                            logger.debug(f"  {code_clean}: has_child = True (found expand button)")
                        except:
                            logger.debug(f"  {code_clean}: has_child = False (no expand button found)")
                            pass
                    
                    # If level 5, extract drugs
                    if level == max_level:
                        drugs = self._extract_drugs_from_level5(driver, row, code_clean)
                        if drugs:
                            level_data["drugs"] = drugs
                            # Add to drug mappings
                            for drug_name in drugs.keys():
                                if drug_name not in self.drug_mappings:
                                    self.drug_mappings[drug_name] = []
                                if code_clean not in self.drug_mappings[drug_name]:
                                    self.drug_mappings[drug_name].append(code_clean)
                    elif has_child:
                        # Try to expand this row to see children
                        try:
                            # Find expand button in first column
                            expand_btn = row.find_element(
                                By.XPATH,
                                ".//td[1]//input[contains(@class, 'rgExpand') or contains(@class, 'rgCollapse')]"
                            )
                            
                            # Check if already expanded
                            btn_class = expand_btn.get_attribute("class") or ""
                            is_expanded = "rgCollapse" in btn_class
                            
                            if not is_expanded:
                                # Scroll into view and click
                                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", expand_btn)
                                time.sleep(0.3)
                                expand_btn.click()
                                # Wait for AJAX expansion - wait for detail table to appear
                                try:
                                    if current_row_id:
                                        WebDriverWait(driver, 5).until(
                                            EC.presence_of_element_located(
                                                (By.XPATH, f"//tr[@id='{current_row_id}']/following-sibling::tr[1]//table[contains(@class, 'rgDetailTable')]")
                                            )
                                        )
                                    else:
                                        time.sleep(2)  # Fallback wait
                                except TimeoutException:
                                    time.sleep(2)  # Fallback wait
                                except:
                                    time.sleep(2)  # Fallback wait
                            
                            # Find the nested Detail table for children
                            # The detail table appears in the following sibling <tr> after expansion
                            # Wait a moment for AJAX to complete
                            time.sleep(0.8)
                            
                            detail_table = None
                            
                            # Method 1: Find following sibling tr with Detail table using row ID (most reliable)
                            if current_row_id:
                                try:
                                    # Find the next sibling row that contains a detail table
                                    # The detail row is the immediate following sibling
                                    detail_table = driver.find_element(
                                        By.XPATH,
                                        f"//tr[@id='{current_row_id}']/following-sibling::tr[1]//table[contains(@class, 'rgDetailTable')]"
                                    )
                                    logger.debug(f"  Found detail table for {code_clean} using Method 1")
                                except Exception as e:
                                    logger.debug(f"  Method 1 failed for {code_clean}: {e}")
                                    pass
                            
                            # Method 2: Find from parent using XPath with preceding-sibling
                            if not detail_table and current_row_id:
                                try:
                                    detail_table = parent_element.find_element(
                                        By.XPATH,
                                        f".//tr[preceding-sibling::tr[@id='{current_row_id}']][1]//table[contains(@class, 'rgDetailTable')]"
                                    )
                                    logger.debug(f"  Found detail table for {code_clean} using Method 2")
                                except Exception as e:
                                    logger.debug(f"  Method 2 failed for {code_clean}: {e}")
                                    pass
                            
                            # Method 3: Find all detail tables and match by position
                            if not detail_table and current_row_id:
                                try:
                                    # Get all rows in tbody to find our row's position
                                    all_tbody_rows = parent_element.find_elements(
                                        By.XPATH,
                                        ".//tbody/tr"
                                    )
                                    
                                    # Find our row's index
                                    our_row_idx = None
                                    for idx, r in enumerate(all_tbody_rows):
                                        if r.get_attribute("id") == current_row_id:
                                            our_row_idx = idx
                                            break
                                    
                                    # If found, check the next row
                                    if our_row_idx is not None and our_row_idx + 1 < len(all_tbody_rows):
                                        next_row = all_tbody_rows[our_row_idx + 1]
                                        try:
                                            detail_table = next_row.find_element(
                                                By.XPATH,
                                                ".//table[contains(@class, 'rgDetailTable')]"
                                            )
                                            logger.debug(f"  Found detail table for {code_clean} using Method 3")
                                        except:
                                            pass
                                except Exception as e:
                                    logger.debug(f"  Method 3 failed for {code_clean}: {e}")
                                    pass
                            
                            if detail_table:
                                # Recursively extract children
                                children = self._extract_level_recursive(
                                    driver, detail_table, code_clean, level + 1, max_level
                                )
                                
                                if children:
                                    level_key = f"level{level + 1}"
                                    level_data[level_key] = children
                                    logger.debug(f"  Extracted {len(children)} children for {code_clean}")
                            else:
                                logger.debug(f"No detail table found for {code_clean} at level {level} (has_child={has_child})")
                        
                        except NoSuchElementException:
                            # No nested table found - might be at leaf level
                            logger.debug(f"No detail table found for {code_clean} at level {level} (NoSuchElementException)")
                            pass
                        except Exception as e:
                            logger.debug(f"Could not expand {code_clean} at level {level}: {e}")
                    
                    result[code_clean] = level_data
                
                except Exception as e:
                    logger.debug(f"Error processing row {code_clean} at level {level}: {e}", exc_info=True)
                    continue
            
            return result
            
        except Exception as e:
            logger.warning(f"Error extracting level {level} for {parent_code}: {e}", exc_info=True)
            return result
    
    def _extract_drugs_from_level5(
        self,
        driver,
        row,
        atc_code: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract drug information from a level 5 ATC row.
        
        Args:
            driver: Selenium WebDriver instance
            row: The table row element for level 5
            atc_code: ATC code for this row (e.g., "A01AA01")
            
        Returns:
            Dictionary mapping drug names to their information
        """
        try:
            from selenium.webdriver.common.by import By
            from selenium.common.exceptions import NoSuchElementException
        except ImportError:
            logger.error("Selenium not available in _extract_drugs_from_level5")
            return {}
        
        drugs = {}
        
        try:
            # First, ensure the level 5 row is expanded
            # Check if it has an expand button and click it if needed
            try:
                expand_btn = row.find_element(
                    By.XPATH,
                    ".//td[1]//input[contains(@class, 'rgExpand')]"
                )
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", expand_btn)
                time.sleep(0.3)
                expand_btn.click()
                time.sleep(1.5)  # Wait for AJAX expansion
            except NoSuchElementException:
                # Already expanded or no expand button
                pass
            
            # Find the nested Detail table that contains drug information
            # The detail table appears in the following sibling <tr> after expansion
            detail_table = None
            
            # Method 1: Find following sibling tr with Detail table
            try:
                next_row = row.find_element(By.XPATH, "./following-sibling::tr[1]")
                detail_table = next_row.find_element(
                    By.XPATH,
                    ".//table[contains(@class, 'rgDetailTable')]"
                )
            except:
                pass
            
            # Method 2: Find Detail table by ID pattern (contains "Detail" and is nested)
            if not detail_table:
                try:
                    row_id = row.get_attribute("id") or ""
                    # Look for detail table in following rows
                    detail_table = driver.find_element(
                        By.XPATH,
                        f"//tr[preceding-sibling::tr[@id='{row_id}']]//table[contains(@class, 'rgDetailTable')]"
                    )
                except:
                    pass
            
            if not detail_table:
                logger.debug(f"No detail table found for level 5 {atc_code}")
                return drugs
            
            # Find all drug rows in the detail table
            # Drug rows have class rgRow or rgAltRow and are in tbody
            drug_rows = detail_table.find_elements(
                By.XPATH,
                ".//tbody/tr[contains(@class, 'rgRow') or contains(@class, 'rgAltRow')]"
            )
            
            logger.debug(f"Found {len(drug_rows)} drug rows for {atc_code}")
            
            for drug_row in drug_rows:
                try:
                    # Drug name is in column 3 (td[3]) - columns 1 and 2 are hidden
                    # The drug name is in a link with class "productlink"
                    try:
                        drug_name_link = drug_row.find_element(
                            By.XPATH,
                            ".//td[3]//a[contains(@class, 'productlink')]"
                        )
                        drug_name = drug_name_link.text.strip()
                    except:
                        # Fallback: get text from column 3
                        try:
                            drug_name_cell = drug_row.find_element(By.XPATH, ".//td[3]")
                            drug_name = drug_name_cell.text.strip()
                        except:
                            continue
                    
                    if not drug_name or len(drug_name) < 2:
                        continue
                    
                    # Skip if it looks like an ATC code (too short or matches pattern)
                    code_clean = drug_name.replace(" ", "").upper()
                    if len(code_clean) <= 7 and code_clean.isalnum() and any(c.isdigit() for c in code_clean):
                        continue
                    
                    # Form and strength in column 4 (td[4])
                    form_strength = ""
                    try:
                        form_cell = drug_row.find_element(By.XPATH, ".//td[4]")
                        form_strength = form_cell.text.strip()
                    except:
                        pass
                    
                    # Document links in column 5 (td[5])
                    documents = []
                    try:
                        doc_cell = drug_row.find_element(By.XPATH, ".//td[5]")
                        # Look for links in the document cell
                        links = doc_cell.find_elements(By.XPATH, ".//a")
                        for link in links:
                            doc_text = link.text.strip()
                            doc_url = link.get_attribute("href")
                            if doc_text or (doc_url and (".pdf" in doc_url.lower() or "FileRepos" in doc_url)):
                                # Get date if available (in em tag)
                                date_text = ""
                                try:
                                    date_elem = link.find_element(By.XPATH, "./following-sibling::em[1]")
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
                            "atc_code": atc_code,
                            "form_strength": form_strength,
                            "documents": documents
                        }
                
                except Exception as e:
                    logger.debug(f"Error extracting drug from row: {e}")
                    continue
        
        except Exception as e:
            logger.debug(f"Error extracting drugs for {atc_code}: {e}", exc_info=True)
        
        return drugs
    
    def save(self, output_path: Optional[Path] = None) -> None:
        """
        Save scraped data to JSON file.
        
        Args:
            output_path: Optional path to save file (defaults to Config.ATC_INDEX_PATH)
        """
        if output_path is None:
            output_path = Config.ATC_INDEX_PATH
        
        Config.ensure_directories()
        
        data = {
            "hierarchy": self.hierarchy,
            "drug_mappings": self.drug_mappings,
            "scraped_at": time.strftime("%Y-%m-%dT%H:%M:%S")
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved ATC index to {output_path}")


def scrape_atc_index(use_selenium: bool = False) -> Dict[str, Any]:
    """
    Scrape ATC index from Icelandic website.
    
    Args:
        use_selenium: If True, use Selenium for JavaScript rendering
        
    Returns:
        Dictionary with scraped ATC data
    """
    scraper = ATCScraper()
    
    if use_selenium:
        data = scraper.scrape_with_selenium()
    else:
        data = scraper.scrape()
    
    scraper.save()
    return data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    use_selenium = "--selenium" in sys.argv
    
    try:
        data = scrape_atc_index(use_selenium=use_selenium)
        print(f"Successfully scraped ATC index")
        print(f"  - Level 1 categories: {len(data['hierarchy'])}")
        print(f"  - Drug mappings: {len(data['drug_mappings'])}")
    except Exception as e:
        logger.error(f"Failed to scrape ATC index: {e}", exc_info=True)
        sys.exit(1)
