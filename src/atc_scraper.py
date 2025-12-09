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
            from selenium.common.exceptions import TimeoutException, NoSuchElementException, InvalidSessionIdException
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
            EXPECTED_LEVEL1_SECTIONS = ['A', 'B', 'C', 'D', 'G', 'H', 'J', 'L', 'M', 'N', 'P', 'R', 'S', 'V']
            level1_codes = EXPECTED_LEVEL1_SECTIONS.copy()
            failed_sections: List[str] = []
            
            def reinitialize_driver(chrome_options: Options) -> webdriver.Chrome:
                """Reinitialize Chrome driver after session expiration."""
                logger.info("Reinitializing Chrome driver...")
                if driver:
                    try:
                        driver.quit()
                    except:
                        pass
                new_driver = webdriver.Chrome(options=chrome_options)
                new_driver.get(self.ATC_LIST_URL)
                WebDriverWait(new_driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                logger.info("Driver reinitialized successfully")
                return new_driver
            
            for code in level1_codes:
                logger.info(f"Processing level 1: {code}")
                try:
                    # Find and click the link for this code
                    # The link is in the alphabet list
                    link = driver.find_element(By.LINK_TEXT, code)
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", link)
                    time.sleep(0.3)
                    link.click()
                    # Wait for page load and AJAX - use WebDriverWait for reliability
                    try:
                        WebDriverWait(driver, 10).until(
                            EC.presence_of_element_located(
                                (By.XPATH, "//table[contains(@id, 'resultGrid') and contains(@class, 'rgMasterTable')]")
                            )
                        )
                    except TimeoutException:
                        logger.warning(f"Timeout waiting for result grid after clicking {code}")
                        time.sleep(2)
                    
                    # Extract full hierarchy starting from level 1
                    level1_data = {
                        "code": code,
                        "name": self._get_level_name(code),
                        "level2": {}
                    }
                    
                    # Extract level 2 and below using row ID-based navigation
                    # For level 2, parent_row_id is None (we use main grid)
                    level1_data["level2"] = self._extract_level_by_row_ids(
                        driver, None, code, level=2
                    )
                    
                    if level1_data.get("level2"):
                        self.hierarchy[code] = level1_data
                        logger.info(f"  Extracted {len(level1_data['level2'])} level 2 categories for {code}")
                    else:
                        logger.warning(f"  No level 2 data extracted for {code}")
                    
                    # Go back to main page
                    try:
                        driver.get(self.ATC_LIST_URL)
                        time.sleep(1.5)
                    except InvalidSessionIdException:
                        # Session expired during navigation - reinitialize and continue
                        logger.warning(f"Session expired while navigating after {code}. Reinitializing driver...")
                        try:
                            driver = reinitialize_driver(chrome_options)
                        except Exception as reinit_error:
                            logger.error(f"Failed to reinitialize driver after {code}: {reinit_error}")
                            failed_sections.append(code)
                            continue
                    
                except InvalidSessionIdException as e:
                    # Session expired - check if we already saved this section
                    if code in self.hierarchy:
                        logger.warning(f"Session expired after successfully saving {code}. Reinitializing driver...")
                        try:
                            driver = reinitialize_driver(chrome_options)
                            # Continue to next section since this one was already saved
                            continue
                        except Exception as reinit_error:
                            logger.error(f"Failed to reinitialize driver after {code}: {reinit_error}")
                            failed_sections.append(code)
                            continue
                    else:
                        # Section not saved yet - try to recover and retry
                        logger.error(f"Session expired while processing {code} (not yet saved). Reinitializing and retrying...")
                        try:
                            driver = reinitialize_driver(chrome_options)
                            # Retry this section
                            logger.info(f"Retrying {code} with new session...")
                            # Find and click the link for this code
                            link = driver.find_element(By.LINK_TEXT, code)
                            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", link)
                            time.sleep(0.3)
                            link.click()
                            # Wait for page load
                            try:
                                WebDriverWait(driver, 10).until(
                                    EC.presence_of_element_located(
                                        (By.XPATH, "//table[contains(@id, 'resultGrid') and contains(@class, 'rgMasterTable')]")
                                    )
                                )
                            except TimeoutException:
                                logger.warning(f"Timeout waiting for result grid after clicking {code} (retry)")
                                time.sleep(2)
                            
                            # Extract full hierarchy starting from level 1
                            level1_data = {
                                "code": code,
                                "name": self._get_level_name(code),
                                "level2": {}
                            }
                            
                            # Extract level 2 and below using row ID-based navigation
                            level1_data["level2"] = self._extract_level_by_row_ids(
                                driver, None, code, level=2
                            )
                            
                            if level1_data.get("level2"):
                                self.hierarchy[code] = level1_data
                                logger.info(f"  Extracted {len(level1_data['level2'])} level 2 categories for {code} (retry)")
                            else:
                                logger.warning(f"  No level 2 data extracted for {code} (retry)")
                                failed_sections.append(code)
                            
                            # Go back to main page
                            try:
                                driver.get(self.ATC_LIST_URL)
                                time.sleep(1.5)
                            except InvalidSessionIdException:
                                logger.error(f"Session expired again after retry of {code}")
                                failed_sections.append(code)
                                driver = reinitialize_driver(chrome_options)
                        except Exception as retry_error:
                            logger.error(f"Failed to retry {code} after session expiration: {retry_error}", exc_info=True)
                            failed_sections.append(code)
                            # Try to reinitialize driver for next section
                            try:
                                driver = reinitialize_driver(chrome_options)
                            except:
                                pass
                            continue
                    
                except Exception as e:
                    logger.warning(f"Error processing {code}: {e}", exc_info=True)
                    failed_sections.append(code)
                    # Try to recover by going back to main page
                    try:
                        driver.get(self.ATC_LIST_URL)
                        time.sleep(1)
                    except InvalidSessionIdException:
                        # Session expired during recovery - reinitialize
                        logger.warning(f"Session expired during recovery for {code}. Reinitializing driver...")
                        try:
                            driver = reinitialize_driver(chrome_options)
                        except Exception as reinit_error:
                            logger.error(f"Failed to reinitialize driver during recovery for {code}: {reinit_error}")
                    except:
                        pass
                    continue
            
            # Validate that all expected sections were scraped
            scraped_sections = set(self.hierarchy.keys())
            missing_sections = set(EXPECTED_LEVEL1_SECTIONS) - scraped_sections
            
            if missing_sections:
                error_msg = (
                    f"INCOMPLETE SCRAPING: Only {len(scraped_sections)}/{len(EXPECTED_LEVEL1_SECTIONS)} "
                    f"sections scraped successfully. Missing sections: {sorted(missing_sections)}. "
                    f"Failed sections: {sorted(failed_sections) if failed_sections else 'none'}"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            if failed_sections:
                logger.warning(f"Some sections had errors but were eventually scraped: {sorted(failed_sections)}")
            
            logger.info(f"Successfully scraped all {len(EXPECTED_LEVEL1_SECTIONS)} level 1 categories")
            logger.info(f"Found {len(self.drug_mappings)} drug mappings")
            
            return {
                "hierarchy": self.hierarchy,
                "drug_mappings": self.drug_mappings,
                "scraped_at": time.strftime("%Y-%m-%dT%H:%M:%S")
            }
            
        finally:
            if driver:
                driver.quit()
    
    def _find_row_by_id(
        self,
        driver,
        row_id: str,
        max_retries: int = 3
    ):
        """
        Safely re-find a row by its ID with retry logic.
        
        Args:
            driver: Selenium WebDriver instance
            row_id: The ID attribute of the row to find
            max_retries: Maximum number of retry attempts
            
        Returns:
            WebElement if found, None otherwise
        """
        try:
            from selenium.webdriver.common.by import By
            from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
        except ImportError:
            logger.error("Selenium not available in _find_row_by_id")
            return None
        
        if not row_id:
            return None
        
        for attempt in range(max_retries):
            try:
                row = driver.find_element(By.XPATH, f"//tr[@id='{row_id}']")
                return row
            except NoSuchElementException:
                if attempt < max_retries - 1:
                    time.sleep(0.2 * (attempt + 1))  # Exponential backoff
                    continue
                logger.debug(f"Row with ID '{row_id}' not found after {max_retries} attempts")
                return None
            except StaleElementReferenceException:
                if attempt < max_retries - 1:
                    time.sleep(0.2 * (attempt + 1))
                    continue
                logger.debug(f"Row with ID '{row_id}' became stale during lookup")
                return None
            except Exception as e:
                logger.debug(f"Error finding row with ID '{row_id}': {e}")
                return None
        
        return None
    
    def _expand_row_by_id(
        self,
        driver,
        row_id: str
    ) -> bool:
        """
        Expand a row by its ID if it's collapsed.
        
        Args:
            driver: Selenium WebDriver instance
            row_id: The ID attribute of the row to expand
            
        Returns:
            True if expansion was successful or already expanded, False otherwise
        """
        try:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            from selenium.common.exceptions import NoSuchElementException, TimeoutException
        except ImportError:
            logger.error("Selenium not available in _expand_row_by_id")
            return False
        
        row = self._find_row_by_id(driver, row_id)
        if not row:
            logger.debug(f"Could not find row with ID '{row_id}' for expansion")
            return False
        
        try:
            # Find expand/collapse button
            expand_btn = row.find_element(
                By.XPATH,
                ".//td[1]//input[contains(@class, 'rgExpand') or contains(@class, 'rgCollapse')]"
            )
            
            btn_class = expand_btn.get_attribute("class") or ""
            is_expanded = "rgCollapse" in btn_class
            
            if is_expanded:
                # Already expanded
                logger.debug(f"Row {row_id} is already expanded")
                return True
            
            # Need to expand
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", expand_btn)
            time.sleep(0.3)
            expand_btn.click()
            
            # Wait for detail table to appear
            try:
                WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located(
                        (By.XPATH, f"//tr[@id='{row_id}']/following-sibling::tr[1]//table[contains(@class, 'rgDetailTable')]")
                    )
                )
                logger.debug(f"Successfully expanded row {row_id}")
                return True
            except TimeoutException:
                logger.warning(f"Timeout waiting for detail table after expanding row {row_id}")
                time.sleep(1)  # Give it a bit more time
                return False
            
        except NoSuchElementException:
            # No expand button - row might not have children
            logger.debug(f"Row {row_id} has no expand button (no children)")
            return False
        except Exception as e:
            logger.debug(f"Error expanding row {row_id}: {e}")
            return False
    
    def _collapse_row_by_id(
        self,
        driver,
        row_id: str
    ) -> bool:
        """
        Collapse a row by its ID if it's expanded.
        
        Args:
            driver: Selenium WebDriver instance
            row_id: The ID attribute of the row to collapse
            
        Returns:
            True if collapse was successful or already collapsed, False otherwise
        """
        try:
            from selenium.webdriver.common.by import By
            from selenium.common.exceptions import NoSuchElementException
        except ImportError:
            logger.error("Selenium not available in _collapse_row_by_id")
            return False
        
        row = self._find_row_by_id(driver, row_id)
        if not row:
            logger.debug(f"Could not find row with ID '{row_id}' for collapse")
            return False
        
        try:
            # Find collapse button
            collapse_btn = row.find_element(
                By.XPATH,
                ".//td[1]//input[contains(@class, 'rgCollapse')]"
            )
            
            # Row is expanded, collapse it
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", collapse_btn)
            time.sleep(0.2)
            collapse_btn.click()
            time.sleep(0.3)  # Wait for collapse animation
            logger.debug(f"Successfully collapsed row {row_id}")
            return True
            
        except NoSuchElementException:
            # No collapse button - already collapsed
            logger.debug(f"Row {row_id} is already collapsed")
            return True
        except Exception as e:
            logger.debug(f"Error collapsing row {row_id}: {e}")
            return False
    
    def _collect_child_row_ids(
        self,
        driver,
        parent_row_id: str,
        level: int
    ) -> List[str]:
        """
        Collect all child row IDs from a parent row's detail table.
        
        Args:
            driver: Selenium WebDriver instance
            parent_row_id: The ID of the parent row
            level: Current level (for logging)
            
        Returns:
            List of child row IDs
        """
        try:
            from selenium.webdriver.common.by import By
            from selenium.common.exceptions import NoSuchElementException
        except ImportError:
            logger.error("Selenium not available in _collect_child_row_ids")
            return []
        
        # Expand the parent row if needed
        if not self._expand_row_by_id(driver, parent_row_id):
            logger.debug(f"Could not expand parent row {parent_row_id}, no children")
            return []
        
        # Wait a moment for AJAX
        time.sleep(0.8)
        
        # Find the detail table
        detail_table = None
        try:
            detail_table = driver.find_element(
                By.XPATH,
                f"//tr[@id='{parent_row_id}']/following-sibling::tr[1]//table[contains(@class, 'rgDetailTable')]"
            )
        except NoSuchElementException:
            logger.debug(f"Could not find detail table for parent row {parent_row_id}")
            return []
        
        # Find all child rows in the detail table
        # Important: We need to find DIRECT children only, not rows from nested detail tables
        # Use ./tbody instead of .//tbody to get only direct children, not descendants
        child_row_ids = []
        try:
            # Get only direct child rows (not from nested detail tables)
            # The detail table structure is: <table><tbody><tr>...</tr></tbody></table>
            # We want only the <tr> elements that are direct children of this table's tbody
            all_rows = detail_table.find_elements(
                By.XPATH,
                "./tbody/tr[(contains(@class, 'rgRow') or contains(@class, 'rgAltRow')) and td[4]]"
            )
            
            # Filter to only direct children (rows that don't have nested detail tables as siblings)
            # The structure is: <tr class="rgRow">...</tr> followed by <tr><td><table class="rgDetailTable">...</table></td></tr>
            # We want only the first <tr> in each pair
            for row in all_rows:
                try:
                    # Check if this row has a nested detail table as a following sibling
                    # If it does, it means this row has children and we should include it
                    # But we also need to check if this row itself is a container row (has nested table)
                    # Actually, we want ALL rows with class rgRow or rgAltRow that have td[4]
                    # The nested detail tables are in separate <tr> elements, not in the row itself
                    
                    row_id = row.get_attribute("id")
                    if not row_id:
                        continue
                    
                    # Get the code from td[4] to validate it's a real row
                    try:
                        code_cell = row.find_element(By.XPATH, ".//td[4]")
                        code_text = code_cell.text.strip()
                        if not code_text:
                            continue
                    except:
                        continue
                    
                    child_row_ids.append(row_id)
                    
                except Exception as e:
                    logger.debug(f"Error processing row in collection: {e}")
                    continue
            
            logger.info(f"Collected {len(child_row_ids)} child row IDs from parent {parent_row_id} at level {level}")
            if len(child_row_ids) > 0:
                # Log the codes we found to help debug
                codes_found = []
                for rid in child_row_ids[:10]:  # Check first 10
                    try:
                        r = self._find_row_by_id(driver, rid)
                        if r:
                            code_cell = r.find_element(By.XPATH, ".//td[4]")
                            code_text = code_cell.text.strip()
                            if code_text:
                                codes_found.append(code_text.replace(" ", "").upper())
                    except:
                        pass
                logger.info(f"Codes found in first {min(10, len(child_row_ids))} rows: {codes_found}")
            
        except Exception as e:
            logger.warning(f"Error collecting child row IDs: {e}", exc_info=True)
        
        return child_row_ids
    
    def _is_drug_detail_table(self, driver, parent_row_id: str) -> bool:
        """
        Check if a detail table contains drugs (not ATC codes) by examining headers.
        
        Args:
            driver: Selenium WebDriver instance
            parent_row_id: ID of the parent row
            
        Returns:
            True if detail table contains drugs, False if it contains ATC codes
        """
        try:
            from selenium.webdriver.common.by import By
            from selenium.common.exceptions import NoSuchElementException
        except ImportError:
            return False
        
        try:
            # Find the detail table
            detail_table = driver.find_element(
                By.XPATH,
                f"//tr[@id='{parent_row_id}']/following-sibling::tr[1]//table[contains(@class, 'rgDetailTable')]"
            )
            
            # Check the table headers
            # Drug tables have "Lyfjaheiti" header, ATC code tables have "ATC Flokkur" header
            try:
                # Look for "Lyfjaheiti" header (drug table)
                detail_table.find_element(By.XPATH, ".//th[contains(text(), 'Lyfjaheiti')]")
                return True
            except NoSuchElementException:
                # Look for "ATC Flokkur" header (ATC code table)
                try:
                    detail_table.find_element(By.XPATH, ".//th[contains(text(), 'ATC Flokkur')]")
                    return False
                except NoSuchElementException:
                    # Can't determine - default to ATC codes (safer)
                    return False
        except:
            return False
    
    def _extract_level_by_row_ids(
        self,
        driver,
        parent_row_id: Optional[str],
        parent_code: str,
        level: int = 2,
        max_level: int = 5
    ) -> Dict[str, Any]:
        """
        Recursively extract ATC hierarchy using row ID-based navigation.
        
        This method never maintains references to DOM elements, always re-finding
        by row ID to avoid stale element issues.
        
        Args:
            driver: Selenium WebDriver instance
            parent_row_id: ID of the parent row (None for level 2, which uses main grid)
            parent_code: ATC code of the parent level (e.g., "A" for level 1)
            level: Current level (2-5)
            max_level: Maximum level to extract (5 for full ATC)
            
        Returns:
            Dictionary mapping ATC codes to their data
        """
        try:
            from selenium.webdriver.common.by import By
            from selenium.common.exceptions import NoSuchElementException
        except ImportError:
            logger.error("Selenium not available in _extract_level_by_row_ids")
            return {}
        
        result = {}
        
        if level > max_level:
            return result
        
        try:
            # Track processed row IDs to avoid duplicates
            processed_row_ids = set()
            
            # For level 2, we need to get row IDs from the main grid
            # For deeper levels, we get child row IDs from the parent row
            if level == 2:
                # Level 2: Get row IDs from main result grid
                try:
                    result_grid = driver.find_element(
                        By.XPATH,
                        "//table[contains(@id, 'resultGrid') and contains(@class, 'rgMasterTable')]"
                    )
                    rows = result_grid.find_elements(
                        By.XPATH,
                        ".//tbody/tr[(contains(@class, 'rgRow') or contains(@class, 'rgAltRow')) and td[4]]"
                    )
                    row_ids = []
                    for row in rows:
                        try:
                            row_id = row.get_attribute("id")
                            if row_id:
                                row_ids.append(row_id)
                        except:
                            continue
                    logger.info(f"Level {level}: Found {len(row_ids)} rows in main grid for {parent_code}")
                except Exception as e:
                    logger.warning(f"Could not find main result grid for level {level}: {e}")
                    return result
            else:
                # Deeper levels: Get child row IDs from parent row
                if not parent_row_id:
                    logger.warning(f"No parent_row_id provided for level {level}")
                    return result
                
                row_ids = self._collect_child_row_ids(driver, parent_row_id, level)
                if not row_ids:
                    logger.debug(f"Level {level}: No child rows found for parent {parent_code} (row_id: {parent_row_id})")
                    return result
                logger.debug(f"Level {level}: Found {len(row_ids)} child rows for {parent_code}")
            
            # Process each row ID independently
            for row_id in row_ids:
                if row_id in processed_row_ids:
                    continue
                
                processed_row_ids.add(row_id)
                
                try:
                    # Re-find row by ID
                    row = self._find_row_by_id(driver, row_id)
                    if not row:
                        logger.debug(f"Could not find row with ID '{row_id}', skipping")
                        continue
                    
                    # Extract code and name from row
                    try:
                        code_cell = row.find_element(By.XPATH, ".//td[4]")
                        code_text = code_cell.text.strip()
                        if not code_text:
                            continue
                        
                        code_clean = code_text.replace(" ", "").upper()
                        
                        # Validate code length matches expected level
                        expected_lengths = {2: 3, 3: 4, 4: 5, 5: 7}
                        if level in expected_lengths and len(code_clean) != expected_lengths[level]:
                            logger.debug(f"Row {row_id} code {code_clean} length {len(code_clean)} doesn't match expected {expected_lengths[level]} for level {level}, skipping")
                            continue
                        
                        # Validate code starts with parent_code
                        if not code_clean.startswith(parent_code):
                            logger.warning(f"Row {row_id} code {code_clean} does not start with parent {parent_code}, skipping")
                            continue
                        
                        # Get name from 5th column
                        try:
                            name_cell = row.find_element(By.XPATH, ".//td[5]")
                            name_text = name_cell.text.strip()
                        except:
                            name_text = ""
                        
                        level_data = {
                            "code": code_clean,
                            "name": name_text
                        }
                        
                        if level <= 3:
                            logger.info(f"  Processing {code_clean} at level {level} (row_id: {row_id})")
                        else:
                            logger.debug(f"  Processing {code_clean} at level {level} (row_id: {row_id})")
                        
                        # Check if this row has children
                        has_child = False
                        try:
                            has_child_cell = row.find_element(By.XPATH, ".//td[3]")
                            has_child_value = has_child_cell.get_attribute("textContent") or ""
                            has_child_str = has_child_value.strip()
                            
                            try:
                                child_count = int(has_child_str)
                                has_child = child_count > 0
                            except ValueError:
                                has_child = has_child_str == "1"
                        except:
                            # Try to find expand button as indicator
                            try:
                                row.find_element(
                                    By.XPATH,
                                    ".//td[1]//input[contains(@class, 'rgExpand') or contains(@class, 'rgCollapse')]"
                                )
                                has_child = True
                            except:
                                pass
                        
                        # Process based on level
                        if level == max_level:
                            # Level 5: Extract drugs
                            drugs = self._extract_drugs_from_level5(driver, row_id, code_clean)
                            if drugs:
                                level_data["drugs"] = drugs
                                for drug_name in drugs.keys():
                                    if drug_name not in self.drug_mappings:
                                        self.drug_mappings[drug_name] = []
                                    if code_clean not in self.drug_mappings[drug_name]:
                                        self.drug_mappings[drug_name].append(code_clean)
                        elif has_child:
                            # Check if this is level 4 with drug rows directly (irregular pattern)
                            # vs level 4 with level 5 ATC code children (normal pattern)
                            if level == 4:
                                # Check if detail table contains drugs or ATC codes
                                is_drug_table = self._is_drug_detail_table(driver, row_id)
                                
                                if is_drug_table:
                                    # Case 2: Level 4 expands directly to drugs (A02AH pattern)
                                    logger.info(f"  {code_clean} has drug table directly, extracting drugs")
                                    drugs = self._extract_drugs_from_level5(driver, row_id, code_clean)
                                    if drugs:
                                        level_data["drugs"] = drugs
                                        logger.info(f"  Extracted {len(drugs)} drug(s) for {code_clean}")
                                        for drug_name in drugs.keys():
                                            if drug_name not in self.drug_mappings:
                                                self.drug_mappings[drug_name] = []
                                            if code_clean not in self.drug_mappings[drug_name]:
                                                self.drug_mappings[drug_name].append(code_clean)
                                    else:
                                        logger.debug(f"  No drugs found for {code_clean}")
                                else:
                                    # Case 1: Level 4 has level 5 ATC code children (normal pattern)
                                    children = self._extract_level_by_row_ids(
                                        driver, row_id, code_clean, level + 1, max_level
                                    )
                                    if children:
                                        level_key = f"level{level + 1}"
                                        level_data[level_key] = children
                                        logger.debug(f"  Extracted {len(children)} children for {code_clean}")
                            else:
                                # Levels 2-3: Always have ATC code children
                                children = self._extract_level_by_row_ids(
                                    driver, row_id, code_clean, level + 1, max_level
                                )
                                if children:
                                    level_key = f"level{level + 1}"
                                    level_data[level_key] = children
                                    logger.debug(f"  Extracted {len(children)} children for {code_clean}")
                            
                            # DON'T collapse here - we need to keep the row expanded so we can
                            # continue processing its siblings. We'll collapse the parent row
                            # after all children are processed (see below)
                        
                        result[code_clean] = level_data
                        
                    except Exception as e:
                        logger.debug(f"Error processing row {row_id}: {e}", exc_info=True)
                        continue
                        
                except Exception as e:
                    logger.debug(f"Error processing row ID {row_id}: {e}", exc_info=True)
                    continue
            
            # After processing all children, collapse the parent row if we expanded it
            # This only applies to deeper levels (not level 2, which uses main grid)
            # _collect_child_row_ids expanded the parent row to get child row_ids
            if level > 2 and parent_row_id:
                self._collapse_row_by_id(driver, parent_row_id)
                logger.debug(f"Collapsed parent row {parent_row_id} after processing all children")
            
            return result
            
        except Exception as e:
            logger.warning(f"Error extracting level {level} for {parent_code}: {e}", exc_info=True)
            return result
    
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
            from selenium.common.exceptions import NoSuchElementException, TimeoutException, StaleElementReferenceException
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
            
            # Process rows using depth-first traversal
            # After each branch is processed, re-find all remaining rows to handle DOM changes
            processed_codes = set()
            
            while True:
                # Re-find parent element if it becomes stale (can happen at any level)
                # At level 2, we need to find the main result grid
                # At deeper levels, we need to find the detail table from the parent row
                try:
                    # Try to use parent_element - if it's stale, this will fail
                    _ = parent_element.tag_name
                except StaleElementReferenceException:
                    # Parent element is stale, need to re-find it
                    if level == 2:
                        # Re-find the main result grid
                        try:
                            parent_element = driver.find_element(
                                By.XPATH,
                                "//table[contains(@id, 'resultGrid') and contains(@class, 'rgMasterTable')]"
                            )
                        except:
                            try:
                                parent_element = driver.find_element(
                                    By.XPATH,
                                    "//table[@class='rgMasterTable']"
                                )
                            except:
                                logger.warning(f"Could not re-find parent table at level {level}")
                                break
                    else:
                        # For deeper levels, we need to find the detail table again
                        # This is trickier - we'll try multiple strategies to re-find it
                        parent_found = False
                        
                        # Strategy 1: Find detail tables and check which one has our parent_code
                        try:
                            detail_tables = driver.find_elements(
                                By.XPATH,
                                "//table[contains(@class, 'rgDetailTable')]"
                            )
                            for dt in detail_tables:
                                try:
                                    # Check if this table has a row with parent_code
                                    test_row = dt.find_element(
                                        By.XPATH,
                                        f".//tbody/tr[td[4]='{parent_code}']"
                                    )
                                    parent_element = dt
                                    parent_found = True
                                    logger.debug(f"Re-found parent detail table for {parent_code} using Strategy 1")
                                    break
                                except:
                                    continue
                        except Exception as e:
                            logger.debug(f"Strategy 1 failed for {parent_code}: {e}")
                        
                        # Strategy 2: If parent_code has a parent (e.g., B06AC -> B06A), find parent's detail table
                        # and then find the detail table within it
                        if not parent_found and len(parent_code) > 2:
                            try:
                                # Get parent of parent_code (e.g., B06AC -> B06A, B06A -> B06)
                                parent_of_parent = parent_code[:-1] if len(parent_code) > 3 else parent_code[:-1]
                                
                                # Find detail table containing parent_of_parent
                                parent_tables = driver.find_elements(
                                    By.XPATH,
                                    f"//table[contains(@class, 'rgDetailTable')]//tbody/tr[td[4]='{parent_of_parent}']"
                                )
                                if parent_tables:
                                    # Get the table containing this row
                                    parent_row = parent_tables[0]
                                    parent_table = parent_row.find_element(By.XPATH, "./ancestor::table[contains(@class, 'rgDetailTable')][1]")
                                    
                                    # Now find the detail table for parent_code within this parent table
                                    try:
                                        detail_table_row = parent_table.find_element(
                                            By.XPATH,
                                            f".//tbody/tr[td[4]='{parent_code}']"
                                        )
                                        # Find the detail table that follows this row
                                        detail_table_tr = detail_table_row.find_element(By.XPATH, "./following-sibling::tr[1]")
                                        parent_element = detail_table_tr.find_element(
                                            By.XPATH,
                                            ".//table[contains(@class, 'rgDetailTable')]"
                                        )
                                        parent_found = True
                                        logger.debug(f"Re-found parent detail table for {parent_code} using Strategy 2")
                                    except:
                                        pass
                            except Exception as e:
                                logger.debug(f"Strategy 2 failed for {parent_code}: {e}")
                        
                        # Strategy 3: Try to find by navigating from the main result grid
                        if not parent_found:
                            try:
                                # Start from main grid and navigate down
                                main_grid = driver.find_element(
                                    By.XPATH,
                                    "//table[contains(@id, 'resultGrid') and contains(@class, 'rgMasterTable')]"
                                )
                                # Try to find a path to parent_code's detail table
                                # This is a fallback - navigate through expanded rows
                                all_detail_tables = driver.find_elements(
                                    By.XPATH,
                                    "//table[contains(@class, 'rgDetailTable')]"
                                )
                                # Check each table to see if it's the right one by checking if it has rows
                                # that would be children of parent_code
                                for dt in all_detail_tables:
                                    try:
                                        # Check if this table appears to be at the right level
                                        # by checking if it has rows with codes that would be children
                                        rows_in_table = dt.find_elements(By.XPATH, ".//tbody/tr[td[4]]")
                                        if rows_in_table:
                                            # Check first row to see if code length matches expected level
                                            first_code_cell = rows_in_table[0].find_element(By.XPATH, ".//td[4]")
                                            first_code = first_code_cell.text.strip().replace(" ", "").upper()
                                            expected_length = {2: 3, 3: 4, 4: 5, 5: 7}.get(level, 0)
                                            if expected_length > 0 and len(first_code) == expected_length:
                                                # This might be our table - verify by checking if parent_code
                                                # appears in the ancestor chain
                                                try:
                                                    # Check if we can find parent_code in an ancestor row
                                                    ancestor_row = dt.find_element(
                                                        By.XPATH,
                                                        f"./ancestor::tr[td[4]='{parent_code}']"
                                                    )
                                                    parent_element = dt
                                                    parent_found = True
                                                    logger.debug(f"Re-found parent detail table for {parent_code} using Strategy 3")
                                                    break
                                                except:
                                                    pass
                                    except:
                                        continue
                            except Exception as e:
                                logger.debug(f"Strategy 3 failed for {parent_code}: {e}")
                        
                        if not parent_found:
                            logger.warning(f"Could not re-find parent detail table at level {level} for {parent_code}")
                            # Try to recover by finding rows through parent's parent
                            # Find the parent's parent row, then find its detail table, then find parent_code's detail table
                            if len(parent_code) > 2:
                                try:
                                    # Get parent of parent_code (e.g., A02A -> A02)
                                    parent_of_parent = parent_code[:-1] if len(parent_code) > 3 else parent_code[:-1]
                                    
                                    # Find the row with parent_of_parent code
                                    parent_row = driver.find_element(
                                        By.XPATH,
                                        f"//tr[td[4]='{parent_of_parent}']"
                                    )
                                    
                                    # Find the detail table that follows this row
                                    parent_detail_tr = parent_row.find_element(By.XPATH, "./following-sibling::tr[1]")
                                    parent_detail_table = parent_detail_tr.find_element(
                                        By.XPATH,
                                        ".//table[contains(@class, 'rgDetailTable')]"
                                    )
                                    
                                    # Now find the row with parent_code in this table
                                    parent_code_row = parent_detail_table.find_element(
                                        By.XPATH,
                                        f".//tbody/tr[td[4]='{parent_code}']"
                                    )
                                    
                                    # Find the detail table that follows parent_code row
                                    detail_tr = parent_code_row.find_element(By.XPATH, "./following-sibling::tr[1]")
                                    parent_element = detail_tr.find_element(
                                        By.XPATH,
                                        ".//table[contains(@class, 'rgDetailTable')]"
                                    )
                                    
                                    parent_found = True
                                    logger.debug(f"Re-found parent detail table for {parent_code} via parent's parent")
                                except Exception as e:
                                    logger.debug(f"Could not recover via parent's parent for {parent_code}: {e}")
                            
                            # If still not found, try to continue by finding rows directly
                            if not parent_found:
                                # Last resort: try to find any detail table that contains unprocessed rows
                                # This is less precise but better than breaking
                                try:
                                    # Find all detail tables
                                    all_detail_tables = driver.find_elements(
                                        By.XPATH,
                                        "//table[contains(@class, 'rgDetailTable')]"
                                    )
                                    
                                    # Find one that has rows matching our level pattern
                                    expected_length = {2: 3, 3: 4, 4: 5, 5: 7}.get(level, 0)
                                    for dt in all_detail_tables:
                                        try:
                                            rows_in_table = dt.find_elements(By.XPATH, ".//tbody/tr[td[4]]")
                                            if rows_in_table:
                                                # Check if first row matches expected level
                                                first_code_cell = rows_in_table[0].find_element(By.XPATH, ".//td[4]")
                                                first_code = first_code_cell.text.strip().replace(" ", "").upper()
                                                if expected_length > 0 and len(first_code) == expected_length:
                                                    # Check if this table is a descendant of parent_code's hierarchy
                                                    # by checking if we can find parent_code in ancestor
                                                    try:
                                                        dt.find_element(
                                                            By.XPATH,
                                                            f"./ancestor::tr[td[4]='{parent_code}']"
                                                        )
                                                        parent_element = dt
                                                        parent_found = True
                                                        logger.debug(f"Re-found parent detail table for {parent_code} via fallback search")
                                                        break
                                                    except:
                                                        pass
                                        except:
                                            continue
                                except Exception as e:
                                    logger.debug(f"Fallback search failed for {parent_code}: {e}")
                            
                            # If we still can't find it, log warning but try to continue
                            # by re-finding from the main grid (only works for level 2)
                            if not parent_found:
                                if level == 2:
                                    try:
                                        parent_element = driver.find_element(
                                            By.XPATH,
                                            "//table[contains(@id, 'resultGrid') and contains(@class, 'rgMasterTable')]"
                                        )
                                        logger.debug(f"Re-initialized parent_element from main grid for level {level}")
                                    except:
                                        logger.warning(f"Could not recover parent element at level {level}, will try to continue")
                                        # Don't break - let the row finding logic try to work around it
                                else:
                                    logger.warning(f"Could not recover parent element at level {level} for {parent_code}, will try to continue")
                                    # Don't break - continue and let row finding try alternative methods
                
                # Re-find all rows at current level (DOM may have changed)
                try:
                    rows = parent_element.find_elements(By.XPATH, row_xpath)
                except StaleElementReferenceException:
                    # Parent became stale during iteration, re-find and retry
                    logger.debug(f"Parent element became stale at level {level}, re-finding...")
                    if level == 2:
                        try:
                            parent_element = driver.find_element(
                                By.XPATH,
                                "//table[contains(@id, 'resultGrid') and contains(@class, 'rgMasterTable')]"
                            )
                            rows = parent_element.find_elements(By.XPATH, row_xpath)
                        except:
                            logger.warning(f"Could not recover from stale element at level {level}, trying alternative method")
                            # Try to find rows by searching for all rows at this level, filtered by parent_code
                            try:
                                expected_length = {2: 3, 3: 4, 4: 5, 5: 7}.get(level, 0)
                                if expected_length > 0:
                                    # Find all rows with ATC codes of expected length
                                    all_rows = driver.find_elements(
                                        By.XPATH,
                                        f"//tbody/tr[td[4] and string-length(translate(td[4], ' ', ''))={expected_length}]"
                                    )
                                    # Filter to only rows that start with parent_code
                                    rows = []
                                    for r in all_rows:
                                        try:
                                            code_cell = r.find_element(By.XPATH, ".//td[4]")
                                            code_text = code_cell.text.strip().replace(" ", "").upper()
                                            if code_text.startswith(parent_code):
                                                rows.append(r)
                                        except:
                                            continue
                                    logger.debug(f"Found {len(rows)} rows using alternative method at level {level} (filtered by parent_code {parent_code})")
                                else:
                                    rows = []
                            except:
                                logger.warning(f"Could not find rows at level {level}")
                                rows = []
                    else:
                        # For deeper levels, try to re-find parent using strategies from above
                        logger.debug(f"Stale element at level {level}, attempting recovery")
                        # The parent re-finding logic above should have handled this, but if we get here,
                        # try one more time to find rows by searching for detail tables
                        try:
                            # Find detail table containing parent_code
                            parent_code_row = driver.find_element(
                                By.XPATH,
                                f"//tr[td[4]='{parent_code}']"
                            )
                            detail_tr = parent_code_row.find_element(By.XPATH, "./following-sibling::tr[1]")
                            parent_element = detail_tr.find_element(
                                By.XPATH,
                                ".//table[contains(@class, 'rgDetailTable')]"
                            )
                            rows = parent_element.find_elements(By.XPATH, row_xpath)
                            logger.debug(f"Recovered parent element and found {len(rows)} rows")
                        except:
                            logger.debug(f"Could not recover at level {level}, will try to continue")
                            # Try to find rows by pattern matching, but filter by parent_code prefix
                            try:
                                expected_length = {2: 3, 3: 4, 4: 5, 5: 7}.get(level, 0)
                                if expected_length > 0:
                                    # Find all rows at this level, then filter by parent_code
                                    all_rows = driver.find_elements(
                                        By.XPATH,
                                        f"//table[contains(@class, 'rgDetailTable')]//tbody/tr[td[4] and string-length(translate(td[4], ' ', ''))={expected_length}]"
                                    )
                                    # Filter to only rows that start with parent_code
                                    rows = []
                                    for r in all_rows:
                                        try:
                                            code_cell = r.find_element(By.XPATH, ".//td[4]")
                                            code_text = code_cell.text.strip().replace(" ", "").upper()
                                            if code_text.startswith(parent_code):
                                                rows.append(r)
                                        except:
                                            continue
                                    logger.debug(f"Found {len(rows)} rows using pattern matching at level {level} (filtered by parent_code {parent_code})")
                                else:
                                    rows = []
                            except:
                                rows = []
                
                # Find next unprocessed row
                next_row = None
                next_code = None
                
                for row in rows:
                    try:
                        # Skip rows with nested tables (already expanded)
                        try:
                            row.find_element(By.XPATH, ".//table[contains(@class, 'rgDetailTable')]")
                            continue
                        except (NoSuchElementException, StaleElementReferenceException):
                            pass
                        
                        code_cell = row.find_element(By.XPATH, ".//td[4]")
                        code_text = code_cell.text.strip()
                        if not code_text:
                            continue
                        
                        code_clean = code_text.replace(" ", "").upper()
                        expected_lengths = {2: 3, 3: 4, 4: 5, 5: 7}
                        if level in expected_lengths and len(code_clean) != expected_lengths[level]:
                            continue
                        
                        if code_clean not in processed_codes:
                            next_row = row
                            next_code = code_clean
                            break
                    except StaleElementReferenceException:
                        # Row became stale, skip it and continue
                        continue
                    except:
                        continue
                
                if not next_row:
                    # No more rows to process
                    break
                
                # Process this row
                code_clean = next_code
                row = next_row
                processed_codes.add(code_clean)
                try:
                    # Re-find the row by code (DOM may have changed from previous expansions)
                    try:
                        row = parent_element.find_element(
                            By.XPATH,
                            f".//tbody/tr[td[4]='{code_clean}']"
                        )
                    except (StaleElementReferenceException, AttributeError):
                        # Parent became stale or is None, try to re-find row directly
                        try:
                            # Find row directly by code, but ensure it starts with parent_code
                            if code_clean.startswith(parent_code):
                                row = driver.find_element(
                                    By.XPATH,
                                    f"//tr[td[4]='{code_clean}']"
                                )
                                # Try to update parent_element if we can find it
                                try:
                                    if level == 2:
                                        parent_element = driver.find_element(
                                            By.XPATH,
                                            "//table[contains(@id, 'resultGrid') and contains(@class, 'rgMasterTable')]"
                                        )
                                    else:
                                        # Find the detail table that contains this row
                                        # The row is inside a detail table, so find the ancestor table
                                        parent_element = row.find_element(
                                            By.XPATH,
                                            "./ancestor::table[contains(@class, 'rgDetailTable')][1]"
                                        )
                                except:
                                    pass  # Keep using row even if we can't update parent_element
                            else:
                                logger.debug(f"Row {code_clean} does not belong to parent {parent_code}, skipping")
                                continue
                        except:
                            logger.debug(f"Could not re-find row for {code_clean}, skipping")
                            continue
                    except:
                        logger.debug(f"Could not find row for {code_clean}, skipping")
                        continue
                    
                    # Check if this row has a nested detail table (already expanded)
                    # If so, skip it as it's a container row
                    try:
                        row.find_element(By.XPATH, ".//table[contains(@class, 'rgDetailTable')]")
                        logger.debug(f"  {code_clean}: Skipping (contains nested table)")
                        continue
                    except NoSuchElementException:
                        pass
                    
                    # Get row ID
                    row_id = row.get_attribute("id")
                    
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
                    
                    # Get current row ID for later use
                    current_row_id = row.get_attribute("id")
                    
                    if level <= 3:
                        logger.info(f"  Processing {code_clean} at level {level} (depth-first)")
                    else:
                        logger.debug(f"  Processing {code_clean} at level {level} (depth-first)")
                    
                    # Process this row immediately (depth-first approach)
                    # Check if this row has children
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
                            has_child = has_child_str == "1"
                    except:
                        # Try to find expand button as indicator
                        try:
                            expand_btn = row.find_element(
                                By.XPATH, 
                                ".//td[1]//input[contains(@class, 'rgExpand') or contains(@class, 'rgCollapse')]"
                            )
                            has_child = True
                        except:
                            pass
                    
                    # If level 5, extract drugs
                    if level == max_level:
                        drugs = self._extract_drugs_from_level5(driver, row, code_clean)
                        if drugs:
                            level_data["drugs"] = drugs
                            for drug_name in drugs.keys():
                                if drug_name not in self.drug_mappings:
                                    self.drug_mappings[drug_name] = []
                                if code_clean not in self.drug_mappings[drug_name]:
                                    self.drug_mappings[drug_name].append(code_clean)
                    elif has_child:
                        # Expand → Process → Collapse pattern
                        # Remember entry point (row_id) before expanding
                        entry_point_row_id = current_row_id
                        was_expanded_before = False
                        
                        try:
                            expand_btn = row.find_element(
                                By.XPATH,
                                ".//td[1]//input[contains(@class, 'rgExpand') or contains(@class, 'rgCollapse')]"
                            )
                            
                            btn_class = expand_btn.get_attribute("class") or ""
                            is_expanded = "rgCollapse" in btn_class
                            was_expanded_before = is_expanded
                            
                            if not is_expanded:
                                # Expand the row
                                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", expand_btn)
                                time.sleep(0.3)
                                expand_btn.click()
                                try:
                                    if entry_point_row_id:
                                        WebDriverWait(driver, 5).until(
                                            EC.presence_of_element_located(
                                                (By.XPATH, f"//tr[@id='{entry_point_row_id}']/following-sibling::tr[1]//table[contains(@class, 'rgDetailTable')]")
                                            )
                                        )
                                    else:
                                        time.sleep(2)
                                except TimeoutException:
                                    time.sleep(2)
                                except:
                                    time.sleep(2)
                            
                            # Find detail table
                            time.sleep(0.8)
                            detail_table = None
                            
                            if entry_point_row_id:
                                try:
                                    detail_table = driver.find_element(
                                        By.XPATH,
                                        f"//tr[@id='{entry_point_row_id}']/following-sibling::tr[1]//table[contains(@class, 'rgDetailTable')]"
                                    )
                                except:
                                    pass
                            
                            if not detail_table and entry_point_row_id:
                                try:
                                    detail_table = parent_element.find_element(
                                        By.XPATH,
                                        f".//tr[preceding-sibling::tr[@id='{entry_point_row_id}']][1]//table[contains(@class, 'rgDetailTable')]"
                                    )
                                except:
                                    pass
                            
                            if detail_table:
                                # Recursively process children (depth-first - fully process this branch before returning)
                                children = self._extract_level_recursive(
                                    driver, detail_table, code_clean, level + 1, max_level
                                )
                                
                                if children:
                                    level_key = f"level{level + 1}"
                                    level_data[level_key] = children
                                    logger.debug(f"  Extracted {len(children)} children for {code_clean}")
                            
                            # After processing children, collapse the row if we expanded it
                            # Re-find the row by ID to avoid stale element issues
                            if entry_point_row_id and not was_expanded_before:
                                try:
                                    # Re-find the row by ID
                                    entry_row = driver.find_element(
                                        By.XPATH,
                                        f"//tr[@id='{entry_point_row_id}']"
                                    )
                                    
                                    # Check if it's still expanded
                                    try:
                                        collapse_btn = entry_row.find_element(
                                            By.XPATH,
                                            ".//td[1]//input[contains(@class, 'rgCollapse')]"
                                        )
                                        # Row is expanded, collapse it
                                        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", collapse_btn)
                                        time.sleep(0.2)
                                        collapse_btn.click()
                                        # Wait for collapse animation
                                        time.sleep(0.3)
                                        logger.debug(f"  Collapsed {code_clean} after processing")
                                    except NoSuchElementException:
                                        # Already collapsed or no collapse button
                                        pass
                                except Exception as e:
                                    # Row might be stale or not found - that's okay, continue
                                    logger.debug(f"  Could not collapse {code_clean} (row may be stale): {e}")
                        
                        except NoSuchElementException:
                            pass
                        except Exception as e:
                            logger.debug(f"Could not expand {code_clean} at level {level}: {e}")
                    
                    result[code_clean] = level_data
                
                except Exception as e:
                    logger.debug(f"Error processing {code_clean} at level {level}: {e}", exc_info=True)
                    continue
            
            return result
            
        except Exception as e:
            logger.warning(f"Error extracting level {level} for {parent_code}: {e}", exc_info=True)
            return result
    
    def _extract_drugs_from_level5(
        self,
        driver,
        row_id: str,
        atc_code: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract drug information from a level 5 ATC row.
        
        Args:
            driver: Selenium WebDriver instance
            row_id: The ID of the table row for level 5
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
            # Re-find row by ID
            row = self._find_row_by_id(driver, row_id)
            if not row:
                logger.debug(f"Could not find row with ID '{row_id}' for drug extraction")
                return {}
            
            # First, ensure the level 5 row is expanded
            if not self._expand_row_by_id(driver, row_id):
                logger.debug(f"Could not expand row {row_id} for drug extraction")
                return {}
            
            # Find the nested Detail table that contains drug information
            # The detail table appears in the following sibling <tr> after expansion
            # Wait a moment for AJAX to complete
            time.sleep(0.8)
            
            detail_table = None
            
            # Find following sibling tr with Detail table using row ID (most reliable)
            try:
                detail_table = driver.find_element(
                    By.XPATH,
                    f"//tr[@id='{row_id}']/following-sibling::tr[1]//table[contains(@class, 'rgDetailTable')]"
                )
                logger.debug(f"  Found drug detail table for {atc_code} using row ID")
            except NoSuchElementException:
                # If we can't find the detail table, return empty rather than guessing
                # This typically means the ATC code has no associated drugs (empty category)
                logger.debug(f"  Could not find drug detail table for {atc_code} - empty category")
                return drugs
            except Exception as e:
                logger.debug(f"  Error finding drug detail table for {atc_code}: {e}")
                return drugs
            
            if not detail_table:
                logger.debug(f"  No drug detail table found for {atc_code}")
                return drugs
            
            # Find all drug rows in the detail table
            # Drug rows have class rgRow or rgAltRow and are in tbody
            drug_rows = detail_table.find_elements(
                By.XPATH,
                ".//tbody/tr[contains(@class, 'rgRow') or contains(@class, 'rgAltRow')]"
            )
            
            logger.info(f"Found {len(drug_rows)} drug row(s) for {atc_code}")
            
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
                        # Store formulations as array to handle multiple formulations of same drug
                        if drug_name in drugs:
                            # Append to existing formulations array
                            drugs[drug_name]["formulations"].append({
                                "form_strength": form_strength,
                                "documents": documents
                            })
                        else:
                            # Create new entry with formulations array
                            drugs[drug_name] = {
                                "atc_code": atc_code,
                                "formulations": [{
                                    "form_strength": form_strength,
                                    "documents": documents
                                }]
                            }
                
                except Exception as e:
                    logger.debug(f"Error extracting drug from row: {e}")
                    continue
            
            # Log successful extraction
            if drugs:
                logger.info(f"Extracted {len(drugs)} drug(s) for {atc_code}")
        
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
