"""Debug script for ATC scraper - test with just category A."""
import logging
import sys
from pathlib import Path

# Add parent directory to path
_script_dir = Path(__file__).parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from src.atc_scraper import ATCScraper

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')

scraper = ATCScraper()

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    import time
    
    chrome_options = Options()
    # Don't use headless for debugging
    # chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(scraper.ATC_LIST_URL)
    
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )
    
    # Click on "A"
    link = driver.find_element(By.LINK_TEXT, "A")
    link.click()
    time.sleep(3)
    
    # Find the result grid
    result_grid = driver.find_element(
        By.XPATH,
        "//table[contains(@id, 'resultGrid') and contains(@class, 'rgMasterTable')]"
    )
    
    # Find all rows
    rows = result_grid.find_elements(By.XPATH, ".//tbody/tr[contains(@class, 'rgRow') or contains(@class, 'rgAltRow')]")
    print(f"\nFound {len(rows)} total rows in table")
    
    for i, row in enumerate(rows[:10]):  # Just first 10
        try:
            code_cell = row.find_element(By.XPATH, ".//td[4]")
            code_text = code_cell.text.strip()
            name_cell = row.find_element(By.XPATH, ".//td[5]")
            name_text = name_cell.text.strip()
            print(f"Row {i}: code='{code_text}' ({len(code_text)} chars), name='{name_text[:50]}'")
        except Exception as e:
            print(f"Row {i}: Error - {e}")
    
    # Now test the recursive extraction
    print("\n--- Testing recursive extraction ---")
    level2_data = scraper._extract_level_recursive(driver, result_grid, "A", level=2)
    print(f"\nExtracted {len(level2_data)} level 2 categories:")
    for code, data in list(level2_data.items())[:5]:
        print(f"  {code}: {data.get('name', '')[:50]}")
        if 'level3' in data:
            print(f"    -> {len(data['level3'])} level 3 categories")
    
    driver.quit()
    
except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()
    if 'driver' in locals():
        driver.quit()
