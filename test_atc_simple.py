"""Simple test - just scrape category A to see if it works."""
import logging
import sys
from pathlib import Path

_script_dir = Path(__file__).parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from src.atc_scraper import ATCScraper

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

scraper = ATCScraper()

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    import time
    
    chrome_options = Options()
    chrome_options.add_argument('--headless')
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
    
    # Find result grid
    result_grid = driver.find_element(
        By.XPATH,
        "//table[contains(@id, 'resultGrid') and contains(@class, 'rgMasterTable')]"
    )
    
    # Extract just category A
    level1_data = {
        "code": "A",
        "name": scraper._get_level_name("A"),
        "level2": {}
    }
    
    level1_data["level2"] = scraper._extract_level_recursive(driver, result_grid, "A", level=2)
    
    print(f"\nExtracted {len(level1_data['level2'])} level 2 categories")
    
    # Count total drugs
    total_drugs = sum(
        len(data.get('drugs', {})) 
        for level2_data in level1_data['level2'].values()
        for level3_data in level2_data.get('level3', {}).values()
        for level4_data in level3_data.get('level4', {}).values()
        for level5_data in level4_data.get('level5', {}).values()
        for data in [level5_data]
    )
    
    print(f"Total drugs found: {total_drugs}")
    print(f"Drug mappings: {len(scraper.drug_mappings)}")
    
    # Show a sample
    if scraper.drug_mappings:
        sample_drug = list(scraper.drug_mappings.keys())[0]
        print(f"\nSample drug: {sample_drug} -> {scraper.drug_mappings[sample_drug]}")
    
    driver.quit()
    
except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()
    if 'driver' in locals():
        driver.quit()
