"""Simple test - just scrape letter I to see if it works (for Ibuprofen)."""
import logging
import sys
from pathlib import Path

_script_dir = Path(__file__).parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from src.ingredients_scraper import IngredientsScraper

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

scraper = IngredientsScraper()

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
    driver.get(scraper.INGREDIENTS_URL)
    
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )
    
    # Click on "I"
    link = driver.find_element(By.LINK_TEXT, "I")
    link.click()
    time.sleep(3)
    
    # Find result grid
    result_grid = driver.find_element(
        By.XPATH,
        "//table[contains(@id, 'resultGrid') and contains(@class, 'rgMasterTable')]"
    )
    
    # First, let's inspect the table structure
    rows = result_grid.find_elements(
        By.XPATH,
        ".//tbody/tr[(contains(@class, 'rgRow') or contains(@class, 'rgAltRow'))]"
    )
    print(f"\nFound {len(rows)} rows in table")
    
    # Inspect first few rows to understand structure
    print("\nInspecting first 3 rows:")
    for i, row in enumerate(rows[:3]):
        try:
            cells = row.find_elements(By.XPATH, ".//td")
            print(f"\nRow {i}: {len(cells)} cells")
            for j, cell in enumerate(cells):
                cell_text = cell.text.strip()
                if cell_text:
                    print(f"  Cell {j}: '{cell_text[:50]}'")
        except Exception as e:
            print(f"  Error inspecting row {i}: {e}")
    
    # Extract just letter I
    letter_ingredients = scraper._scrape_letter(driver, result_grid, "I")
    
    print(f"\nExtracted {len(letter_ingredients)} ingredients for letter I")
    print(f"Ingredients found: {list(letter_ingredients.keys())}")
    
    # Look for Ibuprofen (try different variations)
    ibuprofen_found = False
    for key in letter_ingredients.keys():
        if "ibuprofen" in key.lower():
            print(f"\nFound Ibuprofen variant: {key}")
            ibuprofen_data = letter_ingredients[key]
            print(f"  INN Name: {ibuprofen_data.get('inn_name', 'N/A')}")
            print(f"  INN Code: {ibuprofen_data.get('inn_code', 'N/A')}")
            print(f"  Drugs: {len(ibuprofen_data.get('drugs', {}))}")
            ibuprofen_found = True
    
    if "Ibuprofenum INN" in letter_ingredients:
        ibuprofen_data = letter_ingredients["Ibuprofenum INN"]
        print(f"\nFound Ibuprofenum INN:")
        print(f"  INN Name: {ibuprofen_data.get('inn_name', 'N/A')}")
        print(f"  INN Code: {ibuprofen_data.get('inn_code', 'N/A')}")
        print(f"  Drugs: {len(ibuprofen_data.get('drugs', {}))}")
        
        # Show first few drugs
        drugs = ibuprofen_data.get('drugs', {})
        if drugs:
            print(f"\n  Sample drugs:")
            for i, (drug_name, drug_info) in enumerate(list(drugs.items())[:5]):
                print(f"    - {drug_name}: {drug_info.get('form_strength', 'N/A')}")
    else:
        print("\nIbuprofenum INN not found. Available ingredients:")
        for ingredient_key in list(letter_ingredients.keys())[:10]:
            print(f"  - {ingredient_key}")
    
    driver.quit()
    
except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()
    if 'driver' in locals():
        driver.quit()

