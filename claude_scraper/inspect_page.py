#!/usr/bin/env python3
"""
Page Structure Inspector
Use this to understand the actual HTML structure before scraping
"""

import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def inspect_page(letter='A'):
    """Inspect the page structure for a given category"""

    options = webdriver.ChromeOptions()
    # options.add_argument('--headless')  # Comment out to see the browser

    driver = webdriver.Chrome(options=options)
    wait = WebDriverWait(driver, 10)

    try:
        url = f"https://old.serlyfjaskra.is/ATCList.aspx?FirstLetter={letter}&d=1"
        print(f"Loading: {url}")
        driver.get(url)
        time.sleep(3)

        # Save full HTML
        with open(f'page_structure_{letter}.html', 'w', encoding='utf-8') as f:
            f.write(driver.page_source)
        print(f"Saved full HTML to page_structure_{letter}.html")

        # Find the grid
        print("\n" + "=" * 60)
        print("GRID STRUCTURE:")
        print("=" * 60)

        grids = driver.find_elements(By.CSS_SELECTOR, "div.RadGrid")
        print(f"Found {len(grids)} RadGrid containers")

        tables = driver.find_elements(By.CSS_SELECTOR, "table.rgMasterTable")
        print(f"Found {len(tables)} rgMasterTable tables")

        # Examine first few rows
        print("\n" + "=" * 60)
        print("FIRST FEW ROWS:")
        print("=" * 60)

        rows = driver.find_elements(By.CSS_SELECTOR, "table.rgMasterTable tbody tr")
        print(f"Total rows: {len(rows)}")

        for idx, row in enumerate(rows[:5]):  # First 5 rows
            print(f"\nRow {idx}:")
            print(f"  Class: {row.get_attribute('class')}")

            cells = row.find_elements(By.TAG_NAME, "td")
            print(f"  Cells: {len(cells)}")

            for cell_idx, cell in enumerate(cells):
                text = cell.text.strip()
                html = cell.get_attribute('innerHTML')[:100]
                print(f"    Cell {cell_idx}: '{text}' | HTML: {html}...")

            # Check for expand button
            try:
                expand = row.find_element(By.CSS_SELECTOR, "input[type='image'], a")
                print(f"  Expand element found: {expand.tag_name}")
                if expand.tag_name == 'input':
                    print(f"    Src: {expand.get_attribute('src')}")
            except:
                print(f"  No expand element")

        # Try expanding first row
        print("\n" + "=" * 60)
        print("TESTING EXPAND:")
        print("=" * 60)

        try:
            first_row = rows[0]
            expand_btn = first_row.find_element(By.CSS_SELECTOR, "input[type='image'], a")
            print("Clicking expand button...")
            expand_btn.click()
            time.sleep(2)

            # Look for detail table
            detail_tables = driver.find_elements(By.CSS_SELECTOR, "table.rgDetailTable")
            print(f"Detail tables after expand: {len(detail_tables)}")

            if detail_tables:
                detail_rows = detail_tables[0].find_elements(By.CSS_SELECTOR, "tbody tr")
                print(f"Detail rows: {len(detail_rows)}")

                for idx, row in enumerate(detail_rows[:3]):
                    print(f"\nDetail Row {idx}:")
                    cells = row.find_elements(By.TAG_NAME, "td")
                    for cell_idx, cell in enumerate(cells):
                        print(f"  Cell {cell_idx}: '{cell.text.strip()}'")

        except Exception as e:
            print(f"Could not test expand: {e}")

        # Keep browser open for manual inspection
        print("\n" + "=" * 60)
        print("Browser will stay open for 30 seconds for manual inspection...")
        print("=" * 60)
        time.sleep(30)

    finally:
        driver.quit()


if __name__ == "__main__":
    import sys

    letter = sys.argv[1] if len(sys.argv) > 1 else 'A'
    print(f"Inspecting category: {letter}")
    inspect_page(letter)
