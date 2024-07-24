from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime, timedelta
import csv
import time

# from webdriver_manager.chrome import ChromeDriverManager

# driver = webdriver.Chrome(ChromeDriverManager().install())
def setup_driver():
    chrome_options = Options()
    # chrome_options.add_argument("--headless")
    # Run in headless mode
    service = Service(r'chromedriver.exe')
    return webdriver.Chrome(service=service, options=chrome_options)

def scrape_concert_data(url):
    driver = setup_driver()
    time.sleep(5)
    # WebDriverWait(driver, 1).until(EC.presence_of_element_located((By.CLASS_NAME, 'table')))

    concerts = []
    page = 1
    one_year_ago = datetime.now() - timedelta(days=365)

    try:
        while True:
            driver.get(f"{url}&page={page}")
            print("we have reached here")
            # rows = driver.find_elements(By.CSS_SELECTOR, "#concert-table tr")
            # print(rows)
            # WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CLASS_NAME, "table")))
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "table.table-condensed"))
            )
            print("we have reached here")
            # rows = driver.find_elements(By.TAG_NAME, "tbody")
            rows = driver.find_elements(By.CSS_SELECTOR, "table.table-condensed tbody tr")
            # print(rows)
            print(len(rows))
            breakpoint()
            for row in rows[:-1]:
                print(row)
                try:
                    date = row.find_element(By.CSS_SELECTOR, "td:nth-child(1) span").text.strip()
                    concert = row.find_element(By.CSS_SELECTOR, "td:nth-child(2) strong a").text.strip()
                    venue = row.find_element(By.CSS_SELECTOR, "td:nth-child(3) span a").text.strip()
                    location = row.find_element(By.CSS_SELECTOR, "td:nth-child(4) span a").text.strip()

                    try:
                        concert_date = datetime.strptime(date, '%b %d, %Y')
                    except ValueError:
                        continue  # Skip rows with invalid date format
                    
                    if concert_date < one_year_ago:
                        return concerts  # Stop if we've gone beyond one year

                    concerts.append({
                        'date': date,
                        'concert': concert,
                        'venue': venue,
                        'location': location
                    })
                    print(concerts)
                except Exception as e:
                    continue
            # Check if there's a next page
            print("we have exited loop")
            next_button = driver.find_elements(By.CSS_SELECTOR, ".next_page:not(.disabled)")
            if not next_button:
                print("no next button")
                continue

            page += 1
            time.sleep(1)  # Be polite, delay between requests

    finally:
        driver.quit()

    return concerts

def save_to_csv(concerts, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['date', 'artist', 'venue'])
        writer.writeheader()
        for concert in concerts:
            writer.writerow(concert)

# Main execution
url = "https://www.concertarchives.org/locations/toronto-on?date=past"
concerts = scrape_concert_data(url)
save_to_csv(concerts, 'toronto_concerts_past_year.csv')

print(f"Scraped {len(concerts)} concerts from the past year.")
