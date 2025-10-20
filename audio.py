from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import csv


def setup_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    return webdriver.Chrome(options=options)


def open_main_page(driver):
    driver.get("https://findafair.societyforscience.org/")
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "lstCountries")))
    time.sleep(1)


def select_fair_type(driver, fair_type):
    fair_radio = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, f"//input[@value='{fair_type}']"))
    )
    driver.execute_script("arguments[0].scrollIntoView(true);", fair_radio)
    time.sleep(0.5)
    driver.execute_script("arguments[0].click();", fair_radio)
    time.sleep(1)


def perform_search(driver, filter_type, value):
    if filter_type == "Country":
        dropdown = Select(driver.find_element(By.ID, "lstCountries"))
    else:
        dropdown = Select(driver.find_element(By.ID, "lstStates"))

    dropdown.select_by_visible_text(value)
    time.sleep(1)
    driver.find_element(By.ID, "btnSearch").click()
    time.sleep(3)


def extract_results(driver, csv_writer, fair_type, filter_type, filter_value):
    try:
        fairs = driver.find_elements(By.CSS_SELECTOR, "div#divFairDisplay .row.m-bt-20")
        if not fairs:
            print(f"⚠️ No results for {fair_type} - {filter_type}: {filter_value}")
            return

        for fair in fairs:
            try:
                fair_name = fair.find_element(By.TAG_NAME, "a").text.strip()
                full_text = fair.text.strip().split("\n")

                fair_code = full_text[0] if full_text else ""
                location = full_text[1] if len(full_text) > 1 else ""
                competition_dates = ""
                entry_deadline = ""
                contact_person = ""
                contact_email = ""
                territories = []

                for i, line in enumerate(full_text):
                    if line.strip() == "Competition Dates":
                        competition_dates = full_text[i + 1] if i + 1 < len(full_text) else ""
                    elif line.strip() == "Entry Deadline":
                        entry_deadline = full_text[i + 1] if i + 1 < len(full_text) else ""
                    elif line.strip() == "Contact Person":
                        contact_person = full_text[i + 1] if i + 1 < len(full_text) else ""
                    elif line.strip() == "Territory":
                        territories = full_text[i + 1:]

                # Extract email directly
                try:
                    contact_email = fair.find_element(By.XPATH, ".//a[contains(@href, 'mailto:')]").get_attribute("href").replace("mailto:", "").strip()
                except:
                    contact_email = ""

                csv_writer.writerow([
                    fair_type, filter_type, filter_value, fair_code, fair_name,
                    location, competition_dates, entry_deadline,
                    contact_person, contact_email, ", ".join(territories)
                ])

                print(f"✅ Scraped {fair_code} - {fair_name} ({filter_value})")

            except Exception as fair_error:
                print(f"⚠️ Error scraping fair: {fair_error}")

    except Exception as e:
        print(f"⚠️ Error extracting results: {e}")


def scrape_fairs(driver, fair_type, csv_writer, filter_type, values):
    total = len(values)
    for idx, value in enumerate(values, 1):
        print(f"[{idx}/{total}] Scraping {fair_type} - {filter_type}: {value}")
        open_main_page(driver)
        select_fair_type(driver, fair_type)
        perform_search(driver, filter_type, value)
        extract_results(driver, csv_writer, fair_type, filter_type, value)


def main():
    driver = setup_driver()

    output_file = "all_fairs_combined.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Fair Type", "Filter Type", "Filter Value", "Fair Code", "Fair Name",
            "Location", "Competition Dates", "Entry Deadline",
            "Contact Person", "Contact Email", "Territories"
        ])

        # Load filters
        open_main_page(driver)
        select_fair_type(driver, "chkISEF")
        country_dropdown = Select(driver.find_element(By.ID, "lstCountries"))
        state_dropdown = Select(driver.find_element(By.ID, "lstStates"))

        countries = [option.text for option in country_dropdown.options if option.text.strip() and "Select" not in option.text]
        states = [option.text for option in state_dropdown.options if option.text.strip() and "Select" not in option.text]

        # ISEF
        print("\n🔍 Starting ISEF Fairs scraping by Country...")
        scrape_fairs(driver, "chkISEF", writer, "Country", countries)
        print("\n🔍 Starting ISEF Fairs scraping by State...")
        scrape_fairs(driver, "chkISEF", writer, "State", states)

        # MSC
        print("\n🔍 Starting MSC Fairs scraping by Country...")
        scrape_fairs(driver, "chkMSC", writer, "Country", countries)
        print("\n🔍 Starting MSC Fairs scraping by State...")
        scrape_fairs(driver, "chkMSC", writer, "State", states)

    driver.quit()
    print(f"\n✅ Scraping complete. Data saved in '{output_file}'.")


if __name__ == "__main__":
    main()
