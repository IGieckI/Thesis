import re
import os
import time
import json

import pandas as pd
from tqdm import tqdm
import lxml.etree as ET

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

isLinux = True
default_linux_path = os.getcwd().replace("/Data", "/Documents/Downloaded") if "/Data" in os.getcwd() else os.getcwd() + "/Documents/Downloaded"
default_windows_path = os.getcwd().replace("\\Data", "\\Documents\\Downloaded") if "\\Data" in os.getcwd() else os.getcwd() + "\\Documents\\Downloaded"
default_path = default_linux_path if isLinux else default_windows_path

DEFAULT_SAVE_DIR = default_path.replace("/Downloaded", "/Generated") if isLinux else default_path.replace("\\Downloaded", "\\Generated")
CHROME_DRIVERS_PATH = "/antonelli2/chromedriver-linux64/chromedriver"
# Unibo docker:/antonelli2/chromedriver-linux64/chromedriver
# Unibo: /home/antonelli2/chromedriver-linux64/chromedriver
# WSL: /home/giacomo/chromedriver-linux64/chromedriver
# Maggioli: /home/giacomo.antonelli/work/chromedriver-linux64/chromedriver
ALL_LAWS_CSV = DEFAULT_SAVE_DIR + ("/All laws extracted.csv" if isLinux else "\\All laws extracted.csv")
SCRAPED_PAGES_JSON = DEFAULT_SAVE_DIR + ("/Scraped pages.json" if isLinux else "\\Scraped pages.json")

# Utility functions and constants
def write_to_json_file(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
        
def read_from_json_file(filename):
    f = open(filename, "r").read()
    return json.loads(f)

def save_df_to_csv(df, filename):
    df.to_csv(filename, index=False)   

def extractCommaNumber(articleElement):
    articleElement = articleElement.strip()
    if "art." in articleElement:
        return articleElement.split(" ")[1]
    return articleElement

def extractArticleNumber(articleElement):
    match = re.search(r'n\..*?(\d+)', articleElement)
    if match:
        return int(match.group(1))
    return None

def clearCommaContent(content):
    if "<" not in content:
        return content
    
    # Check for comma class tags
    match = re.search(r'<span class="art_text_in_comma">(.*?)</span>', content, re.DOTALL)
    comma_content = match.group(1) if match else content
    comma_content = re.sub(r"<div class=\"ins-akn\" eid=\"ins_\d+\">\(\(", "", comma_content, flags=re.DOTALL)
    comma_content = re.sub(r"\)\)</div>", "", comma_content, flags=re.DOTALL)
    comma_content = re.sub(r"\n", "", comma_content, flags=re.DOTALL)
    comma_content = re.sub(r"<br>", "", comma_content, flags=re.DOTALL)
    
    # Check for a tags
    aPattern = re.compile(r'<a.*?>(.*?)</a>', re.DOTALL)
    matches = aPattern.findall(content)
    if matches:
        for match in matches:
            content = re.sub(r'<a.*?>.*?</a>', match, content, count=1)

    # Check for span tags
    sPattern = re.compile(r'<span.*?>(.*?)</span>', re.DOTALL)
    matches = sPattern.findall(content)
    if matches:
        for match in matches:
            content = re.sub(r'<span.*?>.*?</span>', match, content, count=1)

    # Check for list div tags
    dlPattern = re.compile(r'<div class="pointedList-rest-akn">(.*?)</div>', re.DOTALL)
    matches = dlPattern.findall(content)
    if matches:
        for match in matches:
            content = re.sub(r'<div class="pointedList-rest-akn">.*?</div>', re.escape(match), content, count=1)
    
    # Delete remaining tags
    content = re.sub(r'<.+?>', "", content)
    
    return content

class NormattivaAllLawsScraper:
    def __init__(self, driver_path, headless=True):
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
        self.service = Service(driver_path)
        self.driver = webdriver.Chrome(service=self.service, options=chrome_options)
    
    # Get the originario version of the law
    def fill_field(self, field_id, value):
        input_field = WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.ID, field_id)))
        input_field.clear()
        input_field.send_keys(value)
    
    def get_years(self):
        years = self.driver.find_elements(By.CLASS_NAME, "btn-secondary")
        return years
        
    # Get the text of a specific article
    def get_articles(self):
        articles_list = []
        
        # Ensure is multivigente version
        time.sleep(1)
        
        # Get articles
        try:
            albero = self.driver.find_element(By.ID, "albero")
            articles = albero.find_elements(By.CLASS_NAME, "numero_articolo")
        except:
            return articles_list
                
        for i, article in enumerate(articles):
            if article.text.strip() != "" and (article.text.strip().isdigit() or "art." in article.text):
                try:
                    article.click()                    
                    time.sleep(1)
                    
                    article_title = self.driver.find_element(By.CLASS_NAME, "article-num-akn").text.strip()
                    commas = self.driver.find_elements(By.CLASS_NAME, "art-comma-div-akn")                    
                except:
                    continue
                
                pre_article = ""
                try:
                    pre_article = self.driver.find_element(By.CLASS_NAME, "article-pre-comma-text-akn")
                    pre_article = pre_article.text
                except:
                    pass
                                
                if len(commas) == 0:
                    continue
                
                comma_number = extractCommaNumber(article.text)
                firstTime = True                
                for comma in commas:
                    if firstTime:
                        time.sleep(0.5)
                        firstTime = False
                    
                    # Check if the content contains any multivigenza change
                    try:
                        comma_content = comma.get_attribute('outerHTML')
                    except:
                        continue
                    
                    # Skip if the content contains "<span class="art_text_in_comma">((</span>" or "<span class="art_text_in_comma">))</span>"
                    if "<span class=\"art_text_in_comma\">((</span>" in comma_content or "<span class=\"art_text_in_comma\">))</span>" in comma_content:
                        continue
                    
                    try:
                        comma_number = comma.find_element(By.CLASS_NAME, "comma-num-akn").text.strip()
                    except:
                        continue
                    comma_content_element = comma.text

                    # Clear the output
                    comma_content_element = pre_article + comma_content_element
                    pre_article = ""
                    comma_content = clearCommaContent(comma_content_element)
                                        
                    articles_list.append({ "article_source": "All laws",
                                            "article_text": comma_content.strip()})

        return articles_list

    # Navigate to a specific page
    def navigate_to_page(self, url):
        self.driver.get(url)
        
    # Close the driver
    def close(self):
        self.driver.quit()

# Check if the laws have already been scraped
if os.path.exists(SCRAPED_PAGES_JSON):
    scraped_pages = read_from_json_file(SCRAPED_PAGES_JSON)
    scraped_pages = json.loads(scraped_pages)
    print(type(scraped_pages))
    print(len(scraped_pages))
else:
    scraped_pages = []
    
if os.path.exists(ALL_LAWS_CSV):
    data = pd.read_csv(ALL_LAWS_CSV)
else:
    data = pd.DataFrame()

scraper = NormattivaAllLawsScraper(CHROME_DRIVERS_PATH, headless=True)

#1861, 2025
# [1971, 1981, 1988, 1990, 1997, 1999, 2001, 2003, 2005, 2008, 2009, 2012, 2019]
for year in [1971, 1981, 1988, 1990, 1997, 1999, 2001, 2003, 2005, 2008, 2009, 2012, 2019]:
    scraper.driver.get("https://www.normattiva.it/ricerca/avanzata")
    scraper.fill_field("annoProvvedimento", year)
    scraper.driver.find_element(By.CSS_SELECTOR, "[type*='submit']").click()
    time.sleep(0.5)
    
    validPage = True
    curr_page = 1
    law_urls = []
    
    while validPage:
        # Collect all law detail URLs on the current page
        laws = scraper.driver.find_elements(By.CSS_SELECTOR, "[title^='Dettaglio atto']")
        
        for law in laws:
            law_url = law.get_attribute('href')
            if law_url:
                law_urls.append(law_url)
        
        # Try a new page of laws
        curr_page += 1
        pages_link = scraper.driver.find_elements(By.CLASS_NAME, "page-link")
        validPage = False
        for page in pages_link:
            if page.text == str(curr_page):
                validPage = True
                page.click()
                time.sleep(0.5)
                break
    
    # Visit each law's detail page and scrape the articles
    for i, url in enumerate(law_urls):
        if f"{year}/{i}" in scraped_pages:
            print(f"Skipping {year}/{i}")
            continue
        scraper.driver.get(url)
        articles = scraper.get_articles()

        # Get law's number
        page_title =  scraper.driver.title
        print(page_title, " -> ", end="")
        law_number = extractArticleNumber(page_title)
        print(law_number)
        
        # Add year to all articles
        for article in articles:
            article['year'] = year
            article['article_number'] = law_number
            print(article)

        if articles:
            df_articles = pd.DataFrame(articles)
            data = pd.concat([data, df_articles], ignore_index=True)
            
            save_df_to_csv(data, ALL_LAWS_CSV)
        
        scraped_pages.append(f"{year}/{i}")
        write_to_json_file(SCRAPED_PAGES_JSON, json.dumps(scraped_pages))

scraper.close()

data.head()
