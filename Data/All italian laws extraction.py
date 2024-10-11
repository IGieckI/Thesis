import re
import os
import time
import json
import PyPDF2
import pandas as pd
import lxml.etree as ET

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

isLinux = True
default_linux_path = os.getcwd().replace("/Data", "/Documents/Downloaded") if "/Data" in os.getcwd() else os.getcwd() + "/Documents/Downloaded"
default_windows_path = os.getcwd().replace("\\Data", "\\Documents\\Downloaded") if "\\Data" in os.getcwd() else os.getcwd() + "\\Documents\\Downloaded"
default_path = default_linux_path if isLinux else default_windows_path

DEFAULT_SAVE_DIR = default_path.replace("/Downloaded", "/Generated") if isLinux else default_path.replace("\\Downloaded", "\\Generated")
CHROME_DRIVERS_PATH = "/home/giacomo.antonelli/work/chromedriver-linux64/chromedriver" if isLinux else "C:\\Users\\giaco\\Downloads\\chromedriver-win64\\chromedriver.exe"
# Unibo: /chromedriver-linux64/chromedriver or /home/antonelli2/chromedriver-linux64/chromedriver
# WSL: /home/giacomo/chromedriver-linux64/chromedriver

COSTITUZIONE_CSV = DEFAULT_SAVE_DIR + ("/Costituzione.csv" if isLinux else "\\Costituzione.csv")

CODICE_PENALE_PDF = default_path + ("/Codice penale well formatted edited.pdf" if isLinux else "\\Codice penale well formatted edited.pdf")
CODICE_PENALE_CSV = DEFAULT_SAVE_DIR + ("/Codice penale.csv" if isLinux else "\\Codice penale.csv")

CPP_CSV = DEFAULT_SAVE_DIR + ("/Codice procedura penale.csv" if isLinux else "\\Codice procedura penale.csv")

CPA_CSV = DEFAULT_SAVE_DIR + ("/Codice processo amministrativo.csv" if isLinux else "\\Codice procedura amministrativo.csv")

REF_MERG = DEFAULT_SAVE_DIR + ('/references_merged.csv' if isLinux else '\\references_merged.csv')
ALREADY_SCRAPED_DLGS_JSON = DEFAULT_SAVE_DIR + ('/scraped_dlgs.json' if isLinux else '\\scraped_dlgs.json')
DLGS_CSV = DEFAULT_SAVE_DIR + ('/dlgs.csv' if isLinux else '\\dlgs.csv')

ALL_ITALIAN_LAWS_CSV = DEFAULT_SAVE_DIR + ("/All Italian laws.csv" if isLinux else "\\All Italian laws.csv")
ALL_ITALIAN_LAWS_SCRAPED_JSON = DEFAULT_SAVE_DIR + ("/All laws.json" if isLinux else "\\All laws.json")

LAWS_CSV = DEFAULT_SAVE_DIR + ("/laws.csv" if isLinux else "\\laws.csv")


# Utility functions and constants
def write_to_file(filename, content):
    with open(filename, 'w+') as f:
        f.write(content)

def read_from_file(filename):
    with open(filename, 'r') as f:
        return f.read()

# JSON stuff
def write_list_to_json(lst, filename):
    with open(filename, 'w') as json_file:
        json.dump(lst, json_file)

def read_list_from_json(filename):
    with open(filename, 'r') as json_file:
        py_list = json.load(json_file)
    return py_list

# CSV stuff
def save_df_to_csv(df, filename):
    df.to_csv(filename, index=False)

def read_df_from_csv(filename):
    return pd.read_csv(filename)

# Different kind of text extraction from each type of file
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text    

def clearLawContent(content):
    # Check for comma class tags
    matches =  re.findall(r'<span class="art_text_in_comma">(.*?)</span>', content, re.DOTALL)
    content = ' '.join(matches) if matches else content
    content = re.sub(r"<div class=\"ins-akn\" eid=\"ins_\d+\">\(\(", "", content, flags=re.DOTALL)
    content = re.sub(r"\n", "", content, flags=re.DOTALL)
    content = re.sub(r"<br>", "", content, flags=re.DOTALL)

    
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
    content = content.replace("\n", "")
    content = content.replace("((", "")
    content = content.replace("))", "")
    
    return content.strip()

def extractCommaNumber(articleElement):
    articleElement = articleElement.strip()
    if "art." in articleElement:
        return articleElement.split(" ")[1]
    return articleElement

def extractArticleNumber(articleElement):
    match = re.search(r'n\..*?(\d+)', articleElement)
    if match:
        return int(match.group(1))
    raise Exception("Article number not found")

class NormattivaAllLawsScraper:
    def __init__(self, driver_path, headless=True):
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--no-sandbox")
        self.service = Service(driver_path)
        self.driver = webdriver.Chrome(service=self.service, options=chrome_options)
    
    # Get the originario version of the law
    def fill_field(self, field_id, value):
        input_field = self.driver.find_element(By.ID, field_id)
        input_field.clear()
        input_field.send_keys(value)
    
    def get_years(self):
        years = self.driver.find_elements(By.CLASS_NAME, "btn-secondary")
        time.sleep(1)
        return years
        
    # Get the text of a specific article
    def get_articles(self, year, law_number):
        print(f"{year} - {law_number}")
        articles_list = []
        
        # Ensure is multivigente version
        time.sleep(1)
        
        # Get articles
        albero = self.driver.find_element(By.ID, "albero")
        articles = albero.find_elements(By.CLASS_NAME, "numero_articolo")
                
        for i, article in enumerate(articles):
            article_number = article.text.strip()
            
            multiplicatives = ["bis", "ter", "quater", "quinquies", "sexies"]
            if (article_number == "") or ("allegato" in article_number.lower()) or ((not article_number.isdigit()) and ("art" not in article_number) and ("Art" not in article_number) and (not any(multiplicative in article_number for multiplicative in multiplicatives))):
                print("Out ", article.text)
                continue
            
            try:
                time.sleep(1)
                article.click()                    
                time.sleep(1)
            except:                
                continue
            
            text = self.driver.find_elements(By.CLASS_NAME, "art-commi-div-akn")
            if not text:
                text = self.driver.find_elements(By.CLASS_NAME, "art-just-text-akn")
            if not text:
                print("Error in article: ", article.text)
                return articles_list            
                
            text = text[0]
            
            law_content = text.get_attribute('outerHTML')
            law_content = law_content.replace("\n", "")
            law_content = clearLawContent(law_content)
                            
            articles_list.append({ 
                "law_source": f"Legge {law_number}",
                "year": year,
                "law_number": article_number,
                "law_text": law_content
            })

            print(articles_list[-1])

        return articles_list

    # Navigate to a specific page
    def navigate_to_page(self, url):
        self.driver.get(url)
        
    # Close the driver
    def close(self):
        self.driver.quit()
 
# Check if the laws have already been scraped
scraped_pages = []
df_all_laws = pd.DataFrame()
if os.path.exists(ALL_ITALIAN_LAWS_SCRAPED_JSON):
    scraped_pages = read_from_file(ALL_ITALIAN_LAWS_SCRAPED_JSON)    
    df_all_laws = read_df_from_csv(ALL_ITALIAN_LAWS_CSV)

scraper = NormattivaAllLawsScraper(CHROME_DRIVERS_PATH, headless=True)
scraper.navigate_to_page("https://www.normattiva.it/ricerca/elencoPerData")
years = scraper.get_years()
data = []

for year in range(2024, 1861, -1): # from 2024 to 1861
    scraper.driver.get("https://www.normattiva.it/ricerca/avanzata")
    scraper.fill_field("annoProvvedimento", year)
    scraper.driver.find_element(By.CSS_SELECTOR, "[type*='submit']").click()
    time.sleep(0.5)
    
    validPage = True
    curr_page = 1
    law_urls = []
    
    while validPage:
        laws = scraper.driver.find_elements(By.CSS_SELECTOR, "[title^='Dettaglio atto']")
        
        for law in laws:
            law_url = law.get_attribute('href')
            if law_url and "LEGGE" in law.text:                                
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
            continue
        scraper.driver.get(url)
        
        page_title = scraper.driver.title
        law_number = extractArticleNumber(page_title)
        
        articles = scraper.get_articles(year, law_number)
        
        # Append individual articles to data, not the entire list
        data.extend(articles)
        
        scraped_pages.append(f"{year}/{i}")
        write_to_file(ALL_ITALIAN_LAWS_SCRAPED_JSON, json.dumps(scraped_pages))
        
        # Create a temporary DataFrame for the new articles and append to the main DataFrame
        df_tmp = pd.DataFrame(articles)  # Not `data` but `articles`
        df_all_laws = pd.concat([df_all_laws, df_tmp], ignore_index=True)
        
        save_df_to_csv(df_all_laws, ALL_ITALIAN_LAWS_CSV)

# df_all_laws now contains all the data
df_all_laws.head()