import re
import os
import time
import json
import PyPDF2
import pypandoc
import pandas as pd
from tqdm import tqdm
import lxml.etree as ET


from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

isLinux = True
default_linux_path = os.getcwd().replace("/Data", "/Documents/Downloaded") if "/Data" in os.getcwd() else os.getcwd() + "/Documents/Downloaded"
default_windows_path = os.getcwd().replace("\\Data", "\\Documents\\Downloaded") if "\\Data" in os.getcwd() else os.getcwd() + "\\Documents\\Downloaded"
default_path = default_linux_path if isLinux else default_windows_path

DEFAULT_SAVE_DIR = default_path.replace("/Downloaded", "/Generated") if isLinux else default_path.replace("\\Downloaded", "\\Generated")
CHROME_DRIVERS_PATH = "/chromedriver-linux64/chromedriver" if isLinux else "C:\\Users\\giaco\\Downloads\\chromedriver-win64\\chromedriver.exe"

ALL_LAWS_CSV = DEFAULT_SAVE_DIR + ("/All laws extracted.csv" if isLinux else "\\All laws extracted.csv")
ALL_LAWS_SCRAPED_JSON = DEFAULT_SAVE_DIR + ("/All laws.json" if isLinux else "\\All laws.json")

# Utility functions and constants
def write_to_file(filename, content):
    with open(filename, 'w+') as f:
        f.write(content)

def read_from_file(filename):
    with open(filename, 'r') as f:
        return f.read()
    
def save_df_to_csv(df, filename):
    df.to_csv(filename, index=False)

# Different kind of text extraction from each type of file
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text    

def extract_text_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    return ET.tostring(root, encoding='unicode', method='text')

def extract_text_from_rtf(rtf_path):
    return pypandoc.convert_file(rtf_path, 'plain', format='rtf')


def split_text(text, pattern):
    parts = re.split(pattern, text, flags=re.MULTILINE)
    
    parts = [part for part in parts if part]
    
    if not re.match(pattern, parts[0]):
        parts = parts[1:]
    
    return parts

def split_text(text, max_chunk_size=7000, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(separators=[
        "\n\n",
        "\n",
        ".",
    ],
    chunk_size=max_chunk_size,
    chunk_overlap=chunk_overlap)
    
    return text_splitter.split_text(text)

def clearCommaContent(content):
    if "<" not in content:
        return content
    
    # Check for comma class tags
    match =  re.search(r'<span class="art_text_in_comma">(.*?)</span>', comma_content, re.DOTALL)
    comma_content = match.group(1) if match else comma_content
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
        input_field = self.driver.find_element(By.ID, field_id)
        input_field.clear()
        input_field.send_keys(value)
    
    def get_years(self):
        years = self.driver.find_elements(By.CLASS_NAME, "btn-secondary")
        time.sleep(1)
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
                
        for i, article in enumerate(articles[1:]):
            if article.text.strip() != "" and article.text[0].isdigit():                
                try:
                    article.click()                    
                    time.sleep(1)
                    
                    article_title = self.driver.find_element(By.CLASS_NAME, "article-num-akn").text.strip()
                    commas = self.driver.find_elements(By.CLASS_NAME, "art-comma-div-akn")                    
                except:
                    continue
                
                if len(commas) == 0:
                    continue
                
                firstTime = True                
                for comma in commas:
                    if firstTime:
                        time.sleep(1)
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
                    comma_content_element = comma.text#find_element(By.CLASS_NAME, "art_text_in_comma")

                    # Clear the output
                    comma_content = clearCommaContent(comma_content_element)
                    
                    print(article_title, comma_number, comma_content)
                    
                    articles_list.append({ "Source": "All laws",
                                            "Article": article_title,
                                            "Comma number": comma_number,
                                            "Comma content": comma_content.strip()}) # Numeration not working in case of -bis... extract

        return articles_list

    # Navigate to a specific page
    def navigate_to_page(self, url):
        self.driver.get(url)
        
    # Close the driver
    def close(self):
        self.driver.quit()

# Check if the laws have already been scraped
scraped_pages_set = set()
if os.path.exists(ALL_LAWS_SCRAPED_JSON):
    scraped_pages = read_from_file(ALL_LAWS_SCRAPED_JSON)
    
   # Assuming the JSON is an array of values
    scraped_pages_set = set(scraped_pages)

scraper = NormattivaAllLawsScraper(CHROME_DRIVERS_PATH, headless=True)
data = []

for year in range(1861, 2025):
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
        if f"{year}/{i}" in scraped_pages_set:
            print(f"Skipping {year}/{i}")
            continue
        scraper.driver.get(url)
        articles = scraper.get_articles()
        print(articles)
        if articles:
            data.append(articles)
            
            df_all_laws = pd.DataFrame(data)
            save_df_to_csv(df_all_laws, ALL_LAWS_CSV)
        
        scraped_pages_set.add(f"{year}/{i}")
        write_to_file(ALL_LAWS_SCRAPED_JSON, json.dumps(list(scraped_pages_set)))        

df_all_laws.head()