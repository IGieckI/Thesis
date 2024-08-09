import re
import os
import time
import json
import PyPDF2
import pypandoc
import numpy as np
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

# Useful global variables
isLinux = True
default_linux_path = os.getcwd().replace("/Data/Console Python Scripts", "/Documents/Downloaded")
default_windows_path = os.getcwd().replace("\\Data\\Console\ Python\ Scripts", "\\Documents\\Downloaded\\")
default_path = default_linux_path if isLinux else default_windows_path

DEFAULT_SAVE_DIR = default_path.replace("/Downloaded", "/Generated") if isLinux else default_path.replace("\\Downloaded", "\\Generated")
CHROME_DRIVERS_PATH = "/home/antonelli/chromedriver-linux64/chromedriver" if isLinux else "C:\\Users\\giaco\\Downloads\\chromedriver-win64\\chromedriver.exe"
REF_MERG = DEFAULT_SAVE_DIR + ('/references_merged.csv' if isLinux else '\\references_merged.csv')
INV_LAWS = DEFAULT_SAVE_DIR + ('/invalid_laws.json' if isLinux else '\\invalid_laws.json')
LAWS = DEFAULT_SAVE_DIR + ('/laws.csv' if isLinux else '\\laws.csv')

# Core code
class NormattivaScraper:
    def __init__(self, driver_path, headless=True):
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")
        self.service = Service(driver_path)
        self.driver = webdriver.Chrome(service=self.service, options=chrome_options)
    
    # Get the originario version of the law
    def fill_field(self, field_id, value):
        input_field = self.driver.find_element(By.ID, field_id)
        input_field.clear()
        input_field.send_keys(value)
    
    # Get the text of a specific article
    def get_article_text(self, numeroProvvedimento, anno, article_num=[]):
        articles_list = []
        time.sleep(0.5)

        self.fill_field("numeroProvvedimento", numeroProvvedimento)
        self.fill_field("annoProvvedimento", anno)
        
        try:
            self.driver.find_element(By.CSS_SELECTOR, "[type*='submit']").click()
            self.driver.find_elements(By.CSS_SELECTOR, "[title*='Dettaglio atto']")[0].click()
            time.sleep(0.5)
        except:
            return []
        
        # Ensure is multivigente version
        multivigente_button = self.driver.find_element(By.XPATH, '//a[contains(@href, "multivigenza")]')
        multivigente_button.click()
        
        # Get articles
        albero = self.driver.find_element(By.ID, "albero")
        articles = albero.find_elements(By.CLASS_NAME, "numero_articolo")
        print("Article len: ", len(articles))
                
        for article in articles:            
            if article.text.strip() != "" and article.text[0].isdigit() and "orig" not in article.text and "Allegato" not in article.text and "agg" not in article.text:
                try:
                    article.click()
                except:
                    print("Failed clicking article")
                    continue
                
                time.sleep(0.5)
                try:
                    article_title = self.driver.find_element(By.CLASS_NAME, "article-num-akn").text.strip()
                    commas = self.driver.find_elements(By.CLASS_NAME, "art-comma-div-akn")
                except:
                    continue
                print("Comma len: ", len(commas))
                if len(commas) == 0:
                    continue
                
                firstTime = True                
                for i, comma in enumerate(commas):
                    if firstTime:
                        time.sleep(0.5)
                        firstTime = False
                    
                    time.sleep(0.2)
                    print(numeroProvvedimento, anno, article_title, i)
                    
                    # Check if the content contains any multivigenza change
                    try:
                        comma_content = comma.get_attribute('outerHTML')
                    except:
                        continue
                    
                    # Skip if the content contains "<span class="art_text_in_comma">((</span>" or "<span class="art_text_in_comma">))</span>"
                    if ">((<" in comma_content or ">))<" in comma_content:
                        continue
                    print("in")
                    try:
                        comma_number = comma.find_element(By.CLASS_NAME, "comma-num-akn").text.strip()
                    except:
                        continue
                    comma_content_element = comma.text#find_element(By.CLASS_NAME, "art_text_in_comma")

                    #print(comma_number, comma_content)
                    
                    # Clear the output
                    match =  re.search(r'<span class="art_text_in_comma">(.*?)</span>', comma_content, re.DOTALL)
                    comma_content = match.group(1) if match else comma_content
                    comma_content = re.sub(r"<div class=\"ins-akn\" eid=\"ins_\d+\">\(\(", "", comma_content, flags=re.DOTALL)
                    comma_content = re.sub(r"\)\)</div>", "", comma_content, flags=re.DOTALL)
                    comma_content = re.sub(r"\n", "", comma_content, flags=re.DOTALL)
                    comma_content = re.sub(r"<br>", "", comma_content, flags=re.DOTALL)
                    
                    articles_list.append({ "Dlgs":f"{numeroProvvedimento}/{anno}".strip(),
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

def save_articles(articles, filename):
    df = pd.DataFrame(articles)
    df.to_csv(filename, index=False)
    
def save_invalid(invalids, filename):
    with open(filename, "w") as file:
        file.write(json.dumps(invalids))

scraper = NormattivaScraper(CHROME_DRIVERS_PATH, headless=True)

# Check if the laws have already been scraped
scraped_laws_set = set()
scraped_laws = []
if os.path.exists(LAWS):
    df = pd.read_csv(LAWS)
    first_column = df.iloc[:, 0]
    scraped_laws = df.to_dict('records')
    
    # put the already scraped laws in a set to not scrape them again
    scraped_laws_set = set(first_column)

invalid_laws_set = set()
if os.path.exists(INV_LAWS):
    with open(INV_LAWS, "r") as file:
        invalid_laws = json.load(file)
    invalid_laws_set = set(invalid_laws)

scraped_laws_set.update(invalid_laws_set)

df = pd.read_csv(REF_MERG)

for index, row in tqdm(df.iterrows(), total=df.shape[0]): # 445
    # Check if it's a D. lgs.
    if '/' not in row['Source'] or row['Source'].strip() in scraped_laws_set:
        continue
    scraped_laws_set.add(row['Source'])
    print(f"Scraping |{row['Source']}|")
    num, year = row['Source'].split("/")
            
    scraper.navigate_to_page("https://www.normattiva.it/ricerca/avanzata")
    res = scraper.get_article_text(num, year)
    if res == []:
        invalid_laws_set.add(f"{num}/{year}".strip())
    else:
        scraped_laws.extend(res)
    
    #if scraped_laws and len(scraped_laws) % 3 == 0:
    save_articles(scraped_laws, LAWS)
    save_invalid(list(invalid_laws_set), INV_LAWS)

save_articles(scraped_laws, LAWS)
save_invalid(list(invalid_laws_set), INV_LAWS)