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
CHROME_DRIVERS_PATH = "/antonelli2/tmp/chromedriver-linux64/chromedriver" if isLinux else "C:\\Users\\giaco\\Downloads\\chromedriver-win64\\chromedriver.exe"

CODICE_PENALE_PDF = default_path + ("/Codice penale well formatted edited.pdf" if isLinux else "\\Codice penale well formatted edited.pdf")
CODICE_PENALE_CSV = DEFAULT_SAVE_DIR + ("/Codice penale well formatted edited.csv" if isLinux else "\\Codice penale well formatted edited.csv")

CPP_CSV = DEFAULT_SAVE_DIR + ("/Codice procedura penale.csv" if isLinux else "\\Codice procedura penale.csv")

REF_MERG = DEFAULT_SAVE_DIR + ('/references_merged.csv' if isLinux else '\\references_merged.csv')
INV_LAWS_JSON = DEFAULT_SAVE_DIR + ('/invalid_laws.json' if isLinux else '\\invalid_laws.json')
LAWS_CSV = DEFAULT_SAVE_DIR + ('/laws.csv' if isLinux else '\\laws.csv')

ALL_LAWS_CSV = DEFAULT_SAVE_DIR + ("/All laws extracted.csv" if isLinux else "\\All laws extracted.csv")
ALL_LAWS_SCRAPED_JSON = DEFAULT_SAVE_DIR + ("/All laws.json" if isLinux else "\\All laws.json")

# Utility functions and constants
def write_to_file(filename, content):
    with open(filename, 'w+') as f:
        f.write(content)

def read_from_file(filename):
    with open(filename, 'r') as f:
        return f.read()
    
def read_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)
    
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

class NormattivaCpScraper:
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
    
    # Get the text of a specific article
    def get_cp_articles(self):
        articles_list = []
        
        # Ensure is multivigente version
        time.sleep(1)
        
        # Get articles
        albero = self.driver.find_element(By.ID, "albero")
        articles = albero.find_elements(By.CLASS_NAME, "numero_articolo")
        print("Article len: ", len(articles))
                
        for i, article in enumerate(articles[1:]):            
            print(i)
            if article.text.strip() != "" and article.text[0].isdigit():                
                try:
                    article.click()                    
                    time.sleep(1)
                    
                    article_title = self.driver.find_element(By.CLASS_NAME, "article-num-akn").text.strip()
                    commas = self.driver.find_elements(By.CLASS_NAME, "art-comma-div-akn")                    
                except:
                    continue
                
                print("Comma len: ", len(commas))
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
                    #print(comma_content)
                    
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
                    
                    articles_list.append({ "Source": "c.p.",
                                            "Article": article_title,
                                            "Comma number": comma_number,
                                            "Comma content": comma_content.strip()}) # Numeration not working in case of -bis... extract
            else:
                print("Out ", article.text)

        return articles_list

    # Navigate to a specific page
    def navigate_to_page(self, url):
        self.driver.get(url)
        
    # Close the driver
    def close(self):
        self.driver.quit()

scraper = NormattivaCpScraper(CHROME_DRIVERS_PATH, headless=True)
scraper.navigate_to_page("https://www.normattiva.it/uri-res/N2Ls?urn:nir:stato:regio.decreto:1930-10-19;1398")
articles = scraper.get_cp_articles()

df_cp = pd.DataFrame(articles)
df_cp.to_csv(CODICE_PENALE_CSV, index=False)

class NormattivaCppScraper:
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
    
    # Get the text of a specific article
    def get_cpp_articles(self):
        articles_list = []
        
        # Ensure is multivigente version
        time.sleep(1)
        
        # Get articles
        albero = self.driver.find_element(By.ID, "albero")
        articles = albero.find_elements(By.CLASS_NAME, "numero_articolo")
        print("Article len: ", len(articles))
                
        for i, article in enumerate(articles[1:]):            
            print(i)
            if article.text.strip() != "" and article.text[0].isdigit():                
                try:
                    article.click()                    
                    time.sleep(1)
                    
                    article_title = self.driver.find_element(By.CLASS_NAME, "article-num-akn").text.strip()
                    commas = self.driver.find_elements(By.CLASS_NAME, "art-comma-div-akn")                    
                except:
                    continue
                
                print("Comma len: ", len(commas))
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
                    #print(comma_content)
                    
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
                    
                    articles_list.append({ "Source": "c.p.p.",
                                            "Article": article_title,
                                            "Comma number": comma_number,
                                            "Comma content": comma_content.strip()}) # Numeration not working in case of -bis... extract
            else:
                print("Out ", article.text)

        return articles_list

    # Navigate to a specific page
    def navigate_to_page(self, url):
        self.driver.get(url)
        
    # Close the driver
    def close(self):
        self.driver.quit()

scraper = NormattivaCppScraper(CHROME_DRIVERS_PATH, headless=True)
scraper.navigate_to_page("https://www.normattiva.it/uri-res/N2Ls?urn:nir:stato:decreto.del.presidente.della.repubblica:1988-09-22;447")
articles = scraper.get_cpp_articles()

df_cpp = pd.DataFrame(articles)
df_cpp.to_csv(CPP_CSV, index=False)

class NormattivaDlgsScraper:
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
    
    # Get the text of a specific article
    def get_article_text(self, numeroProvvedimento, anno, article_num=[]):
        articles_list = []
            
        self.fill_field("numeroProvvedimento", numeroProvvedimento)
        self.fill_field("annoProvvedimento", anno)

        self.driver.find_element(By.CSS_SELECTOR, "[type*='submit']").click()
        self.driver.find_elements(By.CSS_SELECTOR, "[title*='Dettaglio atto']")[0].click()
        
        time.sleep(2)
        # Ensure is multivigente version
        multivigente_button = self.driver.find_element(By.XPATH, '//a[contains(@href, "multivigenza")]')
        multivigente_button.click()
        
        # Get articles
        albero = self.driver.find_element(By.ID, "albero")
        articles = albero.find_elements(By.CLASS_NAME, "numero_articolo")
        print("Article len: ", len(articles))
                
        for article in articles:            
            if article.text.strip() != "" and article.text[0].isdigit() and "orig" not in article.text and "Allegato" not in article.text and "agg" not in article.text:
                print("In ", article.text)
                try:
                    article.click()
                except:
                    continue
                
                time.sleep(2)
                
                article_title = self.driver.find_element(By.CLASS_NAME, "article-num-akn").text.strip()
                commas = self.driver.find_elements(By.CLASS_NAME, "art-comma-div-akn")
                print("Comma len: ", len(commas))
                if len(commas) == 0:
                    continue
                
                firstTime = True                
                for i, comma in enumerate(commas):
                    if firstTime:
                        time.sleep(1)
                        firstTime = False
                    print(numeroProvvedimento, anno, article_title, i)
                    
                    # Check if the content contains any multivigenza change
                    comma_content = comma.get_attribute('outerHTML')
                    #print(comma_content)
                    
                    # Skip if the content contains "<span class="art_text_in_comma">((</span>" or "<span class="art_text_in_comma">))</span>"
                    if "<span class=\"art_text_in_comma\">((</span>" in comma_content or "<span class=\"art_text_in_comma\">))</span>" in comma_content:
                        continue
                    
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
            else:
                print("Out ", article.text)

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

scraper = NormattivaDlgsScraper(CHROME_DRIVERS_PATH, headless=True)

# Check if the laws have already been scraped
scraped_laws_set = set()
if os.path.exists(LAWS_CSV):
    df_scraped = pd.read_csv(LAWS_CSV)
    first_column = df_scraped.iloc[:, 0]
    scraped_laws = df_scraped.to_dict('records')
    
    # put the already scraped laws in a set to not scrape them again
    scraped_laws_set = set(first_column)
else:
    scraped_laws = []

invalid_laws_set = set()
if os.path.exists(INV_LAWS_JSON):
    invalid_laws = read_json(INV_LAWS_JSON)
    invalid_laws_set = set(invalid_laws)
else:
    invalid_laws = []

scraped_laws_set.update(invalid_laws_set)

df_ref = pd.read_csv(REF_MERG)
articles = []

for index, row in tqdm(df_ref.iterrows(), total=df_ref.shape[0]): # 445
    # Check if it's a D. lgs.
    if '/' not in row['Source'] or row['Source'].strip() in scraped_laws_set:
        continue
    scraped_laws_set.add(row['Source'])
    print(f"Scraping |{row['Source']}|")
    num, year = row['Source'].split("/")
            
    scraper.navigate_to_page("https://www.normattiva.it/ricerca/avanzata")
    res = scraper.get_article_text(num, year)
    if res == []:
        invalid_laws.append(f"{num}/{year}")
    else:
        articles.append(res)
    
    if articles and len(articles) % 3 == 0:
        save_articles([item for sublist in articles for item in sublist] + scraped_laws, LAWS_CSV)
        with open(INV_LAWS_JSON, 'w') as f:
            json.dumps(scraped_laws)

save_articles([item for sublist in articles for item in sublist] + scraped_laws, LAWS_CSV)
with open(INV_LAWS_JSON, 'w') as f:
    json.dumps(scraped_laws)
    
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
    def get_articles(self):
        articles_list = []
        
        # Ensure is multivigente version
        time.sleep(1)
        
        # Get articles
        albero = self.driver.find_element(By.ID, "albero")
        articles = albero.find_elements(By.CLASS_NAME, "numero_articolo")
        print("Article len: ", len(articles))
                
        for i, article in enumerate(articles[1:]):            
            print(i)
            if article.text.strip() != "" and article.text[0].isdigit():                
                try:
                    article.click()                    
                    time.sleep(1)
                    
                    article_title = self.driver.find_element(By.CLASS_NAME, "article-num-akn").text.strip()
                    commas = self.driver.find_elements(By.CLASS_NAME, "art-comma-div-akn")                    
                except:
                    continue
                
                print("Comma len: ", len(commas))
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
                    #print(comma_content)
                    
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
            else:
                print("Out ", article.text)

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
            continue
        scraper.driver.get(url)
        articles = scraper.get_articles()
        data.append(articles)
        
        scraped_pages_set.add(f"{year}/{i}")
        write_to_file(ALL_LAWS_SCRAPED_JSON, json.dumps(list(scraped_pages_set)))
        
        df_all_laws = pd.DataFrame(data)
        save_df_to_csv(df_all_laws, ALL_LAWS_CSV)

df_all_laws.head()

df_laws = pd.read_csv(LAWS_CSV)
final_df = pd.concat([df_laws, df_cp, df_cpp], ignore_index=True)