import re
import os
import json
import pandas as pd
from tqdm import tqdm

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

df = pd.read_csv(LAWS)

def elaboration(content):
    if "<" not in content:
        return content
    
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

df['Comma content'] = df['Comma content'].apply(elaboration)

with open('output.txt', 'w') as f:
    for i in range(len(df)):
        f.write(str(df.iloc[i]["Comma content"]))
        f.write("\n--------------------\n")
    
df.to_csv(LAWS, index=False)
