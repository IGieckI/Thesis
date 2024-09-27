import re
import os
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import HfFolder
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig, AutoModelForCausalLM

from milvus import default_server
from pymilvus import connections, Collection, DataType, FieldSchema, CollectionSchema, utility

isLinux = True
default_linux_path = os.getcwd().replace("/Data", "/Documents/Downloaded") if "/Data" in os.getcwd() else os.getcwd() + "/Documents/Downloaded"
default_windows_path = os.getcwd().replace("\\Data", "\\Documents\\Downloaded") if "\\Data" in os.getcwd() else os.getcwd() + "\\Documents\\Downloaded"
default_path = default_linux_path if isLinux else default_windows_path

DEFAULT_SAVE_DIR = default_path.replace("/Downloaded", "/Generated") if isLinux else default_path.replace("\\Downloaded", "\\Generated")
LAWS_CSV = DEFAULT_SAVE_DIR + ('/laws.csv' if isLinux else '\\laws.csv')
QUIZZES_CSV = DEFAULT_SAVE_DIR + ('/quiz_merged.csv' if isLinux else '\\quiz_merged.csv')

LANGUAGE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
TRAINED_MODEL_DIR = default_path.replace("/Downloaded", "/TrainedModel") if isLinux else default_path.replace("\\Downloaded", "\\TrainedModel")
TRAINED_TOKENIZER_DIR = default_path.replace("/Downloaded", "/TrainedTokenizer") if isLinux else default_path.replace("\\Downloaded", "\\TrainedTokenizer")
EMBEDDING_TOKENIZER = "BAAI/bge-m3"

GENERAL_LAWS_COLLECTION = "laws_collection"
REGIONAL_LAWS_DATASET = "paoloitaliani/regional_laws"
REGIONAL_LAWS_COLLECTION = "regional_laws_collection"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)

# 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
#notebook_login()
HfFolder.save_token("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")

def connect_to_milvus():
    try:
        connections.connect("default", host="0.0.0.0")
    except:
        default_server.start()
        connections.connect("default", host="0.0.0.0")
    
    # Check if Milvus is connected
    print("CONNECTION:", connections.list_connections())    
        
def drop_everything():
    collections = utility.list_collections()

    for collection in collections:
        if collection == GENERAL_LAWS_COLLECTION or collection == REGIONAL_LAWS_COLLECTION:
            utility.drop_collection(collection)

def create_general_laws_collection():
    laws_fields = [
        FieldSchema(name="law_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
        FieldSchema(name="law_source", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="law_year", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="law_number", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="law_text", dtype=DataType.VARCHAR, max_length=60000)
    ]
    schema = CollectionSchema(laws_fields, "laws collection", enable_dynamic_field=True)
    laws_collection = Collection(name=GENERAL_LAWS_COLLECTION, schema=schema)
    laws_collection.create_index("embedding", {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}})
    return laws_collection

def create_regional_laws_collection():
    laws_fields = [
        FieldSchema(name="law_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
        FieldSchema(name="law_region", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="law_year", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="law_number", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="law_text", dtype=DataType.VARCHAR, max_length=60000)
    ]
    schema = CollectionSchema(laws_fields, "regional laws collection", enable_dynamic_field=True)
    regional_laws_collection = Collection(name=REGIONAL_LAWS_COLLECTION, schema=schema)
    regional_laws_collection.create_index("embedding", {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}})
    return regional_laws_collection

def generate_laws_embeddings(data, model, tokenizer):
    data["year"] = data["year"].astype(str)
    
    embeddings = []
    for cc in tqdm(data["law_text"], total=data.shape[0]):
        if type(cc) != str:
            cc = ""        
        encoded_input = tokenizer(cc, padding=True, truncation=True, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            model_output = model(**encoded_input)
            embedding = model_output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()            
        embeddings.append(embedding)        
    data["embedding"] = embeddings
    return data

def insert_general_laws_into_milvus(collection, dataWithEmbeddings):
    embedding_list = dataWithEmbeddings["embedding"].tolist()
    source_list = dataWithEmbeddings["law_source"].tolist()
    year_list = dataWithEmbeddings["year"].tolist()
    number_list = dataWithEmbeddings["law_number"].tolist()
    text_list = dataWithEmbeddings["law_text"].tolist()
    
    data = []
    for i in range(len(embedding_list)):        
        if type(source_list[i]) != str:
            print(f"SOURCE ERROR --> Type: {type(source_list[i])} CONTENT: {source_list[i]}")
        if type(year_list[i]) != str:
            print(f"YEAR ERROR --> Type: {type(year_list[i])} CONTENT: {year_list[i]}")
        if type(number_list[i]) != str:
            print(f"NUMBER ERROR --> Type: {type(number_list[i])} CONTENT: {number_list[i]}")
        if type(text_list[i]) != str:
            print(f"TEXT ERROR --> Type: {type(text_list[i])} CONTENT: {text_list[i]}")
        
        data.append({
            "embedding": embedding_list[i],     # Embedding (FLOAT_VECTOR)
            "law_source": source_list[i],           # Source (VARCHAR)
            "law_year": year_list[i],         # Article (VARCHAR)
            "law_number": number_list[i],             # Comma (VARCHAR)
            "law_text": text_list[i]  # Comma content (VARCHAR)
        })
    #print(len(data[0]["embedding"]))
    collection.insert(data)
    
    collection.flush()
    collection.load()

def insert_regional_laws_into_milvus(collection, dataWithEmbeddings):
    print(dataWithEmbeddings.columns)
    
    embedding_list = dataWithEmbeddings["embedding"].tolist()
    source_list = dataWithEmbeddings["law_source"].tolist()
    year_list = dataWithEmbeddings["year"].tolist()
    number_list = dataWithEmbeddings["law_number"].tolist()
    text_list = dataWithEmbeddings["law_text"].tolist()
    
    data = []
    for i in range(len(embedding_list)):        
        if type(source_list[i]) != str:
            print(f"SOURCE ERROR --> Type: {type(source_list[i])} CONTENT: {source_list[i]}")
        if type(year_list[i]) != str:
            print(f"YEAR ERROR --> Type: {type(year_list[i])} CONTENT: {year_list[i]}")
        if type(number_list[i]) != str:
            print(f"NUMBER ERROR --> Type: {type(number_list[i])} CONTENT: {number_list[i]}")
        if type(text_list[i]) != str:
            print(f"TEXT ERROR --> Type: {type(text_list[i])} CONTENT: {text_list[i]}")
        
        data.append({
            "embedding": embedding_list[i],     # Embedding (FLOAT_VECTOR)
            "law_source": source_list[i],           # Source (VARCHAR)
            "law_year": year_list[i],         # Article (VARCHAR)
            "law_number": number_list[i],             # Comma (VARCHAR)
            "law_text": text_list[i]  # Comma content (VARCHAR)
        })
    #print(len(data[0]["embedding"]))
    collection.insert(data)
    
    collection.flush()
    collection.load()

device = torch.device(DEVICE)

#embedding_model = AutoModel.from_pretrained(EMBEDDING_TOKENIZER).to(device)
embedding_model = AutoModel.from_pretrained(TRAINED_MODEL_DIR).to(device)
embedding_tokenizer = AutoTokenizer.from_pretrained(TRAINED_TOKENIZER_DIR)#(LANGUAGE_MODEL)

connect_to_milvus()
drop_everything() # !!! WARNING !!!
general_laws_collection = create_general_laws_collection()
regional_laws_collection = create_regional_laws_collection()

df_general_laws = pd.read_csv(LAWS_CSV)
regional_laws = load_dataset("paoloitaliani/regional_laws")["train"].shuffle(seed=42).to_pandas()
df_regional_laws = pd.DataFrame()
df_regional_laws["law_source"] = regional_laws["region"]
df_regional_laws["year"] = regional_laws["year"]
df_regional_laws["law_number"] = regional_laws["law_num"]
df_regional_laws["law_text"] = regional_laws["articles"].apply(lambda x: " ".join(x))

insert_general_laws_into_milvus(general_laws_collection, generate_laws_embeddings(df_general_laws, embedding_model, embedding_tokenizer))
insert_regional_laws_into_milvus(regional_laws_collection, generate_laws_embeddings(df_regional_laws, embedding_model, embedding_tokenizer))