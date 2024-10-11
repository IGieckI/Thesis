import re
import os
import grpc
import json
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
TRAINED_MODEL_DIR = default_path.replace("/Downloaded", "/TrainedModelFrozenFirst") if isLinux else default_path.replace("\\Downloaded", "\\TrainedModel")
TRAINED_TOKENIZER_DIR = default_path.replace("/Downloaded", "/TrainedTokenizerFrozenFirst") if isLinux else default_path.replace("\\Downloaded", "\\TrainedTokenizer")
EMBEDDING_TOKENIZER = "BAAI/bge-m3"

GENERAL_LAWS_COLLECTION = "laws_collection"
REGIONAL_LAWS_DATASET = "paoloitaliani/regional_laws"
REGIONAL_LAWS_COLLECTION = "regional_laws_collection"
REGIONAL_LAWS_CSV = DEFAULT_SAVE_DIR + ('/regional_laws.csv' if isLinux else '\\regional_laws.csv')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)

# 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
#notebook_login()
HfFolder.save_token("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")

def connect_to_milvus():
    try:
        connections.connect("default", host="0.0.0.0", port="19530", 
                            client_options=[
                                ("grpc.max_receive_message_length", 250 * 1024 * 1024),  # 200MB
                                ("grpc.max_send_message_length", 250 * 1024 * 1024),    # 200MB
                            ])
    except:
        default_server.start()
        connections.connect("default", host="0.0.0.0", 
                            client_options=[
                                ("grpc.max_receive_message_length", 250 * 1024 * 1024),  # 200MB
                                ("grpc.max_send_message_length", 250 * 1024 * 1024),    # 200MB
                            ])
    
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
        FieldSchema(name="year", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="article_number", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="article_text", dtype=DataType.VARCHAR, max_length=60000)
    ]
    schema = CollectionSchema(laws_fields, "laws collection", enable_dynamic_field=True)
    laws_collection = Collection(name=GENERAL_LAWS_COLLECTION, schema=schema)
    laws_collection.create_index("embedding", {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}})
    return laws_collection

def create_regional_laws_collection():
    laws_fields = [
        FieldSchema(name="law_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
        FieldSchema(name="region", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="year", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="article_number", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="article_text", dtype=DataType.VARCHAR, max_length=60000)
    ]
    schema = CollectionSchema(laws_fields, "regional laws collection", enable_dynamic_field=True)
    regional_laws_collection = Collection(name=REGIONAL_LAWS_COLLECTION, schema=schema)
    regional_laws_collection.create_index("embedding", {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}})
    return regional_laws_collection

def generate_laws_embeddings(data, model, tokenizer, max_text_size=60000, max_embedding_size=1024):
    embeddings = []
    skipped_indices = []
    
    for i, cc in tqdm(enumerate(data["article_text"]), total=data.shape[0]):
        if type(cc) != str:
            cc = ""
        
        if len(cc) > max_text_size:
            print(f"Skipping entity {i}: article text exceeds {max_text_size} characters.")
            skipped_indices.append(i)
            embeddings.append(None)
            continue

        encoded_input = tokenizer(cc, padding=True, truncation=True, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            model_output = model(**encoded_input)
            embedding = model_output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

            # Skip entities whose embedding exceeds max_embedding_size
            if len(embedding) > max_embedding_size:
                print(f"Skipping entity {i}: embedding exceeds {max_embedding_size} dimensions.")
                skipped_indices.append(i)
                embeddings.append(None)
                continue

        embeddings.append(embedding)
    
    data["embedding"] = embeddings
    # Remove rows with None embeddings (entities that were skipped)
    data = data[~data["embedding"].isnull()]
    
    return data


def insert_general_laws_into_milvus(collection, dataWithEmbeddings):
    embedding_list = dataWithEmbeddings["embedding"].tolist()
    source_list = dataWithEmbeddings["law_source"].tolist()
    year_list = dataWithEmbeddings["year"].tolist()
    number_list = dataWithEmbeddings["article_number"].tolist()
    text_list = dataWithEmbeddings["article_text"].tolist()
    
    data = []
    for i in range(len(embedding_list)):        
        if type(source_list[i]) != str:
            print(f"SOURCE ERROR --> Type: {type(source_list[i])} CONTENT: {source_list[i]}")
        if type(year_list[i]) != str:
            year_list[i] = ""
            print(f"YEAR ERROR --> Type: {type(year_list[i])} CONTENT: {year_list[i]}")
        if type(number_list[i]) != str:
            print(f"NUMBER ERROR --> Type: {type(number_list[i])} CONTENT: {number_list[i]}")
        if type(text_list[i]) != str:
            text_list[i] = ""
            print(f"TEXT ERROR --> Type: {type(text_list[i])} CONTENT: {text_list[i]}")
        
        data.append({
            "embedding": embedding_list[i],     # Embedding (FLOAT_VECTOR)
            "law_source": source_list[i],           # Source (VARCHAR)
            "year": year_list[i],         # Article (VARCHAR)
            "article_number": number_list[i],             # Comma (VARCHAR)
            "article_text": text_list[i]  # Comma content (VARCHAR)
        })
    #print(len(data[0]["embedding"]))
    collection.insert(data)
    
    collection.flush()
    collection.load()

def insert_regional_laws_into_milvus(collection, dataWithEmbeddings):    
    embedding_list = dataWithEmbeddings["embedding"].tolist()
    source_list = dataWithEmbeddings["region"].tolist()
    year_list = dataWithEmbeddings["year"].tolist()
    number_list = dataWithEmbeddings["article_number"].tolist()
    text_list = dataWithEmbeddings["article_text"].tolist()
    
    data = []
    for i in range(len(embedding_list)):        
        if type(source_list[i]) != str:
            print(f"SOURCE ERROR --> Type: {type(source_list[i])} CONTENT: {source_list[i]}")
        if type(year_list[i]) != str:
            year_list[i] = ""
            print(f"YEAR ERROR --> Type: {type(year_list[i])} CONTENT: {year_list[i]}")
        if type(number_list[i]) != str:
            print(f"NUMBER ERROR --> Type: {type(number_list[i])} CONTENT: {number_list[i]}")
        if type(text_list[i]) != str:
            text_list[i] = ""
            print(f"TEXT ERROR --> Type: {type(text_list[i])} CONTENT: {text_list[i]}")
        
        data.append({
            "embedding": embedding_list[i],     # Embedding (FLOAT_VECTOR)
            "region": source_list[i],           # Source (VARCHAR)
            "year": year_list[i],         # Article (VARCHAR)
            "article_number": number_list[i],             # Comma (VARCHAR)
            "article_text": text_list[i]  # Comma content (VARCHAR)
        })
    #print(len(data[0]["embedding"]))
    collection.insert(data)
    
    collection.flush()
    collection.load()
    
def createRegionalDataframe(regional_laws):
    df_regional_laws = pd.DataFrame()
    for i, row in regional_laws.iterrows():
        if len(row["articles"]) <= 2:
            continue

        if '{' not in row["articles"] or "': " not in row["articles"]:
            df_tmp = pd.DataFrame([{
                "region": row["region"], 
                "year": row["year"], 
                "law_num": row["law_num"], 
                "article_number": 1, 
                "article_text": row["articles"]
            }])
            df_regional_laws = pd.concat([df_regional_laws, df_tmp], ignore_index=True)
        else:
            articles = row["articles"].strip("{}")
            
            split_text = re.split(r"(\'\d+\':)", articles)
            split_text = [part.strip() for part in split_text if part.strip()]

            article_dict = {}
            
            for j in range(0, len(split_text), 2):
                if j+1 < len(split_text):
                    key = split_text[j].strip("':")
                    value = split_text[j+1].strip("'")
                    article_dict[key] = value

            for article_num, article_text in article_dict.items():
                df_tmp = pd.DataFrame([{
                    "region": row["region"], 
                    "year": row["year"], 
                    "law_num": row["law_num"], 
                    "article_number": article_num, 
                    "article_text": article_text
                }])
                df_regional_laws = pd.concat([df_regional_laws, df_tmp], ignore_index=True)
                print(df_tmp)
    df_regional_laws["year"] = df_regional_laws["year"].astype(str)
    
    return df_regional_laws

device = torch.device(DEVICE)

print("Loading model...")

#embedding_model = AutoModel.from_pretrained(EMBEDDING_TOKENIZER).to(device)
embedding_model = AutoModel.from_pretrained(TRAINED_MODEL_DIR).to(device)
embedding_tokenizer = AutoTokenizer.from_pretrained(TRAINED_TOKENIZER_DIR)#(LANGUAGE_MODEL)

print("Loading dataset...")

connect_to_milvus()
drop_everything() # !!! WARNING !!!
# Count rows for each collection
collections = utility.list_collections()

print("Creating collections...")

general_laws_collection = create_general_laws_collection()
regional_laws_collection = create_regional_laws_collection()

print("Loading general laws...")

df_general_laws = pd.read_csv(LAWS_CSV)[:12000]
df_general_laws["year"] = df_general_laws["year"].astype(str)

print("Loading regional laws...")
regional_laws = load_dataset("paoloitaliani/regional_laws")["train"].shuffle(seed=42).to_pandas()

if os.path.isfile(REGIONAL_LAWS_CSV):
    df_regional_laws = pd.read_csv(REGIONAL_LAWS_CSV)
else:
    df_regional_laws = createRegionalDataframe(regional_laws)
    df_regional_laws.to_csv(REGIONAL_LAWS_CSV, index=False)
df_regional_laws["year"] = df_regional_laws["year"].astype(str)

print("Generate embeddings and inserting into Milvus...")

#insert_regional_laws_into_milvus(regional_laws_collection, generate_laws_embeddings(df_regional_laws, embedding_model, embedding_tokenizer))
insert_general_laws_into_milvus(general_laws_collection, generate_laws_embeddings(df_general_laws, embedding_model, embedding_tokenizer))

for collection_name in collections:
    collection = Collection(collection_name)
    collection.load()
    print(f"Collection {collection_name} has {collection.num_entities} entities")