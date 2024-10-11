import re
import os
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig, AutoModelForCausalLM
import torch
from huggingface_hub import HfFolder

from pymilvus import connections, Collection, DataType, FieldSchema, CollectionSchema, utility
from milvus import default_server


isLinux = True
default_linux_path = os.getcwd().replace("/Data", "/Documents/Downloaded") if "/Data" in os.getcwd() else os.getcwd() + "/Documents/Downloaded"
default_windows_path = os.getcwd().replace("\\Data", "\\Documents\\Downloaded") if "\\Data" in os.getcwd() else os.getcwd() + "\\Documents\\Downloaded"
default_path = default_linux_path if isLinux else default_windows_path

DEFAULT_SAVE_DIR = default_path.replace("/Downloaded", "/Generated") if isLinux else default_path.replace("\\Downloaded", "\\Generated")
LAWS_CSV = DEFAULT_SAVE_DIR + ('/laws.csv' if isLinux else '\\laws.csv')
QUIZZES_CSV = DEFAULT_SAVE_DIR + ('/quiz_merged.csv' if isLinux else '\\quiz_merged.csv')

REFERENCES_CSV = DEFAULT_SAVE_DIR + '/references_merged.csv'

LANGUAGE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
TRAINED_MODEL_DIR = default_path.replace("/Downloaded", "/TrainedModel") if isLinux else default_path.replace("\\Downloaded", "\\TrainedModel")
TRAINED_TOKENIZER_DIR = default_path.replace("/Downloaded", "/TrainedTokenizer") if isLinux else default_path.replace("\\Downloaded", "\\TrainedTokenizer")
EMBEDDING_TOKENIZER = "BAAI/bge-m3"

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

def generate_embedding(text, tokenizer, model):
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        model_output = model(**encoded_input)
        embedding = model_output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding
    
def search_similar_text(collection, query, tokenizer, model, top_k=3):
    query_embedding = generate_embedding(query, tokenizer, model)
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    
    results = collection.search(
        data=[query_embedding], 
        anns_field="embedding", 
        param=search_params, 
        limit=top_k, 
        output_fields=["law_source", "law_year", "article_number", "article_text"]
    )
    
    # Format the results
    formatted_results = []
    for result in results[0]:
        formatted_results.append({
            "score": result.score,
            "law_source": result.entity.get("law_source"),
            "law_year": result.entity.get("law_year"),
            "article_number": result.entity.get("article_number"),
            "article_text": result.entity.get("article_text")
        })
    
    return formatted_results

connect_to_milvus()
laws_collection = Collection(name="laws_collection")
laws_collection.load()

device = torch.device(DEVICE)

#embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL).to(device)
embedding_model = AutoModel.from_pretrained(TRAINED_MODEL_DIR).to(device)
embedding_tokenizer = AutoTokenizer.from_pretrained(TRAINED_TOKENIZER_DIR)#(LANGUAGE_MODEL)

# quiz_id,question,answer_1,answer_2,answer_3
df_quizzes = pd.read_csv(QUIZZES_CSV)
df_references = pd.read_csv(REFERENCES_CSV)

# filter quizzes that have certain words in the question
#quizzes_df = quizzes_df[quizzes_df["question"].str.contains("c.p.|codice penale|Codice Penale|Codice penale|C.p.a.|c.p.a.|Codice del Processo Amministrativo|C.p.p.|c.p.p.|Codice di Procedura Penale|Cost|cost|decreto legislativo|D.Lgs.", case=False)]
num_retrieved = 0
num_correct_retrieved = 0

for i, row in df_references.iterrows():
    question = df_quizzes.loc[df_quizzes['quiz_id'] == row['quiz_id'], 'question']
    if question.empty:
        continue
    question = question.iloc[0]
    article = row['article_text']
    
    if pd.isna(article):
        continue
    
    retrieved_article = search_similar_text(laws_collection, question, embedding_tokenizer, embedding_model, 1)
    retrieved_article = retrieved_article[0]['article_text']
    
    if retrieved_article == article:
        num_correct_retrieved += 1
    num_retrieved += 1

print(f"Model retrieved correctly {num_correct_retrieved} / {num_retrieved} articles: {num_correct_retrieved / num_retrieved * 100:.2f}%")