import os
from tqdm import tqdm
import random
import pandas as pd
from transformers import AutoTokenizer, AutoModel, pipeline, AutoModelForCausalLM
from awq import AutoAWQForCausalLM
import torch
from difflib import SequenceMatcher
from huggingface_hub import notebook_login, HfFolder

from pymilvus import connections, Collection, DataType, FieldSchema, CollectionSchema, utility
from milvus import default_server
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


isLinux = True
default_linux_path = os.getcwd().replace("/Data", "/Documents/Downloaded") if "/Data" in os.getcwd() else os.getcwd() + "/Documents/Downloaded"
default_windows_path = os.getcwd().replace("\\Data", "\\Documents\\Downloaded") if "\\Data" in os.getcwd() else os.getcwd() + "\\Documents\\Downloaded"
default_path = default_linux_path if isLinux else default_windows_path

DEFAULT_SAVE_DIR = default_path.replace("/Downloaded", "/Generated") if isLinux else default_path.replace("\\Downloaded", "\\Generated")
LAWS_CSV = DEFAULT_SAVE_DIR + ('/laws.csv' if isLinux else '\\laws.csv')

# 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
#notebook_login()
HfFolder.save_token("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")

def connect_to_milvus():
    try:
        connections.connect("default", host="0.0.0.0")
    except:
        default_server.start()
        
def drop_everything():
    collections = utility.list_collections()

    for collection in collections:
        utility.drop_collection(collection)

def create_collection():
    laws_fields = [
        FieldSchema(name="law_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=4096),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="article", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="comma", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="comma_content", dtype=DataType.VARCHAR, max_length=5000)
    ] 

    schema = CollectionSchema(laws_fields, "laws collection")

    laws_collection = Collection(name="laws_collection", schema=schema)
    laws_collection.create_index("embedding", {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}})
    
    return laws_collection

def load_model(model_name):    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Normal model
    model = AutoModel.from_pretrained(model_name)
    
    # Quantized model v1
    #quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }
    #model = AutoAWQForCausalLM.from_pretrained(model_name, **{"low_cpu_mem_usage": True}, device_map="cuda",  trust_remote_code = True)
    #model.quantize(tokenizer, quant_config=quant_config)
    
    # Quantized model v2
    #model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map="auto")
        
    return model, tokenizer

def generate_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    embedding_tensor = outputs.last_hidden_state.mean(dim=1).squeeze()
    embedding_list = embedding_tensor.tolist()
    
    return embedding_list

def load_data_and_generate_embeddings(data, model, tokenizer):
    data = data[:3]

    embeddings = []
    for cc in tqdm(data["Comma content"], total=data.shape[0]):
        embeddings.append(generate_embedding(cc, tokenizer, model))
    data["Embedding"] = embeddings

    return data

def insert_data_into_milvus(collection, dataWithEmbeddings):
    source_list = dataWithEmbeddings["Source"].tolist()
    article_list = dataWithEmbeddings["Article"].tolist()
    comma_list = dataWithEmbeddings["Comma number"].tolist()
    comma_content_list = dataWithEmbeddings["Comma content"].tolist()
    embedding_list = dataWithEmbeddings["Embedding"].tolist()
        
    data = []
    for i in range(len(embedding_list)):
        data.append({
            "embedding": embedding_list[i],     # Embedding (FLOAT_VECTOR)
            "source": source_list[i],           # Source (VARCHAR)
            "article": article_list[i],         # Article (VARCHAR)
            "comma": comma_list[i],             # Comma (VARCHAR)
            "comma_content": comma_content_list[i]  # Comma content (VARCHAR)
        })
    
    collection.insert(data)
    
    collection.flush()
    collection.load()
    
def search_similar_text(collection, text, tokenizer, model, top_k=5):
    # Generate the embedding for the input text
    embedding = generate_embedding(text, tokenizer, model)
    
    # Perform a search on the collection
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search([embedding], "embedding", search_params, limit=top_k, output_fields=["source", "article", "comma", "comma_content"])
    
    # Format the results
    formatted_results = []
    for result in results[0]:
        formatted_results.append({
            "score": result.score,
            "source": result.entity.get("source"),
            "article": result.entity.get("article"),
            "comma": result.entity.get("comma"),
            "comma_content": result.entity.get("comma_content")
        })
    
    return formatted_results

def generate_response(prompt, model, tokenizer, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate the response
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=150,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7,
            top_p=0.9
        )
    
    # Decode the generated tokens
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

connect_to_milvus()
drop_everything() # !!! WARNING !!!
laws_collection = create_collection()

model, tokenizer = load_model("meta-llama/Meta-Llama-3.1-8B-Instruct")#("BAAI/bge-m3")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model.to(device)
#model.eval()

dataWithEmbeddings = load_data_and_generate_embeddings(pd.read_csv(LAWS_CSV), model, tokenizer)
insert_data_into_milvus(laws_collection, dataWithEmbeddings)

while True:
    user_prompt = "Citami un articolo"#input("Insert a prompt: ")
    
    search_results = search_similar_text(laws_collection, user_prompt, tokenizer, model)
    print(search_results)

    # Combine retrieved documents into a single context
    context = ";".join([result["comma_content"] for result in search_results])
    system_prompt = "You are an expert in the field of law, and you are gonna replay to the following quiz. You have to choose the correct answer among the three options. These are some articles that could help you: " + context

    # Generate a final response using LLaMA 3 (optional, based on your needs)
    response_input = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    response = generate_response(response_input, model, tokenizer, device)

    print(response)