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
        
def drop_everything():
    collections = utility.list_collections()

    for collection in collections:
        utility.drop_collection(collection)

def create_collection():
    laws_fields = [
        FieldSchema(name="law_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
        FieldSchema(name="law_source", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="law_year", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="law_number", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="law_text", dtype=DataType.VARCHAR, max_length=10000)
    ]
    schema = CollectionSchema(laws_fields, "laws collection", enable_dynamic_field=True)
    laws_collection = Collection(name="laws_collection", schema=schema)
    laws_collection.create_index("embedding", {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}})
    return laws_collection

def generate_embedding(text, tokenizer, model):
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        model_output = model(**encoded_input)
        embedding = model_output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding

def load_data_and_generate_embeddings(data, model, tokenizer):
    data["year"] = data["year"].astype(str)
    # law_source,year,law_number,law_text    
    
    embeddings = []
    for cc in tqdm(data["law_text"], total=data.shape[0]):
        if type(cc) != str:
            cc = ""
        embeddings.append(generate_embedding(cc, tokenizer, model))
    data["embedding"] = embeddings

    return data

def insert_data_into_milvus(collection, dataWithEmbeddings):
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
    
def search_similar_text(collection, query, tokenizer, model, top_k=3):
    query_embedding = generate_embedding(query, tokenizer, model)
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    
    results = collection.search(
        data=[query_embedding], 
        anns_field="embedding", 
        param=search_params, 
        limit=top_k, 
        output_fields=["law_source", "law_year", "law_number", "law_text"]
    )
    
    # Format the results
    formatted_results = []
    for result in results[0]:
        formatted_results.append({
            "score": result.score,
            "law_source": result.entity.get("law_source"),
            "law_year": result.entity.get("law_year"),
            "law_number": result.entity.get("law_number"),
            "law_text": result.entity.get("law_text")
        })
    
    return formatted_results

def generate_response(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    # Generate the response
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=1024,
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

device = torch.device(DEVICE)

bnb_config = BitsAndBytesConfig (
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
language_model = AutoModelForCausalLM.from_pretrained(LANGUAGE_MODEL, quantization_config=bnb_config, device_map="cuda")
language_tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_MODEL)

#embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL).to(device)
embedding_model = AutoModel.from_pretrained(TRAINED_MODEL_DIR).to(device)
embedding_tokenizer = AutoTokenizer.from_pretrained(TRAINED_TOKENIZER_DIR)#(LANGUAGE_MODEL)

dataWithEmbeddings = load_data_and_generate_embeddings(pd.read_csv(LAWS_CSV), embedding_model, embedding_tokenizer)
insert_data_into_milvus(laws_collection, dataWithEmbeddings)

# quiz_id,question,answer_1,answer_2,answer_3
quizzes_df = pd.read_csv(QUIZZES_CSV)[:10]

correct_answers = 0

for idx, row in quizzes_df.iterrows():
    print("Retrieving quiz question...")
    question = row["question"]
    answer_1 = row["answer_1"]
    answer_2 = row["answer_2"]
    answer_3 = row["answer_3"]

    print("Searching for similar text in the laws collection...")
    search_results = search_similar_text(laws_collection, question, embedding_tokenizer, embedding_model)
    context = "; ".join([result["law_text"] for result in search_results])

    print("Generating response using LLaMA...")
    system_prompt = f"""
    You are an expert in the field of law. Based on the following articles, choose the correct answer to the question below:
    Articles: {context}
    Question: {question}
    Options:
    1. {answer_1}
    2. {answer_2}
    3. {answer_3}
    Respond with the number of the correct answer (1, 2, or 3) in this format "La risposta corretta è (numero)".
    """

    # Generate a final response using the language model
    response = generate_response(system_prompt, language_model, language_tokenizer)
    response = response.strip()

    print(f"Question: {question}")
    print(f"Model's Response: {response.strip()}")

    # Extract the model's chosen answer with regex
    matches = re.findall(r'risposta corretta\s*è\s*(\d+)', response, re.DOTALL)

    if len(matches) > 0:
        model_answer = matches[0]
    else:
        print("No answer found.")
        continue  # Skip to the next iteration if no answer is found

    # Optionally, compare to the correct answer (if provided in the CSV)
    if model_answer == "1":
        print(f"Correct Answer")
        correct_answers += 1
    else:
        print("The model's answer was incorrect.")
    
    print("-" * 40)

print(f"Model answered {correct_answers} / {quizzes_df.shape[0]} questions correctly: {correct_answers / quizzes_df.shape[0] * 100:.2f}%")