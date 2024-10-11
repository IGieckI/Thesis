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
TRAINED_MODEL_DIR = default_path.replace("/Downloaded", "/TrainedModelFrozenFirst") if isLinux else default_path.replace("\\Downloaded", "\\TrainedModel")
TRAINED_TOKENIZER_DIR = default_path.replace("/Downloaded", "/TrainedTokenizerFrozenFirst") if isLinux else default_path.replace("\\Downloaded", "\\TrainedTokenizer")
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

def generate_response(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    # Generate the response
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=64, # The model should answer in a sentece or two
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7,
            top_p=0.9
        )
    
    # Decode the generated tokens
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

connect_to_milvus()
laws_collection = Collection(name="laws_collection")
laws_collection.load()

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

#dataWithEmbeddings = load_data_and_generate_embeddings(pd.read_csv(LAWS_CSV), embedding_model, embedding_tokenizer)
#insert_data_into_milvus(laws_collection, dataWithEmbeddings)

# quiz_id,question,answer_1,answer_2,answer_3
quizzes_df = pd.read_csv(QUIZZES_CSV)

# filter quizzes that have certain words in the question
#quizzes_df = quizzes_df[quizzes_df["question"].str.contains("c.p.|codice penale|Codice Penale|Codice penale|C.p.a.|c.p.a.|Codice del Processo Amministrativo|C.p.p.|c.p.p.|Codice di Procedura Penale|Cost|cost|decreto legislativo|D.Lgs.", case=False)]
print(len(quizzes_df))
question_answered = 0
correct_answers = 0

for idx, row in quizzes_df.iterrows():
    print("Retrieving quiz question...")
    question = row["question"]
    answer_1 = row["answer_1"]
    answer_2 = row["answer_2"]
    answer_3 = row["answer_3"]

    print("Searching for similar text in the laws collection...")
    search_results = search_similar_text(laws_collection, question, embedding_tokenizer, embedding_model)
    context = "; ".join([result["article_text"] for result in search_results])

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

    question_answered += 1
    # Optionally, compare to the correct answer (if provided in the CSV)
    if model_answer == "1":
        print(f"Correct Answer")
        correct_answers += 1
    else:
        print("The model's answer was incorrect.")
    
    print("-" * 40)

print(f"Model answered {correct_answers} / {question_answered} questions correctly: {correct_answers / question_answered * 100:.2f}%")