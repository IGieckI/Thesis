import os
import time
from tqdm import tqdm
import random
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, LlamaForCausalLM
from awq import AutoAWQForCausalLM
import torch
from difflib import SequenceMatcher
from huggingface_hub import notebook_login, HfFolder

from milvus import default_server
from pymilvus import connections, Collection, DataType, FieldSchema, CollectionSchema, utility
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

isLinux = True
default_linux_path = os.path.join(os.getcwd().replace("/Data", "/Documents/Downloaded")) if "/Data" in os.getcwd() else os.path.join(os.getcwd(), "Documents", "Downloaded")
default_windows_path = os.path.join(os.getcwd().replace("\\Data", "\\Documents\\Downloaded")) if "\\Data" in os.getcwd() else os.path.join(os.getcwd(), "Documents", "Downloaded")
default_path = default_linux_path if isLinux else default_windows_path

DEFAULT_SAVE_DIR = default_path.replace("/Downloaded", "/Generated") if isLinux else default_path.replace("\\Downloaded", "\\Generated")
LAWS_CSV = os.path.join(DEFAULT_SAVE_DIR, 'laws.csv' if isLinux else 'laws.csv')

HfFolder.save_token("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")

models = {
    "Meta-Llama 8B": {
        'model_name': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'context_window': 8000,
        'prompt_function': lambda system_prompt, user_prompt: f"system{system_prompt}user{user_prompt}assistant"
    },
    #"Saul": {
    #    'model_name': 'Equall/Saul-7B-Instruct-v1',
    #    'context_window': 1024,
    #    'prompt_function': lambda system_prompt, user_prompt: f"\n{system_prompt}\n|<user>|\n{user_prompt}\n|<assistant>|\n\n"
    #},
    #"Falcon-7B": {
    #    'model_name': 'tiiuae/falcon-7b-instruct',
    #    'context_window': 512,
    #    'prompt_function': lambda system_prompt, user_prompt: f"User: {user_prompt}\nAssistant:{system_prompt}"
    #}
}

def connect_to_milvus():
    default_server.start()
    time.sleep(2)
    connections.connect("default", host="0.0.0.0", port="19530")
    
def get_similar_law(original_law):
    embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", model_kwargs={"device": "cuda"})
    law_metadata_db = Milvus(
            embedder,
            collection_name="law_metadata_db_500",
            auto_id=True,
        )
    result = law_metadata_db.similarity_search(original_law)[0]
    article_db = Milvus (
        embedder,
        collection_name="article_db_500",
        auto_id=True,
    )
    law_id = result.metadata["law_id"]
    article_result = article_db.as_retriever(search_kwargs={"expr": f'law_id == "{law_id}"'}).invoke(test_query)[0]
    print(article_result.page_content)
    

df_quiz_plh = pd.read_csv(os.path.join(DEFAULT_SAVE_DIR, 'quiz_merged_plh.csv'))
df_ref = pd.read_csv(os.path.join(DEFAULT_SAVE_DIR, 'references_merged.csv'))

df_quiz_plh = df_quiz_plh[:20]

for model_name, model_data in models.items():
    model_id, context_window, prompt_function = model_data['model_name'], model_data['context_window'], model_data['prompt_function']
    print(f'Running model: {model_name}')
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    model = LlamaForCausalLM.from_pretrained(model_id, load_in_4bit=True, device_map="cuda")
    
    nlp = pipeline("text-generation", model=model, tokenizer=tokenizer)

    correct_count = 0
    attempted_questions = 0
    for index, row in df_quiz_plh.iterrows():
        ref = df_ref[df_ref['Question id'] == row['Index']]
        if ref.empty:
            continue
        
        referece = get_similar_law(ref.loc['Reference'])
        
        input_text = prompt_function(
            f"You are an expert in the field of law, and you are gonna reply to the following quiz. You have to choose the correct answer among the three options. Just use the question and the answers as context. This is the referenced article in the question: "
            + f"n. {ref.iloc[0]['Reference']} from {ref.iloc[0]['Source']}{', ' + str(ref.iloc[0]['Comma']) if ref.iloc[0]['Comma'] != None else ''}",
            row['Question'] + row['Answer 1'] + row['Answer 2'] + row['Answer 3']
        )

        outputs = nlp(input_text, max_new_tokens=1000)
        ans = outputs[0]["generated_text"]

        answers = [row['Answer 1'], row['Answer 2'], row['Answer 3']]
        similarities = [SequenceMatcher(None, ans, a).ratio() for a in answers if a]

        if similarities:
            most_similar_answer = answers[similarities.index(max(similarities))]
            if most_similar_answer == row['Answer 1']:
                correct_count += 1

            attempted_questions += 1

    if attempted_questions > 0:
        accuracy = correct_count / attempted_questions
        print(f'Accuracy of {model_name}: {accuracy}')
    else:
        print(f'No questions were attempted for model: {model_name}')
