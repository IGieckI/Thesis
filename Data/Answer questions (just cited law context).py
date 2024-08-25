import os
import torch
import random
import pandas as pd
from tqdm import tqdm
from awq import AutoAWQForCausalLM
from difflib import SequenceMatcher
from huggingface_hub import notebook_login, HfFolder
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, LlamaForCausalLM, AutoModel, BitsAndBytesConfig

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from pymilvus import connections, Collection, DataType, FieldSchema, CollectionSchema, utility

isLinux = True
default_linux_path = os.path.join(os.getcwd().replace("/Data", "/Documents/Downloaded")) if "/Data" in os.getcwd() else os.path.join(os.getcwd(), "Documents", "Downloaded")
default_windows_path = os.path.join(os.getcwd().replace("\\Data", "\\Documents\\Downloaded")) if "\\Data" in os.getcwd() else os.path.join(os.getcwd(), "Documents", "Downloaded")
default_path = default_linux_path if isLinux else default_windows_path

DEFAULT_SAVE_DIR = default_path.replace("/Downloaded", "/Generated") if isLinux else default_path.replace("\\Downloaded", "\\Generated")
LAWS_CSV = os.path.join(DEFAULT_SAVE_DIR, 'laws.csv' if isLinux else 'laws.csv')
QUIZ_CSV = os.path.join(DEFAULT_SAVE_DIR, 'quiz_merged.csv' if isLinux else 'quiz_merged.csv')
REF_CSV = os.path.join(DEFAULT_SAVE_DIR, 'references_merged.csv' if isLinux else 'references_merged.csv')

HfFolder.save_token("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")

models = {
    "Meta-Llama 8B": {
        'model_name': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'context_window': 8000,
        'prompt_function': lambda system_prompt, user_prompt: f"system{system_prompt}user{user_prompt}assistant",
        'model_load_function': lambda model_name, quant_bab = None: AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="cuda") if quant_bab else LlamaForCausalLM.from_pretrained(model_name, device_map="cuda")
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

df_quiz = pd.read_csv(QUIZ_CSV)
df_ref = pd.read_csv(REF_CSV)

df_quiz = df_quiz[:300]

for model_name, model_data in models.items():
    model_id, context_window, prompt_function, load_model_function = model_data['model_name'], model_data['context_window'], model_data['prompt_function'], model_data['model_load_function']
    print(f'Running model: {model_name}')
    bnb_config = BitsAndBytesConfig(
                                    load_in_4bit=True,
                                    bnb_4bit_use_double_quant=True,
                                    bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=torch.bfloat16,
                                )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = load_model_function(model_id, bnb_config)
    
    nlp = pipeline("text-generation", model=model, tokenizer=tokenizer)

    correct_count = 0
    attempted_questions = 0
    batch_size = 8
    inputs = []
    refs = []
    for index, row in df_quiz.iterrows():
        ref = df_ref[df_ref['Question id'] == row['Index']]
        if ref.empty:
            continue
        ref = ref.iloc[0]
        
        input_text = prompt_function(
            f"You are an expert in the field of law, and you are gonna reply to the following quiz. You have to choose the correct answer among the three options. Just use the question and the answers as context. This is the referenced article in the question: "
            + f"{ref['Reference']}",
            row['Question'] + row['Answer 1'] + row['Answer 2'] + row['Answer 3']
        )

        inputs.append(input_text)
        refs.append(row['Answer 1'])  # Store correct answer to compare later

        # Process batch when the size reaches batch_size
        if len(inputs) == batch_size:
            outputs = nlp(inputs, max_new_tokens=1000)

            for output, correct_answer in zip(outputs, refs):
                ans = output["generated_text"]

                answers = [row['Answer 1'], row['Answer 2'], row['Answer 3']]
                similarities = [SequenceMatcher(None, ans, a).ratio() for a in answers if a]

                if similarities:
                    most_similar_answer = answers[similarities.index(max(similarities))]
                    if most_similar_answer == correct_answer:
                        correct_count += 1

                    attempted_questions += 1

            # Clear batch
            inputs = []
            refs = []

    # If there are any remaining inputs after the loop, process them
    if inputs:
        outputs = nlp(inputs, max_new_tokens=1000)
        for output, correct_answer in zip(outputs, refs):
            ans = output["generated_text"]

            answers = [row['Answer 1'], row['Answer 2'], row['Answer 3']]
            similarities = [SequenceMatcher(None, ans, a).ratio() for a in answers if a]

            if similarities:
                most_similar_answer = answers[similarities.index(max(similarities))]
                if most_similar_answer == correct_answer:
                    correct_count += 1

                attempted_questions += 1

    if attempted_questions > 0:
        accuracy = correct_count / attempted_questions
        print(f'Accuracy of {model_name}: {accuracy}')
    else:
        print(f'No questions were attempted for model: {model_name}')