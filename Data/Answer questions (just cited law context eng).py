import os
import re
import torch
import pandas as pd
from transformers import AutoTokenizer, pipeline, LlamaForCausalLM, BitsAndBytesConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, T5Tokenizer, T5ForConditionalGeneration, AutoModel, MT5ForConditionalGeneration
import difflib

# Determine default paths based on OS
isLinux = True
default_linux_path = os.path.join(os.getcwd().replace("/Data", "/Documents/Downloaded")) if "/Data" in os.getcwd() else os.path.join(os.getcwd(), "Documents", "Downloaded")
default_windows_path = os.path.join(os.getcwd().replace("\\Data", "\\Documents\\Downloaded")) if "\\Data" in os.getcwd() else os.path.join(os.getcwd(), "Documents", "Downloaded")
default_path = default_linux_path if isLinux else default_windows_path

DEFAULT_SAVE_DIR = default_path.replace("/Downloaded", "/Generated") if isLinux else default_path.replace("\\Downloaded", "\\Generated")
LAWS_CSV = os.path.join(DEFAULT_SAVE_DIR, 'laws.csv')
QUIZ_ENG_CSV = os.path.join(DEFAULT_SAVE_DIR, 'quiz_merged_eng.csv')
REF_CSV = os.path.join(DEFAULT_SAVE_DIR, 'references_merged.csv')

"""
Lista modelli piccoli:
google/mt5-large
google/mt5-base
facebook/mbart-large-50
gsarti/it5-large
gsarti/it5-base
"""

models = {
    # Large Models
    "Meta-Llama 3.1 8B": {
        'model_name': 'meta-llama/Llama-3.1-8B-Instruct',
        'context_window': 8000,
        'prompt_function': lambda system_prompt, user_prompt: f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
    {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        'model_load_function': lambda model_name, quant_bab=None: AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="cuda") if quant_bab else LlamaForCausalLM.from_pretrained(model_name, device_map="cuda"),
        'tokenizer_load_function': lambda model_name: AutoTokenizer.from_pretrained(model_name),
        'text_generation': lambda model, tokenizer, text: [output[0]['generated_text'].strip() for output in pipeline("text-generation", model=model, tokenizer=tokenizer)(text, max_new_tokens=64)]
    },
    #"Saul 7B": {
    #    'model_name': 'Equall/Saul-7B-Instruct-v1',
    #    'context_window': 1024,
    #    'prompt_function': lambda system_prompt, user_prompt: f"\n{system_prompt}\n|<user>|\n{user_prompt}\n|<assistant>|\n\n",
    #    'model_load_function': lambda model_name, quant_bab=None: AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="cuda") if quant_bab else AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda"),
    #    'tokenizer_load_function': lambda model_name: AutoTokenizer.from_pretrained(model_name),
    #    'text_generation': lambda model, tokenizer, text: [output[0]['generated_text'].strip() for output in pipeline("text-generation", model=model, tokenizer=tokenizer)(text, max_new_tokens=64)]
    #},
    #"Mistral 7B Instruct": {
    #    'model_name': 'mistralai/Mistral-7B-Instruct-v0.1',
    #    'context_window': 8192,
    #    'prompt_function': lambda system_prompt, user_prompt: f"{system_prompt}\nUser: {user_prompt}\nAssistant: ",
    #    'model_load_function': lambda model_name, quant_bab=None: AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="cuda") if quant_bab else AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda"),
    #    'tokenizer_load_function': lambda model_name: AutoTokenizer.from_pretrained(model_name),
    #    'text_generation': lambda model, tokenizer, text: [output[0]['generated_text'].strip() for output in pipeline("text-generation", model=model, tokenizer=tokenizer)(text, max_new_tokens=64)]
    #},
    #"Phi-3 8k": {
    #    'model_name': 'microsoft/Phi-3-small-8k-instruct',
    #    'context_window': 8192,
    #    'prompt_function': lambda system_prompt, user_prompt: f"<|endoftext|><|system|>{system_prompt}<|end|><|user|>{user_prompt}<|end|><|assistant|>",
    #    'model_load_function': lambda model_name, quant_bab=None: AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="cuda", trust_remote_code=True) if quant_bab else AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", trust_remote_code=True),
    #    'tokenizer_load_function': lambda model_name: AutoTokenizer.from_pretrained(model_name),
    #    'text_generation': lambda model, tokenizer, text: [output[0]['generated_text'].strip() for output in pipeline("text-generation", model=model, tokenizer=tokenizer)(text, max_new_tokens=64)]
    #},
    #"Gemma-2 9B": {
    #    'model_name': 'google/gemma-2-9b-it',
    #    'context_window': 8192,
    #    'prompt_function': lambda system_prompt, user_prompt: f"<start_of_turn>user\n{system_prompt}{user_prompt}<end_of_turn>\n<start_of_turn>model\n",
    #    'model_load_function': lambda model_name, quant_bab=None: AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="cuda") if quant_bab else AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda"),
    #    'tokenizer_load_function': lambda model_name: AutoTokenizer.from_pretrained(model_name),
    #    'text_generation': lambda model, tokenizer, text: generate_text_default_transformers(model, tokenizer, text)
    #},
    
    # Medium Models
    #"Google mt5-base": {
    #    'model_name': 'google/mt5-base', # No prompt
    #    'context_window': 1024,
    #    'prompt_function': lambda system_prompt, user_prompt: f"\n{system_prompt}\n|<user>|\n{user_prompt}\n|<assistant>|\n\n",
    #    'model_load_function': lambda model_name, quant_bab=None: MT5ForConditionalGeneration.from_pretrained(model_name, device_map="cuda"),#T5ForConditionalGeneration.from_pretrained(model_name, device_map="cuda"),
    #    'tokenizer_load_function': lambda model_name: AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir="../llms"),#T5Tokenizer.from_pretrained(model_name),
    #    'text_generation': lambda model, tokenizer, text: generate_text_default_transformers(model, tokenizer, text)
    #},
    #"Google mt5-large": {
    #    'model_name': 'google/mt5-large', # No prompt
    #    'context_window': 1024,
    #    'prompt_function': lambda system_prompt, user_prompt: f"\n{system_prompt}\n|<user>|\n{user_prompt}\n|<assistant>|\n\n",
    #    'model_load_function': lambda model_name, quant_bab=None: AutoModel.from_pretrained(model_name, device_map="cuda"),#T5ForConditionalGeneration.from_pretrained(model_name, device_map="cuda"),
    #    'tokenizer_load_function': lambda model_name: AutoTokenizer.from_pretrained(model_name, cache_dir="../llms"),#T5Tokenizer.from_pretrained(model_name),
    #    'text_generation': lambda model, tokenizer, text: generate_text_googlemt_model(model, tokenizer, text)
    #},
    #"it5-large": {
    #    'model_name': 'gsarti/it5-large',
    #    'context_window': 1024,
    #    'prompt_function': lambda system_prompt, user_prompt: f"\n{system_prompt}\n|<user>|\n{user_prompt}\n|<assistant>|\n\n",
    #    'model_load_function': lambda model_name, quant_bab=None: AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="cuda"),
    #    'tokenizer_load_function': lambda model_name: AutoTokenizer.from_pretrained(model_name),
    #    'text_generation': lambda model, tokenizer, text: [output[0]['generated_text'].strip() for output in pipeline("text-generation", model=model, tokenizer=tokenizer)(text, max_new_tokens=64)]
    #},
    #"it5-base": {
    #    'model_name': 'gsarti/it5-base',
    #    'context_window': 1024,
    #    'prompt_function': lambda system_prompt, user_prompt: f"\n{system_prompt}\n|<user>|\n{user_prompt}\n|<assistant>|\n\n",
    #    'model_load_function': lambda model_name, quant_bab=None: AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="cuda"),
    #    'tokenizer_load_function': lambda model_name: AutoTokenizer.from_pretrained(model_name),
    #    'text_generation': lambda model, tokenizer, text: [output[0]['generated_text'].strip() for output in pipeline("text-generation", model=model, tokenizer=tokenizer)(text, max_new_tokens=64)]
    #},
    #"mbart-large-50": {
    #    'model_name': 'facebook/mbart-large-50', # No prompt
    #    'context_window': 1024,
    #    'prompt_function': lambda system_prompt, user_prompt: f"\n{system_prompt}\n|<user>|\n{user_prompt}\n|<assistant>|\n\n",
    #    'model_load_function': lambda model_name, quant_bab=None: AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="cuda"),
    #    'tokenizer_load_function': lambda model_name: AutoTokenizer.from_pretrained(model_name),
    #    'text_generation': lambda model, tokenizer, text: [output[0]['generated_text'].strip() for output in pipeline("text-generation", model=model, tokenizer=tokenizer)(text, max_new_tokens=64)]
    #}
}

def generate_text_default_transformers(model, tokenizer, text):
    if not type(text) == list:
        text = [text]
    
    outputs = []    
    nlp = pipeline("text-generation", model=model, tokenizer=tokenizer)
    outputs = nlp(text, max_new_tokens=64)

    for i, it in enumerate(outputs):
        outputs[i] = outputs[i][0]['generated_text']
        outputs[i] = extract_reply(outputs[i])

    return outputs

def generate_text_googlemt_model(model, tokenizer, text):
    if not isinstance(text, list):
        text = [text]
    
    outputs = []
    
    model.eval()
    
    device = next(model.parameters()).device

    for t in text:
        inputs = tokenizer(t, return_tensors='pt', padding=True, truncation=True).to(device)

        input_ids = inputs['input_ids']
        
        decoder_input_ids = tokenizer.encode(tokenizer.eos_token, return_tensors='pt').to(device)  # Move to the same device
        
        with torch.no_grad():
            output = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        print(dir(output))
        reply = extract_reply(output)

        outputs.append(reply)
    
    return outputs

def extract_reply(ans):
    if "<start_of_turn>model\n" in ans:
        ans = ans.split("<start_of_turn>model\n")[-1]
    elif "<|start_header_id|>assistant<|end_header_id|>" in ans:
        ans = ans.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
    else:
        ans = ans.split("|<assistant>|")[-1]
    ans = ans.replace("\n", "")
    ans = ans.strip()

    return ans

# Load CSV files
df_quiz_eng = pd.read_csv(QUIZ_ENG_CSV)
df_ref = pd.read_csv(REF_CSV)

for model_name, model_data in models.items():
    model_id = model_data['model_name']
    context_window = model_data['context_window']
    prompt_function = model_data['prompt_function']
    load_model_function = model_data['model_load_function']
    load_tokenizer_function = model_data['tokenizer_load_function']
    text_generation = model_data['text_generation']

    print(f'Running model: {model_name}')
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = load_tokenizer_function(model_id)
    model = load_model_function(model_id, bnb_config)

    correct_count = 0
    attempted_questions = 0
    no_answer_found_count = 0
    no_references_count = 0
    batch_size = 16
    inputs = []
    questions_answers = []

    for index, row in df_quiz_eng.iterrows():
        ref = df_ref[df_ref['quiz_id'] == row['quiz_id']]
        if ref.empty or pd.isna(ref.iloc[0]['article_text']):
            no_references_count += 1
            continue
        ref_text = ref.iloc[0]['article_text']

        input_text = prompt_function(
            f"""You are an expert in the field of law. Based on the following article, choose the correct answer to the question below:
            Article: {ref_text}""",
            f"""Question: {row['question']}
            Options:
            1. {row['answer_1']}
            2. {row['answer_2']}
            3. {row['answer_3']}
            Answer with the number of the correct answer (1, 2, or 3) in this format "The correct answer is (number)".
            """
        )

        inputs.append(input_text)
        questions_answers.append([row['answer_1'], row['answer_2'], row['answer_2']])

        # Process batch when the size reaches batch_size
        if len(inputs) == batch_size:
            outputs = text_generation(model, tokenizer, inputs)
            for output, question_answers in zip(outputs, questions_answers):
                output = extract_reply(output)
                
                # Extract the model's chosen answer with regex
                matches = re.findall(r'correct answer is.{0,3}(\d+)', output, re.DOTALL)

                if len(matches) > 0:
                    model_answer = matches[0]
                else:
                    print("No answer found.")
                    no_answer_found_count += 1
                    continue
                
                attempted_questions += 1

                if model_answer == "1":
                    print(f"Correct Answer")
                    correct_count += 1
                else:
                    print("The model's answer was incorrect.")
                
                print("-" * 40)

            # Clear batch
            inputs = []
            questions_answers = []

    # Process any remaining inputs
    if inputs:
        outputs = text_generation(model, tokenizer, inputs)
        for output, question_answers in zip(outputs, questions_answers):
            output = extract_reply(output)
            
            # Extract the model's chosen answer with regex
            matches = re.findall(r'correct answer is.{0,3}(\d+)', output, re.DOTALL)

            if len(matches) > 0:
                model_answer = matches[0]
            else:
                no_answer_found_count += 1
                print("No answer found.")
                continue

            attempted_questions += 1
                
            if model_answer == "1":
                print(f"Correct Answer")
                correct_count += 1
            else:
                print("The model's answer was incorrect.")
            
            print("-" * 40)
    
    print(f"Model {model_name} answered {correct_count} / {attempted_questions} questions correctly: {correct_count / attempted_questions * 100:.2f}%")
    print(f"Questions without references: {no_references_count}")
    print(f"Questions without answers: {no_answer_found_count}")
    print(f"Total questions attempted: {attempted_questions}")

    # Print the prints on a file too
    with open('output.txt', 'a') as fle:
        fle.write(f"Model {model_name} answered {correct_count} / {attempted_questions} questions correctly: {correct_count / attempted_questions * 100:.2f}%\n")
        fle.write(f"Questions without references: {no_references_count}\n")
        fle.write(f"Questions without answers: {no_answer_found_count}\n")
        fle.write(f"Total questions attempted: {attempted_questions}\n")
        fle.write("------------------------\n")
