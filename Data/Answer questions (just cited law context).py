import os
import re
import torch
import pandas as pd
from transformers import AutoTokenizer, pipeline, LlamaForCausalLM, BitsAndBytesConfig, AutoModelForCausalLM
import difflib

# Determine default paths based on OS
isLinux = True
default_linux_path = os.path.join(os.getcwd().replace("/Data", "/Documents/Downloaded")) if "/Data" in os.getcwd() else os.path.join(os.getcwd(), "Documents", "Downloaded")
default_windows_path = os.path.join(os.getcwd().replace("\\Data", "\\Documents\\Downloaded")) if "\\Data" in os.getcwd() else os.path.join(os.getcwd(), "Documents", "Downloaded")
default_path = default_linux_path if isLinux else default_windows_path

DEFAULT_SAVE_DIR = default_path.replace("/Downloaded", "/Generated") if isLinux else default_path.replace("\\Downloaded", "\\Generated")
LAWS_CSV = os.path.join(DEFAULT_SAVE_DIR, 'laws.csv')
QUIZ_CSV = os.path.join(DEFAULT_SAVE_DIR, 'quiz_merged.csv')
REF_CSV = os.path.join(DEFAULT_SAVE_DIR, 'references_merged.csv')

models = {
    "Meta-Llama 8B": {
        'model_name': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'context_window': 8000,
        'prompt_function': lambda system_prompt, user_prompt: f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        'model_load_function': lambda model_name, quant_bab=None: AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="cuda") if quant_bab else LlamaForCausalLM.from_pretrained(model_name, device_map="cuda")
    },
    #"Saul": {
    #    'model_name': 'Equall/Saul-7B-Instruct-v1',
    #    'context_window': 1024,
    #    'prompt_function': lambda system_prompt, user_prompt: f"\n{system_prompt}\n|<user>|\n{user_prompt}\n|<assistant>|\n\n",
    #    'model_load_function': lambda model_name, quant_bab = None: AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="cuda") if quant_bab else AutoAWQForCausalLM.from_pretrained(model_name, device_map="cuda")
    #},
    #"Falcon-7B": {
    #    'model_name': 'tiiuae/falcon-7b-instruct',
    #    'context_window': 512,
    #    'prompt_function': lambda system_prompt, user_prompt: f"User: {user_prompt}\nAssistant:{system_prompt}",
    #    'model_load_function': lambda model_name, quant_bab = None: AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="cuda") if quant_bab else AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda")
    #}
}

# Load CSV files
df_quiz = pd.read_csv(QUIZ_CSV)
df_ref = pd.read_csv(REF_CSV)

df_quiz = df_quiz

for model_name, model_data in models.items():
    model_id = model_data['model_name']
    context_window = model_data['context_window']
    prompt_function = model_data['prompt_function']
    load_model_function = model_data['model_load_function']

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

    cont = 0 # DELETE
    correct_count = 0
    attempted_questions = 0
    no_answer_found_count = 0
    no_references_count = 0
    batch_size = 16
    inputs = []
    questions_answers = []

    for index, row in df_quiz.iterrows():
        ref = df_ref[df_ref['quiz_id'] == row['quiz_id']]
        if ref.empty or pd.isna(ref.iloc[0]['law_text']):
            no_references_count += 1
            continue
        ref_text = ref.iloc[0]['law_text']

        input_text = prompt_function(
            """You are an expert in the field of law. Based on the following article, choose the correct answer to the question below:
            Article: {context}""".format(context=ref_text),
            """Question: {question}
            Options:
            1. {answer_1}
            2. {answer_2}
            3. {answer_3}
            Answer with the number of the correct answer (1, 2, or 3) in this format "La risposta corretta è (numero)".
            """.format(question=row['question'], answer_1=row['answer_1'], answer_2=row['answer_2'], answer_3=row['answer_3'])
        )

        inputs.append(input_text)
        questions_answers.append([row['answer_1'], row['answer_2'], row['answer_2']])

        # Process batch when the size reaches batch_size
        if len(inputs) == batch_size:
            outputs = nlp(inputs, max_new_tokens=50)

            for output, question_answers in zip(outputs, questions_answers):
                cont += 1
                ans = output[0]['generated_text'].strip()
                ans = ans.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
                print(ans)
                # Extract the model's chosen answer with regex
                matches = re.findall(r'risposta corretta\s*è\s*(\d+)', ans, re.DOTALL)

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
        outputs = nlp(inputs, max_new_tokens=64)
        for output, question_answers in zip(outputs, questions_answers):
            cont += 1
            
            ans = output[0]['generated_text'].strip()
            ans = ans.split("<|start_header_id|>assistant<|end_header_id|>")[-1]

            # Extract the model's chosen answer with regex
            matches = re.findall(r'risposta corretta\s*è\s*(\d+)', ans, re.DOTALL)

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
                
    print(f"Model answered {correct_count} / {attempted_questions} questions correctly: {correct_count / attempted_questions * 100:.2f}%")
    print(f"Questions without references: {no_references_count}")
    print(f"Questions without answers: {no_answer_found_count}")
    print(f"Total questions attempted: {attempted_questions}")
