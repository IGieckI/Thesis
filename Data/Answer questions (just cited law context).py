import os
import torch
import pandas as pd
from transformers import AutoTokenizer, pipeline, LlamaForCausalLM, BitsAndBytesConfig
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
        'prompt_function': lambda system_prompt, user_prompt: f"system: {system_prompt}\nuser: {user_prompt}\nassistant:",
        'model_load_function': lambda model_name, quant_bab=None: LlamaForCausalLM.from_pretrained(model_name, device_map="cuda")
    }
}

# Load CSV files
df_quiz = pd.read_csv(QUIZ_CSV)
df_ref = pd.read_csv(REF_CSV)

df_quiz = df_quiz[:225]

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

    correct_count = 0
    attempted_questions = 0
    batch_size = 8
    inputs = []
    correct_answers = []

    for index, row in df_quiz.iterrows():
        ref = df_ref[df_ref['Question id'] == row['Index']]
        if ref.empty or pd.isna(ref.iloc[0]['Law text']):
            continue
        ref_text = ref.iloc[0]['Law text']

        input_text = prompt_function(
            "You are an expert in the field of law. Answer the following quiz. Choose the correct answer among the three options. This is the referenced article in the question: " + ref_text,
            row['Question'] + "\n" + row['Answer 1'] + "\n" + row['Answer 2'] + "\n" + row['Answer 3'] + "\nJust answer the question, don't add anything else!"
        )

        inputs.append(input_text)
        correct_answers.append(row['Answer 1'])  # Store correct answer to compare later

        # Process batch when the size reaches batch_size
        if len(inputs) == batch_size:
            outputs = nlp(inputs, max_new_tokens=100)

            for output, correct_answer in zip(outputs, correct_answers):
                ans = output['generated_text'].strip()

                answers = [row['Answer 1'], row['Answer 2'], row['Answer 3']]
                most_similar = difflib.get_close_matches(ans, answers, n=1)

                print(f"Model answer: {ans}, Most similar: {most_similar}, Correct answer: {correct_answer}")
                if most_similar and most_similar[0] == correct_answer:
                    correct_count += 1
                attempted_questions += 1

            # Clear batch
            inputs = []
            correct_answers = []

    # Process any remaining inputs
    if inputs:
        outputs = nlp(inputs, max_new_tokens=100)
        for output, correct_answer in zip(outputs, correct_answers):
            ans = output['generated_text'].strip()

            answers = [row['Answer 1'], row['Answer 2'], row['Answer 3']]
            most_similar = difflib.get_close_matches(ans, answers, n=1)

            print(f"Model answer: {ans}, Most similar: {most_similar}, Correct answer: {correct_answer}")
            if most_similar and most_similar[0] == correct_answer:
                correct_count += 1
            attempted_questions += 1

    if attempted_questions > 0:
        accuracy = correct_count / attempted_questions
        print(f'Accuracy of {model_name}: {accuracy}')
    else:
        print(f'No questions were attempted for model: {model_name}')
