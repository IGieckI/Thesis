import os
import torch
import difflib
import pandas as pd
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, pipeline, LlamaForCausalLM, BitsAndBytesConfig, AutoModelForCausalLM
from pymilvus import connections
from milvus import default_server


# Determine default paths based on OS
isLinux = True
default_linux_path = os.path.join(os.getcwd().replace("/Data", "/Documents/Downloaded")) if "/Data" in os.getcwd() else os.path.join(os.getcwd(), "Documents", "Downloaded")
default_windows_path = os.path.join(os.getcwd().replace("\\Data", "\\Documents\\Downloaded")) if "\\Data" in os.getcwd() else os.path.join(os.getcwd(), "Documents", "Downloaded")
default_path = default_linux_path if isLinux else default_windows_path

DEFAULT_SAVE_DIR = default_path.replace("/Downloaded", "/Generated") if isLinux else default_path.replace("\\Downloaded", "\\Generated")
LAWS_CSV = os.path.join(DEFAULT_SAVE_DIR, 'laws.csv')
QUIZ_CSV = os.path.join(DEFAULT_SAVE_DIR, 'quiz_merged.csv')
REF_CSV = os.path.join(DEFAULT_SAVE_DIR, 'references_merged.csv')

try:
    connections.connect("default", host="0.0.0.0")
except:
    default_server.start()

models = {
    "Meta-Llama 8B": {
        'model_name': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'context_window': 8000,
        'prompt_function': lambda system_prompt, user_prompt: f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
    {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        'prompt_delimiter': "<|start_header_id|>assistant<|end_header_id|>",
        'model_load_function': lambda model_name, quant_bab=None: AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="cuda") if quant_bab else LlamaForCausalLM.from_pretrained(model_name, device_map="cuda")
    },
    #"Saul": {
    #    'model_name': 'Equall/Saul-7B-Instruct-v1',
    #    'context_window': 1024,
    #    'prompt_function': lambda system_prompt, user_prompt: f"\n{system_prompt}\n|<user>|\n{user_prompt}\n|<assistant>|\n",
    #    'prompt_delimiter': "|<assistant>|\n",
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

df_quiz = df_quiz[:250]

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
    batch_size = 16
    inputs = []
    questions_answers = []

    for index, row in df_quiz.iterrows():
        ref = df_ref[df_ref['Question id'] == row['Index']]
        if ref.empty or pd.isna(ref.iloc[0]['Law text']):
            continue
        ref_text = ref.iloc[0]['Law text']
        # Look for a similar law in the database
        test_query = row['Question']
        embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", model_kwargs={"device": "cuda"})
        
        law_metadata_db = Milvus(
            embedder,
            collection_name="law_metadata_db_500",
            auto_id=True,
        )

        result = law_metadata_db.similarity_search(test_query)[0]
        print("\n")
        print(result)
        print("\n")

        article_db = Milvus(
            embedder,
            collection_name="article_db_500",
            auto_id=True,
        )
        law_id = result.metadata["law_id"]
        article_result = article_db.as_retriever(search_kwargs={"expr": f'law_id == "{law_id}"'}).invoke(test_query)
        print(article_result)
            
        #---------------------------------#

        input_text = prompt_function(
            "You are an expert in the field of law. Answer the following quiz. Choose the correct answer among the three options. This is the referenced article in the question: " + ref_text,
            row['Question'] + "\n" + row['Answer 1'] + "\n" + row['Answer 2'] + "\n" + row['Answer 3'] + "\nJust answer the question rewriting the answer you think is correct, don't add anything else!"
        )

        inputs.append(input_text)
        questions_answers.append([row['Answer 1'], row['Answer 2'], row['Answer 3']])

        # Process batch when the size reaches batch_size
        if len(inputs) == batch_size:
            outputs = nlp(inputs, max_new_tokens=100)

            for output, question_answers in zip(outputs, questions_answers):
                ans = output[0]['generated_text'].strip()
                ans = ans.split("<|start_header_id|>assistant<|end_header_id|>")[-1]

                most_similar = difflib.get_close_matches(ans, question_answers, n=1, cutoff=0)

                if most_similar and most_similar[0] == question_answers[0]:
                    correct_count += 1
                else:
                    print(f"Model answer: {ans}, Most similar: {most_similar}, Correct answer: {question_answers[0]}")
                    print(f"WRONG")
                attempted_questions += 1

            # Clear batch
            inputs = []
            questions_answers = []

    # Process any remaining inputs
    if inputs:
        outputs = nlp(inputs, max_new_tokens=100)
        for output, question_answers in zip(outputs, questions_answers):
                ans = output[0]['generated_text'].strip()
                ans = ans.split("<|start_header_id|>assistant<|end_header_id|>")[-1]

                most_similar = difflib.get_close_matches(ans, question_answers, n=1, cutoff=0)

                if most_similar and most_similar[0] == question_answers[0]:
                    correct_count += 1
                else:
                    print(f"Model answer: {ans}, Most similar: {most_similar}, Correct answer: {question_answers[0]}")
                    print(f"WRONG")
                attempted_questions += 1
                
    if attempted_questions > 0:
        accuracy = correct_count / attempted_questions
        print(f'Accuracy of {model_name}: {accuracy} ({correct_count}/{attempted_questions})')
    else:
        print(f'No questions were attempted for model: {model_name}')
