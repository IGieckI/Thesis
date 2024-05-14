from transformers import AutoModelForCausalLM, AutoTokenizer
import streamlit as st

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def generate_response(input_text):
    input_text = "Generate a SQL query that: " + input_text + "."
    
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.9,
        max_length=100,
    )
    generated_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]
    return generated_text

st.title("Test with Huggingface GPT2 and LangChain")

with st.form("my_form"):
    text = st.text_area("Describe a SQL query:", "")
    submitted = st.form_submit_button("Submit")
    
    if submitted:
        with st.spinner("Generating response..."):
            response = generate_response(text)
        st.info(response)

# To run use: python -m streamlit run "FilePath.py"