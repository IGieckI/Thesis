# With pipeline, just specify the task and the model id from the Hub.
from transformers import pipeline
pipe = pipeline("text-generation", model="JosephusCheung/LL7M")

input_text = "Onece upon a time"

responses = pipe(input_text, max_length=100, num_return_sequences=1, do_sample=True, temperature=0.9)
print(responses)

generated_text = responses[0]['generated_text']
print(generated_text)



""" TEST SCRIPT FOR HUGGING FACE MODELS
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("JosephusCheung/LL7M")
model = AutoModelForMaskedLM.from_pretrained("JosephusCheung/LL7M")

input = "How are you?"

tokenized_input = tokenizer(input, return_tensors="pt")
print("Tokenized input:", tokenized_input, "\n")

model_output = model(**tokenized_input)
print("Model output:", model_output, "\n")

predicted_index = model_output.logits[0, 6].argmax(-1).item()
print("Predicted token index:", predicted_index, "\n")

predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
print("Predicted token:", predicted_token, "\n")
"""