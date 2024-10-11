import torch
import os
import pandas as pd
from milvus import default_server
from pymilvus import (
    MilvusClient, FieldSchema, DataType, CollectionSchema, Collection
)
from transformers import AutoTokenizer, AutoModel

os.environ['TRANSFORMERS_CACHE'] = '/llms'

isLinux = True
default_linux_path = os.getcwd().replace("/Data", "/Documents/Downloaded") if "/Data" in os.getcwd() else os.getcwd() + "/Documents/Downloaded"
default_windows_path = os.getcwd().replace("\\Data", "\\Documents\\Downloaded") if "\\Data" in os.getcwd() else os.getcwd() + "\\Documents\\Downloaded"
default_path = default_linux_path if isLinux else default_windows_path

DEFAULT_SAVE_DIR = default_path.replace("/Downloaded", "/Generated") if isLinux else default_path.replace("\\Downloaded", "\\Generated")
QUIZZES_CSV = DEFAULT_SAVE_DIR + ("/quiz_merged.csv" if isLinux else "\\quiz_merged.csv")
LAWS_CSV = DEFAULT_SAVE_DIR + ("/laws.csv" if isLinux else "\\laws.csv")

TMP_ALL_LAWS_CSV = DEFAULT_SAVE_DIR + ("/TMP All laws.csv" if isLinux else "\\All laws extracted.csv")

def connectToMilvus():
    try:
        client = MilvusClient(uri="tcp://localhost:19530")
        print("Connected to Milvus.")
        return client
    except Exception as e:
        default_server.start()
        client = MilvusClient(uri="tcp://localhost:19530")
        print(f"Restarted Milvus")
        return client

def generateEmbeddings(tokenizer, model, data):
    embeddings = []
    
    for doc in data:
        encoded_input = tokenizer(doc, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
        embeddings.append(model_output.last_hidden_state.mean(dim=1).numpy()[0])

    return embeddings

def createQuizCollection(client):        
    schema = client.create_schema(auto_id=False, enable_dynamic_fields=True)
    
    schema.add_field(field_name="quiz_id", datatype=DataType.INT64, is_primary=True, auto_id=False)
    schema.add_field(field_name="question", datatype=DataType.VARCHAR, max_length=1000)
    schema.add_field(field_name="answer_1", datatype=DataType.VARCHAR, max_length=1000)
    schema.add_field(field_name="answer_2", datatype=DataType.VARCHAR, max_length=1000)
    schema.add_field(field_name="answer_3", datatype=DataType.VARCHAR, max_length=1000)
    schema.add_field(field_name="quiz_embedding", datatype=DataType.FLOAT_VECTOR, dim=1024)

    return client.create_collection(collection_name="quizzes", schema=schema)

def createLawsCollection(client):    
    schema = client.create_schema(auto_id=False, enable_dynamic_fields=True)
    
    schema.add_field(field_name="law_id", datatype=DataType.INT64, is_primary=True, auto_id=False)
    schema.add_field(field_name="law_source", datatype=DataType.VARCHAR, max_length=50)
    schema.add_field(field_name="law_year", datatype=DataType.VARCHAR, max_length=50)
    schema.add_field(field_name="law_number", datatype=DataType.VARCHAR, max_length=50)
    schema.add_field(field_name="law_text", datatype=DataType.VARCHAR, max_length=50000)
    schema.add_field(field_name="law_embedding", datatype=DataType.FLOAT_VECTOR, dim=1024)
    """
    index_params = client.prepare_index_params()
    
    index_params.add_index(
        field_name="law_id",
        index_type="STL_SORT"
    )
    index_params.add_index(
        field_name="law_embedding",
        index_type="AUTOINDEX",
        metric_type="IP",
        params={"nlist": 1024}        
    )
    
    return client.create_collection(collection_name="laws", schema=schema, index_params=index_params)
    """
    return client.create_collection(collection_name="laws", schema=schema)

# Connect to Milvus
client = connectToMilvus()

# Initialize the model and tokenizer
model_name = "BAAI/bge-m3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Read laws, quizzes and references csv
laws_df = pd.read_csv(TMP_ALL_LAWS_CSV)
quizzes_df = pd.read_csv(QUIZZES_CSV)

# Generate embeddings and add them to the DataFrames
laws_df["law_embedding"] = generateEmbeddings(tokenizer, model, laws_df["law_text"])
quizzes_df["quiz_embedding"] = generateEmbeddings(tokenizer, model, quizzes_df["question"])

# Add ID columns to DataFrames
laws_df["law_id"] = range(len(laws_df))
quizzes_df["quiz_id"] = range(len(quizzes_df))

# Change dataframe fields types to match the schema
laws_df["law_id"] = laws_df["law_id"].astype(pd.Int64Dtype())
laws_df["law_source"] = laws_df["law_source"].astype(str)
laws_df["law_year"] = laws_df["law_year"].astype(str)
laws_df["law_number"] = laws_df["law_number"].astype(str)
laws_df["law_text"] = laws_df["law_text"].astype(str)

quizzes_df["quiz_id"] = quizzes_df["quiz_id"].astype(pd.Int64Dtype())
quizzes_df["question"] = quizzes_df["question"].astype(str)
quizzes_df["answer_1"] = quizzes_df["answer_1"].astype(str)
quizzes_df["answer_2"] = quizzes_df["answer_2"].astype(str)
quizzes_df["answer_3"] = quizzes_df["answer_3"].astype(str)

# Fill NaN values with empty strings
laws_df = laws_df.fillna("")
quizzes_df = quizzes_df.fillna("")

print(client.list_collections())
if "quizzes" in client.list_collections():
    print("Dropping existing quizzes collection.")
    client.drop_collection("quizzes")
if "laws" in client.list_collections():
    print("Dropping existing laws collection.")
    client.drop_collection("laws")

# Create or get the collections
createLawsCollection(client)
createQuizCollection(client)

# Ensure DataFrame columns match the schema exactly
laws_data = laws_df[["law_id", "law_source", "law_year", "law_number", "law_text", "law_embedding"]].to_dict('records')
quizzes_data = quizzes_df[["quiz_id", "question", "answer_1", "answer_2", "answer_3", "quiz_embedding"]].to_dict('records')

# Insert data into the collections
client.insert("quizzes", quizzes_data)
client.insert("laws", laws_data)

# Close the connection
client.close()
