import torch
import pandas as pd
from pymilvus import Milvus, connections, CollectionSchema, FieldSchema, DataType, Collection
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

isLinux = False
default_linux_path = os.getcwd().replace("/Data", "/Documents/Downloaded") if "/Data" in os.getcwd() else os.getcwd() + "/Documents/Downloaded"
default_windows_path = os.getcwd().replace("\\Data", "\\Documents\\Downloaded") if "\\Data" in os.getcwd() else os.getcwd() + "\\Documents\\Downloaded"
default_path = default_linux_path if isLinux else default_windows_path

DEFAULT_SAVE_DIR = default_path.replace("/Downloaded", "/Generated") if isLinux else default_path.replace("\\Downloaded", "\\Generated")

QUEST_CSV = DEFAULT_SAVE_DIR + ("/quiz_merged.csv" if isLinux else "\\quiz_merged.csv")
REF_MERG = DEFAULT_SAVE_DIR + ('/references_merged.csv' if isLinux else '\\references_merged.csv')
ALL_LAWS_CSV = DEFAULT_SAVE_DIR + ("/All laws extracted.csv" if isLinux else "\\All laws extracted.csv")

# Load retrieval model
retrieval_model = SentenceTransformer('BAAI/bge-m3')

# Ensure the embedding dimension matches the retrieval model output
embedding_dim = retrieval_model.get_sentence_embedding_dimension()

# Connect to Milvus
MILVUS_HOST = os.getenv('MILVUS_HOST', '127.0.0.1')
MILVUS_PORT = os.getenv('MILVUS_PORT', '19530')

connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

# Define collection schema
fields = [
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim, is_primary=False),
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
]
schema = CollectionSchema(fields, description="Document metadata collection")

# Check if collection exists, create if not
collection_name = "law_metadata_db_500"
if not utility.has_collection(collection_name):
    collection = Collection(name=collection_name, schema=schema)
    print(f"Created collection {collection_name}")
else:
    collection = Collection(name=collection_name)
    print(f"Using existing collection {collection_name}")

# Create index if not already present
index_params = {
    "index_type": "HNSW",
    "metric_type": "L2",  # Use "IP" for cosine similarity
    "params": {"M": 16, "efConstruction": 200}
}
if not collection.has_index():
    collection.create_index(field_name="embedding", index_params=index_params)
    print(f"Created index for collection {collection_name}")

def get_contextual_query_embedding(query, history=[]):
    contextual_query = " ".join(history + [query])
    query_embedding = retrieval_model.encode(contextual_query, convert_to_tensor=True)
    return query_embedding

# Load re-ranking model
rerank_model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
rerank_tokenizer = AutoTokenizer.from_pretrained(rerank_model_name)
rerank_model = AutoModelForSequenceClassification.from_pretrained(rerank_model_name)

def rerank_documents(query, documents):
    scores = []
    for doc in documents:
        inputs = rerank_tokenizer(query, doc, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            logits = rerank_model(**inputs).logits
            scores.append(logits.item())
    ranked_docs = [doc for _, doc in sorted(zip(scores, documents), reverse=True)]
    return ranked_docs, scores

def fine_tune_retrieval_model(query, positive_docs, learning_rate=1e-5, epochs=3, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Ensure model is on the correct device
    retrieval_model.to(device)
    loss_fn = torch.nn.CosineEmbeddingLoss(margin=0.5)
    optimizer = torch.optim.Adam(retrieval_model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0
        for pos_doc in positive_docs:
            query_embedding = retrieval_model.encode(query, convert_to_tensor=True).to(device)
            pos_embedding = retrieval_model.encode(pos_doc, convert_to_tensor=True).to(device)

            pos_label = torch.tensor([1.0], dtype=torch.float).to(device)
            pos_loss = loss_fn(query_embedding, pos_embedding, pos_label)

            total_loss = pos_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss.item():.4f}")
    print("Fine-tuning complete.")

def integrate_retrieval_results(query, documents):
    weighted_context = "\n".join(documents)
    return f"Query: {query}\nContext: {weighted_context}\nAnswer:"

def evaluate_retrieval(retrieved_docs, ground_truth_docs, k=10):
    retrieved_k = retrieved_docs[:k]
    relevant_retrieved = [doc for doc in retrieved_k if doc in ground_truth_docs]
    precision_at_k = len(relevant_retrieved) / k
    recall_at_k = len(relevant_retrieved) / len(ground_truth_docs)
    f1_at_k = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k) if (precision_at_k + recall_at_k) > 0 else 0

    print(f"Precision@{k}: {precision_at_k:.2f}")
    print(f"Recall@{k}: {recall_at_k:.2f}")
    print(f"F1@{k}: {f1_at_k:.2f}")

    return {
        "Precision@K": precision_at_k,
        "Recall@K": recall_at_k,
        "F1@K": f1_at_k
    }
    
# Read questions, laws and references from file
questions = pd.read_csv(QUEST_CSV)
laws = pd.read_csv(ALL_LAWS_CSV)
references = pd.read_csv(REF_MERG)

# Start the training loop
for idx, row in references.iterrows():
    query = row['question']
    positive_docs = laws[laws['law_id'].isin(row['positive_laws'])]['law_text'].tolist()
    negative_docs = laws[laws['law_id'].isin(row['negative_laws'])]['law_text'].tolist()

    fine_tune_retrieval_model(query, positive_docs, negative_docs)
query = "example query"
positive_docs = ["positive doc 1", "positive doc 2"]
negative_docs = ["negative doc 1", "negative doc 2"]

fine_tune_retrieval_model(query, positive_docs, negative_docs)
