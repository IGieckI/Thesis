import os
import gc
import json
import torch
import pandas as pd
from milvus import default_server
from pymilvus import connections, MilvusClient
from transformers import AutoTokenizer, AutoModel

isLinux = True
default_linux_path = os.getcwd().replace("/Data", "/Documents/Downloaded") if "/Data" in os.getcwd() else os.getcwd() + "/Documents/Downloaded"
default_windows_path = os.getcwd().replace("\\Data", "\\Documents\\Downloaded") if "\\Data" in os.getcwd() else os.getcwd() + "\\Documents\\Downloaded"
default_path = default_linux_path if isLinux else default_windows_path

os.environ['TRANSFORMERS_CACHE'] = '/llms'

DEFAULT_SAVE_DIR = default_path.replace("/Downloaded", "/Generated") if isLinux else default_path.replace("\\Downloaded", "\\Generated")

QUEST_CSV = DEFAULT_SAVE_DIR + ("/quiz_merged.csv" if isLinux else "\\quiz_merged.csv")
REF_MERG = DEFAULT_SAVE_DIR + ('/references_merged.csv' if isLinux else '\\references_merged.csv')
ALL_LAWS_CSV = DEFAULT_SAVE_DIR + ("/All laws extracted.csv" if isLinux else "\\All laws extracted.csv")
TRAINING_DATA = DEFAULT_SAVE_DIR + ("/training_data.json" if isLinux else "\\training_data.json")

TMP_ALL_LAWS_CSV = DEFAULT_SAVE_DIR + ("/TMP All laws.csv" if isLinux else "\\TMP All laws.csv")

LAWS_COLLECTION_NAME = "laws"
QUIZZES_COLLECTION_NAME = "quizzes"

TRAINING_BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_JSON = DEFAULT_SAVE_DIR + "train.json"
VAL_JSON = DEFAULT_SAVE_DIR + "val.json"
#EMB_WEIGHTS = DEFAULT_SAVE_DIR + "embedding_weights_H.pt"
TRAINED_MODEL_DIR = default_path.replace("/Downloaded", "/TrainedModelFrozenFirst") if isLinux else default_path.replace("\\Downloaded", "\\TrainedModelFrozenFirst")
TRAINED_TOKENIZER_DIR = default_path.replace("/Downloaded", "/TrainedTokenizerFrozenFirst") if isLinux else default_path.replace("\\Downloaded", "\\TrainedTokenizerFrozenFirst")

# Load retrieval model
retrieval_model = AutoModel.from_pretrained('BAAI/bge-m3').to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')

for name, param in retrieval_model.named_parameters():
    if "embeddings" not in name:
        param.requires_grad = False

MILVUS_HOST = os.getenv('MILVUS_HOST', '127.0.0.1')
MILVUS_PORT = os.getenv('MILVUS_PORT', '19530')

def connectToMilvus():
    try:
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
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
        encoded_input = tokenizer(doc, padding=True, truncation=True, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            model_output = model(**encoded_input)
            pooled_embedding = model_output.last_hidden_state.mean(dim=1).squeeze()
        
        embeddings.append(pooled_embedding.cpu().numpy())  # Detach and move to CPU to free up GPU memory

    return embeddings

    
# @title Define dataloader creator functions
def create_dataloader(triplets: list, tokenizer, batch_size=TRAINING_BATCH_SIZE):
    """Takes in triplets of json as a list,
    `[{'query': $query, 'pos': $posPassage, 'neg': $negPassage}]`
    """

    # Tokenize queries and passages separately
    queries_input_ids, passages_input_ids = [], []
    queries_attention_masks, passages_attention_masks = [], []
    labels = []

    for pair_dict in triplets:
        query = pair_dict['query']
        pos   = pair_dict['pos']
        neg   = pair_dict['neg']

        query_encoded       = tokenizer(query, padding='max_length', max_length=128, truncation=True, return_tensors='pt')
        pos_passage_encoded = tokenizer(pos, padding='max_length', max_length=128, truncation=True, return_tensors='pt')
        neg_passage_encoded = tokenizer(neg, padding='max_length', max_length=128, truncation=True, return_tensors='pt')

        queries_input_ids.append(query_encoded['input_ids'].squeeze(0))
        queries_attention_masks.append(query_encoded['attention_mask'].squeeze(0))
        passages_input_ids.append(pos_passage_encoded['input_ids'].squeeze(0))
        passages_attention_masks.append(pos_passage_encoded['attention_mask'].squeeze(0))
        labels.append(1)

        queries_input_ids.append(query_encoded['input_ids'].squeeze(0))
        queries_attention_masks.append(query_encoded['attention_mask'].squeeze(0))
        passages_input_ids.append(neg_passage_encoded['input_ids'].squeeze(0))
        passages_attention_masks.append(neg_passage_encoded['attention_mask'].squeeze(0))
        labels.append(-1)

    # Convert lists to tensors
    queries_input_ids         = torch.stack(queries_input_ids)
    queries_attention_masks   = torch.stack(queries_attention_masks)
    passages_input_ids        = torch.stack(passages_input_ids)
    passages_attention_masks  = torch.stack(passages_attention_masks)
    labels = torch.tensor(labels)

    dataset = torch.utils.data.TensorDataset(queries_input_ids, queries_attention_masks, passages_input_ids, passages_attention_masks, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    return dataloader

"""
client = connectToMilvus()

if not (QUIZZES_COLLECTION_NAME in client.list_collections() and LAWS_COLLECTION_NAME in client.list_collections()):
    raise Exception("Collections not found. Please run the Data tokenization script first.")

# Load collections and create indexes if they don't exist
quizzes_collection = Collection(name=QUIZZES_COLLECTION_NAME)
laws_collection = Collection(name=LAWS_COLLECTION_NAME)

index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",  # try also "COSINE"
    "params": {"nlist": 128}
}

if not client.list_indexes(LAWS_COLLECTION_NAME):
    laws_collection.create_index(field_name="law_embedding", index_params=index_params, index_name="law_index")
if not client.list_indexes(QUIZZES_COLLECTION_NAME):
    quizzes_collection.create_index(field_name="quiz_embedding", index_params=index_params, index_name="quiz_index")
"""
quizzes_df = pd.read_csv(QUEST_CSV) #quiz_id,question,answer_1,answer_2,answer_3
laws = pd.read_csv(TMP_ALL_LAWS_CSV) #law_source,law_text,law_number,law_year
references = pd.read_csv(REF_MERG) #Source,Comma,Reference,Question id,Question plh,Law 

quizzes_df["quiz_id"] = range(len(quizzes_df))

# create triplets list
# read the dataset json
with open(TRAINING_DATA, 'r') as f:
    text_pairs = json.load(f)

prefix = "Represent this sentence for searching relevant passages: "
triplets = [{'query': prefix + elem["query"], 'pos': elem['pos'], 'neg': elem['neg']} for elem in text_pairs]

train_triplets = triplets[:int(0.9 * len(triplets))]
val_triplets = triplets[int(0.9 * len(triplets)):]

print(len(train_triplets), len(val_triplets))

# Define loss function and optimizer
train_dataloader = create_dataloader(train_triplets, tokenizer=tokenizer, batch_size=TRAINING_BATCH_SIZE)
val_dataloader = create_dataloader(val_triplets, tokenizer=tokenizer, batch_size=TRAINING_BATCH_SIZE)

initial_weights = retrieval_model.embeddings.word_embeddings.weight.cpu().detach().numpy().copy()

criterion = torch.nn.CosineEmbeddingLoss()

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, retrieval_model.parameters()), lr=1e-4)

train_losses = []
val_losses = []

# Stop training when validation loss goes up
patience_count = 0
patience_limit = 2

# Accumulate gradient to reach desired batch_size
# Our batch size is 8, and we aim for 64
# Large batch sizes suit well our dataset with large number of
# Negative passages
accumulation_count = 0
accumulation_steps = 8

for epoch in range(20):
    print(f"Epoch {epoch}")
    retrieval_model.train()
    # Training loop
    epoch_losses = []
    iteration = 0
    for query_input_ids, query_attention_masks, passage_input_ids, passage_attention_masks, batch_labels in train_dataloader:        
        # Move data to GPU
        (query_input_ids,
          query_attention_masks,
          passage_input_ids,
          passage_attention_masks,
          batch_labels) = (query_input_ids.to(DEVICE),
                        query_attention_masks.to(DEVICE),
                        passage_input_ids.to(DEVICE),
                        passage_attention_masks.to(DEVICE),
                        batch_labels.to(DEVICE))

        optimizer.zero_grad()
        print("-" * 50)
        print(f"Epoch {epoch} Iteration {iteration}")
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
        print(f"Torch memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2}")
        print(f"Torch memory allocated: {torch.cuda.memory_reserved() / 1024 ** 2}")
        print("-" * 50)
        # Forward pass
        query_outputs = retrieval_model(query_input_ids, attention_mask=query_attention_masks)
        passage_outputs = retrieval_model(passage_input_ids, attention_mask=passage_attention_masks)

        q_logits = query_outputs.last_hidden_state[:, 0, :]
        p_logits = passage_outputs.last_hidden_state[:, 0, :]

        loss = criterion(q_logits, p_logits, batch_labels)

        loss = loss / accumulation_steps
        
        try:
            loss.backward()
        except Exception as e:
            print(e)
            break

        epoch_losses.append(loss.item())
        accumulation_count += 1
        if accumulation_count == accumulation_steps:
            # Zero out gradients of the old tokens - we only train new tokens
            # model.embeddings.word_embeddings.weight.grad[:old_vocab_size] = 0
            optimizer.step() # Optimize weights
            accumulation_count = 0

        gc.collect()
        del query_outputs, passage_outputs
        iteration+=1
    torch.cuda.empty_cache()

    mean_loss = sum(epoch_losses) / len(epoch_losses) * accumulation_steps
    train_losses.append(mean_loss)
    print(f"Epoch {epoch}, training loss: {mean_loss}")

    # Evaluation loop
    tmp_val_losses = []
    retrieval_model.eval()
    for query_input_ids, query_attention_masks, passage_input_ids, passage_attention_masks, batch_labels in val_dataloader:
        (query_input_ids,
          query_attention_masks,
          passage_input_ids,
          passage_attention_masks,
          batch_labels) = (query_input_ids.to(DEVICE),
                        query_attention_masks.to(DEVICE),
                        passage_input_ids.to(DEVICE),
                        passage_attention_masks.to(DEVICE),
                        batch_labels.to(DEVICE))

        with torch.no_grad():
            query_outputs   = retrieval_model(query_input_ids, attention_mask=query_attention_masks)
            passage_outputs = retrieval_model(passage_input_ids, attention_mask=passage_attention_masks)

            q_logits = query_outputs.last_hidden_state[:, 0, :]
            p_logits = passage_outputs.last_hidden_state[:, 0, :]

            loss = criterion(q_logits, p_logits, batch_labels)
            tmp_val_losses.append(loss.item())

    mean_val_loss = sum(tmp_val_losses) / len(tmp_val_losses)
    print(f"Epoch {epoch}, validation loss: {mean_val_loss}")

    val_losses.append(mean_val_loss)
    current_val_loss = val_losses[epoch]

    if epoch > 0:
        prev_val_loss = val_losses[epoch - 1]

        # Early stopping, if validation loss goes up two consecutive rounds stop the training
        # This could be a sign of overfitting.
        if prev_val_loss < current_val_loss:
            patience_count += 1
            if patience_count >= patience_limit:
                print(f"Validation loss went up by {current_val_loss - prev_val_loss}. Terminating..")
                break
        else:
            patience_count = 0

print(len(train_losses))
# Save the model and the losses
with open(TRAIN_JSON, "w") as f:
    f.write(json.dumps(train_losses))

print(len(val_losses))
with open(VAL_JSON, "w") as f:
    f.write(json.dumps(val_losses))

#torch.save(retrieval_model.embeddings.word_embeddings.weight.H, EMB_WEIGHTS)
retrieval_model.save_pretrained(TRAINED_MODEL_DIR)
tokenizer.save_pretrained(TRAINED_TOKENIZER_DIR)