import ast
import argparse

from tqdm import tqdm

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document


from pymilvus import connections, FieldSchema, CollectionSchema, Collection, DataType, utility
from milvus import default_server

from datasets import load_dataset

try:
    connections.connect("default", host="0.0.0.0")
except:
    default_server.start()


def ingest_single_metadata(data, vector_db, pks):

    metadata_text = f"Legge numero {data['law_num']} anno {data['year']} organo {data['region']}"
    law_id = f"{data['region']}_{data['year']}_{data['law_num']}"
    milvus_instance = Document(page_content=metadata_text, 
                                metadata={"law_id": law_id, "law_num": data["law_num"],
                                                            "year": data["year"], 
                                                            "organ": data["region"]})
    
    text_splitter = CharacterTextSplitter(chunk_size=100000, chunk_overlap=0)
    milvus_instance = text_splitter.split_documents([milvus_instance])
    vector_db.upsert(pks, milvus_instance)


def ingest_single_article(art_num, art_text, data, vector_db, pks):
    article_law_id = f"{data['region']}_{data['year']}_{data['law_num']}_art{art_num}"
    law_id = f"{data['region']}_{data['year']}_{data['law_num']}"
    milvus_instance = Document(page_content=art_text, 
                                metadata={"article_id": article_law_id, "law_id": law_id, "law_num": data["law_num"],
                                                            "year": data["year"], 
                                                            "region": data["region"]})
    
    text_splitter = CharacterTextSplitter(chunk_size=100000, chunk_overlap=0)
    milvus_instance = text_splitter.split_documents([milvus_instance])
    vector_db.upsert(pks, milvus_instance)


def initiate_metadata_table():

    # Define the schema for the collection
    fields = [
        FieldSchema(name="law_id", dtype=DataType.INT64, is_primary=True, auto_id=False),  # Use law_id as primary key
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128),  # Adjust dim to your embedding size
        FieldSchema(name="law_num", dtype=DataType.INT64),
        FieldSchema(name="year", dtype=DataType.INT64),
        FieldSchema(name="region", dtype=DataType.VARCHAR, max_length=100)
    ]

    schema = CollectionSchema(fields, "Collection description")

    # Create the collection
    vector_db = Collection(name="meta_data_test", schema=schema)

    return vector_db


def create_law_metadata_db():
    dataset = load_dataset("paoloitaliani/regional_laws")["train"].shuffle(seed=42)

    collection_name = "law_metadata_db_500"

    
    if utility.has_collection(collection_name):
        collection = Collection(name=collection_name)
        collection.drop()
    else:
        pass

    milvus_instance = Document(page_content="first istance", 
                                metadata={"law_id": "initial_instance", "law_num": "NA",
                                                            "year": 2024, 
                                                            "organ": "NA"})
    
    text_splitter = CharacterTextSplitter(chunk_size=100000, chunk_overlap=0)
    milvus_instance = text_splitter.split_documents([milvus_instance])
        
    embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", model_kwargs={"device": "cuda"})
    vector_db = Milvus.from_documents(milvus_instance, 
                                      embedder,
                                      connection_args={"host": "127.0.0.1", "port": "19530"},
                                      collection_name=collection_name)
    
    expr = "law_id in ['initial_instance']"
    pks = vector_db.get_pks(expr)
    
    for data in tqdm(dataset):        
        ingest_single_metadata(data, vector_db, pks)

def create_article_db():
    dataset = load_dataset("paoloitaliani/regional_laws")["train"].shuffle(seed=42)

    collection_name = "article_db_500"

    
    if utility.has_collection(collection_name):
        collection = Collection(name=collection_name)
        collection.drop()
    else:
        pass

    milvus_instance = Document(page_content="first istance", 
                                metadata={"article_id": "initial_instance", "law_id": "NA", "law_num": "NA",
                                                            "year": 2024, 
                                                            "region": "NA"})
    
    text_splitter = CharacterTextSplitter(chunk_size=100000, chunk_overlap=0)
    milvus_instance = text_splitter.split_documents([milvus_instance])
        
    embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", model_kwargs={"device": "cuda"})
    vector_db = Milvus.from_documents(milvus_instance, 
                                      embedder,
                                      connection_args={"host": "127.0.0.1", "port": "19530"},
                                      collection_name=collection_name)
    
    expr = "article_id in ['initial_instance']"
    pks = vector_db.get_pks(expr)
    
    for data in tqdm(dataset):

        try:
            formatted_data = ast.literal_eval(data["articles"])
            for art_num, art_text in formatted_data.items():
                ingest_single_article(art_num, art_text, data, vector_db, pks)
        except:
            pass
                

def perform_query():
    
    # test_query = "roccia asfaltica regione sicilia"
    test_query = "roccia asfaltica regione sicilia 1951 n. 3"
    embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", model_kwargs={"device": "cuda"})
    if args.search == "metadata_and_article":

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
        article_result = article_db.as_retriever(search_kwargs={"expr": f'law_id == "{law_id}"'}).invoke(test_query)[0]
        print(article_result.page_content)
    
    elif args.search == "metadata":
        law_metadata_db = Milvus(
            embedder,
            collection_name="law_metadata_db_500",
            auto_id=True,
        )

        result = law_metadata_db.similarity_search(test_query)[0]
        print(result)
    
    elif args.search == "article":
        article_db = Milvus(
            embedder,
            collection_name="article_db_500",
            auto_id=True,
        )
        
        article_results = article_db.similarity_search(test_query)[:2]
        print("\n")
        for article_result in article_results:
            print(article_result.page_content)
            print("\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    initiate_metadata_table()
    print(1)
    create_article_db()
    print(2)
    create_law_metadata_db()
    print(3)
    
    parser.add_argument("--search", type=str, default="metadata_and_article")
    args = parser.parse_args()
    print(4)
    perform_query()
    print(5)