import concurrent.futures
import os
import json
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from loader import process_documents, JSONLinesLoader
from langchain.docstore.document import Document

# Embedding configuration
model_name = "BAAI/bge-base-en"
encode_kwargs = {'normalize_embeddings': True}
embedding = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},  # Change to 'cuda' if GPU is available
    encode_kwargs=encode_kwargs
)

def create_vector_db(json_file_path, batch_size=100):
    # Check if preprocessed data exists
    if os.path.exists("split_texts.json"):
        with open("split_texts.json", "r") as f:
            split_texts = json.load(f)
            split_texts = [Document(**doc) for doc in split_texts]  # Convert back to Document objects
        vector_db = Chroma.from_documents(
            documents=split_texts, 
            embedding=embedding,
            collection_name="local-rag"
        )
        return vector_db

    loader = JSONLinesLoader(json_file_path)
    documents = loader.load()
    
    num_batches = len(documents) // batch_size + 1
    batches = [documents[i*batch_size:(i+1)*batch_size] for i in range(num_batches)]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        split_texts_batches = list(executor.map(process_documents, batches))
    
    split_texts = [text for batch in split_texts_batches for text in batch]

    # Convert Document objects to dictionaries
    split_texts_serializable = [doc.dict() for doc in split_texts]

    # Save preprocessed data
    with open("split_texts.json", "w") as f:
        json.dump(split_texts_serializable, f)

    vector_db = Chroma.from_documents(
        documents=split_texts, 
        embedding=embedding,
        collection_name="local-rag"
    )
    return vector_db
