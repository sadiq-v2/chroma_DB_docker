import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Function to load JSON lines
class JSONLinesLoader:
    def __init__(self, file_path):
        self.file_path = file_path
    
    def load(self):
        documents = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    doc = json.loads(line)
                    documents.append(doc)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
        return documents

# Function to process documents and split text
def process_documents(docs):
    doc_objects = [Document(page_content=json.dumps(doc)) for doc in docs]
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n"])
    return text_splitter.split_documents(doc_objects)
