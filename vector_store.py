from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os

load_dotenv()

def create_vector_store(chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    return FAISS.from_texts(chunks, embeddings)

def save_vector_store(db, path="faiss_index"):
    db.save_local(path)

def load_vector_store(path="faiss_index"):
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    return FAISS.load_local(path, embeddings)
