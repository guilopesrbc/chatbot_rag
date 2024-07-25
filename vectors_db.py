from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import PyPDF2

FAISS_PATH = "faiss_db"

def generate_data_store(api_key: str):
    document = load_document()
    chunks = split_text(document)
    db = save_to_faiss(chunks, api_key)
    return db


def load_document():
    reader = PyPDF2.PdfReader('./data/procuradoria_geral_normas.pdf')
    document = ""
    for page in reader.pages:
        document += page.extract_text()
    return document


def split_text(document):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_text(document)

    return chunks


def save_to_faiss(chunks, api_key):
    # Create a new DB from the document.
    db = FAISS.from_texts(
        chunks, OpenAIEmbeddings(api_key=api_key)
    )
    return db
