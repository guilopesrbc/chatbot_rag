__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules["pysqlite3"]

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import os
import shutil
import PyPDF2

CHROMA_PATH = "chroma"

def generate_data_store(api_key: str):
    document = load_document()
    chunks = split_text(document)
    save_to_chroma(chunks, api_key)


def load_document():
    reader = PyPDF2.PdfReader('./data/procuradoria_geral_normas.pdf')
    document = ""
    for page in reader.pages:
        document += page.extract_text()
    return document


def split_text(document):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_text(document)

    return chunks


def save_to_chroma(chunks, api_key):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the document.
    Chroma.from_texts(
        chunks, OpenAIEmbeddings(api_key=api_key), persist_directory=CHROMA_PATH
    )
