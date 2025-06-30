import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

PERSIST_DIR = "faiss_index"


def load_and_split_documents(directory="data"):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(directory, filename))
            documents.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    return chunks


def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})


def build_vector_store(documents, persist_directory=PERSIST_DIR):
    embeddings = get_embeddings()
    vectordb = FAISS.from_documents(documents, embeddings)
    vectordb.save_local(persist_directory)

    return vectordb


def load_vector_store():
    if os.path.exists(PERSIST_DIR):
        embeddings = get_embeddings()
        vectordb = FAISS.load_local(PERSIST_DIR, embeddings,allow_dangerous_deserialization=True)
    else:
        # Load and split your documents again
        docs = load_and_split_documents()
        # Rebuild vector store (FAISS is in-memory, so no persistence)
        vectordb = build_vector_store(docs, PERSIST_DIR)

    return vectordb
