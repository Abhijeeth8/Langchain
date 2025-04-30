import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

from dotenv import load_dotenv

load_dotenv()


if __name__ == "__main__":
    print("-------------Ingesting the pdf to local vector store------------------")

    pdf_path = "D:/Langchain/rag1/demo_pdf.pdf"
    pdf_loader = PyPDFLoader(pdf_path)
    loaded_raw_docs = pdf_loader.load()
    # full_raw_pdf = concat_raw_docs(loaded_raw_docs)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator='\n')
    fixed_split_docs = text_splitter.split_documents(loaded_raw_docs)

    embeddings = OpenAIEmbeddings()

    vector_store = FAISS.from_documents(documents=fixed_split_docs, embedding=embeddings)
    # vector_store_dir_path = 'faiss_vector_store'
    vector_store.save_local(os.environ.get("FAISS_VECTOR_STORE_DIR_PATH"))

    print(f"-----------Vector store {os.environ.get("FAISS_VECTOR_STORE_DIR_PATH")} is ready to use ---------------------")