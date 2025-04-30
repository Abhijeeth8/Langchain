import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def ingest_docs():
    loader = ReadTheDocsLoader("langchain-docs-half", encoding="utf-8")

    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs-half", "https:")
        new_url = new_url.replace("langchain-docs-half", "https:")

        doc.metadata.update({"source": new_url})

    print(f"Going to add {len(documents)} to Pinecone")
    PineconeVectorStore.from_documents(
        documents, embeddings, index_name=os.environ.get("INDEX_NAME")
    )
    print("****Loading to vectorstore done ***")

if __name__ =="__main__":
    print("Langchain documents ingesting to pinecone index")

    ingest_docs()


