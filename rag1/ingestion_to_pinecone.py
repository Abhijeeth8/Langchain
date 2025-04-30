import os

from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

if __name__ == "__main__":
    print("Ingestion to pinecone")

    print("-----------Loading doc to langchain document----------")

    # with open("D:/Langchain/rag1/blog_text.txt", "r", encoding='utf-8') as f:
    #     print(f.read())
    text_loader = TextLoader("D:/Langchain/rag1/blog_text.txt", encoding="utf-8")
    document = text_loader.load()
    #
    print("------------Splitting the document-----------------")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(document)
    print(len(split_docs))

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    print("-------------Embed the text into Pinecone--------------")

    PineconeVectorStore.from_documents(
        split_docs, embeddings, index_name=os.environ.get("INDEX_NAME")
    )

    print("-------------Finished Ingestion------------------------")
