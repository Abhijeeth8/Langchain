import os

from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain import hub
from langchain_pinecone import PineconeVectorStore
from openai import vector_stores
from typing import Any

load_dotenv()

def perform_rag(question:str, chat_history:list[dict[str, Any]]):
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI()
    vector_store = PineconeVectorStore(index_name="rag1-index", embedding=embeddings)

    prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    prompt_chain = create_stuff_documents_chain(llm, prompt)

    rephrase_to_standalone_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

    history_aware_retriever = create_history_aware_retriever(llm, vector_store.as_retriever(), rephrase_to_standalone_prompt )

    rag_chain = create_retrieval_chain(history_aware_retriever, prompt_chain)


    result = rag_chain.invoke(input={"input": question, "chat_history": chat_history})

    new_result = {
        "query":result["input"],
        "result":result["answer"],
        "source_documents": result["context"]
    }
    return new_result


if __name__ =="__main__":
    print("RAG is being performed form the pinecone")
    question = "What is langchain agent"
    result = perform_rag(question)
    print(result["result"])
