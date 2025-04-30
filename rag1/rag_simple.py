import os

from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from callbacks import RAGCallbackHandler
from langchain import hub

load_dotenv()

if __name__ == "__main__":
    print("-----------Retrieving-------------")

    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(temperature=0, callbacks=[RAGCallbackHandler()])

    prompt = """What is Vectors data store"""
    #
    #
    #
    no_rag_chain = PromptTemplate.from_template(prompt) | llm

    res = no_rag_chain.invoke(input={})
    print(res.content)

    vector_store = PineconeVectorStore(
        index_name=os.environ.get("INDEX_NAME"), embedding=embeddings
    )

    # vector_store = FAISS.load_local(os.environ.get("FAISS_VECTOR_STORE_DIR_PATH"), embeddings, allow_dangerous_deserialization=True)

    retrieval_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    # print(retrieval_prompt)

    docs_combiner = create_stuff_documents_chain(llm, retrieval_prompt)

    rag_chain = create_retrieval_chain(
        retriever=vector_store.as_retriever(), combine_docs_chain=docs_combiner
    )

    result = rag_chain.invoke(input={"input": prompt})

    print(result["answer"])
