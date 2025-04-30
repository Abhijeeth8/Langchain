import os

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from rag1.callbacks import RAGCallbackHandler

load_dotenv()


def format_retrieved_docs(docs: list) -> str:
    joined_docs = ""
    for doc in docs:
        # print(doc.page_content + '----------------------------------\n')
        joined_docs = joined_docs + "\n----------------\n" + doc.page_content

    return joined_docs


if __name__ == "__main__":
    print("Using Custom RAG prompt")

    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(temperature=0, callbacks=[RAGCallbackHandler()])

    prompt = """Use the below provided context to answer the following question at the end.
                If you do not know the answer just say that you don't know and do not try to makeup any answers on your own at any cost.
                And to answer the question only use a maximum of 3 sentences and try to keep the answer as concise as possible.
                And in the end always say 'Thanks for asking!'.
                
                Context : {context}
                
                Question: {question}
                
                Answer : """

    rag_custom_prompt = PromptTemplate.from_template(prompt)

    vector_store = PineconeVectorStore(
        index_name=os.environ.get("INDEX_NAME"), embedding=embeddings
    )

    # vector_store = FAISS.load_local(os.environ.get("FAISS_VECTOR_STORE_DIR_PATH"), embeddings, allow_dangerous_deserialization=True)


    retriever = vector_store.as_retriever()

    # ans = retriever.invoke(input="What is RAG")

    # context_retrieved = format_retrieved_docs(ans)
    # print(res)

    rag_chain = (
        {
            "context": retriever | format_retrieved_docs,
            "question": RunnablePassthrough()
        }
        | rag_custom_prompt
        | llm
    )
    question = "What is Vector Database"
    result = rag_chain.invoke(input=question)
    print(result.content)

    # built_rag_prompt = rag_custom_prompt.invoke(input={"question": "this is a random question", "context": "this is a random context"})
    # print(built_rag_prompt)
    #
    # result = rag_custom_prompt | rag_custom_prompt | llm
