import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter

from rag1.callbacks import RAGCallbackHandler

load_dotenv()

# def concat_raw_docs(docs:list) ->str:
#     return " ".join(doc.page_content for doc in docs)

def format_retrieved_docs(docs: list) -> str:
    joined_docs = ""
    for doc in docs:
        # print(doc.page_content + '----------------------------------\n')
        joined_docs = joined_docs + "\n----------------\n" + doc.page_content

    return joined_docs

if __name__ == "__main__":
    print("RAG using local store on a pdf")

    # pdf_path = "D:/Langchain/rag1/demo_pdf.pdf"
    pdf_loader = PyPDFLoader(os.environ.get("PDF_PATH"))
    loaded_raw_docs = pdf_loader.load()
    # full_raw_pdf = concat_raw_docs(loaded_raw_docs)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator='\n')
    fixed_split_docs = text_splitter.split_documents(loaded_raw_docs)

    embeddings = OpenAIEmbeddings()

    llm = ChatOpenAI(temperature = 0, callbacks=[RAGCallbackHandler()])

    # vector_store = FAISS.from_documents(documents=fixed_split_docs, embedding=embeddings)
    # vector_store.save_local('faiss_vector_store_videre')

    vector_store = FAISS.load_local(os.environ.get("FAISS_VECTOR_STORE_DIR_PATH"),embeddings,"index", allow_dangerous_deserialization=True)

    retriever = vector_store.as_retriever()

    prompt = """Use the below provided context to answer the following question at the end.
                If you do not know the answer just say that you don't know and do not try to makeup any answers on your own at any cost.
                And to answer the question only use a maximum of 5 sentences and try to be as concise as possible.
                And in the end always say 'Thanks for asking!'.
                
                Context : {context}
                
                Question: {question}
                
                Answer : """

    prompt_template = PromptTemplate.from_template(prompt)

    rag_chain = {"context": retriever | format_retrieved_docs, "question": RunnablePassthrough()} | prompt_template | llm


    question = "What is User Study 3 about?"
    result = rag_chain.invoke(input=question)
    print(result.content)

