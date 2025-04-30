import streamlit as st
from backend.rag_backend import perform_rag

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_response_history" not in st.session_state:
    st.session_state["chat_response_history"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

st.header("Hello, Langchain!")

prompt = st.text_input("Prompt", placeholder="Enter your prompt")


def format_sources(sources):
    formatted_sources = ""
    for source in sources:
        formatted_sources = formatted_sources + "\n"+source

    return formatted_sources


if prompt:
    with st.spinner("Generating response"):
        generated_response = perform_rag(prompt, st.session_state["chat_history"])

        sources = set([doc.metadata["source"] for doc in generated_response["source_documents"]])

        formatted_result = generated_response["result"] + "\n\n" + format_sources(sources)

        st.session_state["user_prompt_history"].append(generated_response["query"])
        st.session_state["chat_response_history"].append(formatted_result)
        st.session_state["chat_history"].append({"role":"human", "content":generated_response["query"]})
        st.session_state["chat_history"].append({"role":"ai", "content":formatted_result})

if st.session_state["chat_response_history"]:
    for user_prompt, llm_response in zip(st.session_state["user_prompt_history"], st.session_state["chat_response_history"]):
        st.chat_message("user").write(user_prompt)
        st.chat_message("assistant").write(llm_response)