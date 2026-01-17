import streamlit as st
from gemini_rag import GeminiRAG
from config.env_loader import load_env
import json
import os

load_env()


def save_session_state(target_file, current_session_state):
    with open(file=target_file, mode="w", encoding="utf-8") as file:
        json.dump(current_session_state, file, indent=4)
    return


def streamlit_app():
    st.set_page_config(page_title="RAG Assistant", page_icon="./icon.png")

    # Layout
    col1, col2 = st.columns([1, 8])
    with col1:
        st.image("./icon.png", width=60)
    with col2:
        st.title("Knowledge Assistant")

    @st.cache_resource
    def get_rag_instance():
        return GeminiRAG(index_name="gemini-rag-index")

    rag = get_rag_instance()

    # Sidebar: Setup & Indexing
    with st.sidebar:
        st.header("Setup & Indexing")
        if st.button("Initialize Vector Index"):
            with st.spinner("Creating index..."):
                rag.create_vector_index()
                st.success("Index creation submitted!")

        url = st.text_input("Enter PDF or Web URL")
        uploaded_file = st.file_uploader("Upload a file", type=["csv", "pdf", "txt"])

        if st.button("Index Content"):
            save_path = "temp_knowledge"
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            with st.spinner("Indexing..."):
                target = url
                if uploaded_file:
                    target = os.path.join(save_path, uploaded_file.name)
                    with open(target, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                if target:
                    rag.index_pdf(target)
                    st.success("Content indexed successfully!")

    # --- Chat Interface ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history from session state on rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask me anything about your documents..."):
        # Display user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display assistant response
        with st.chat_message("assistant"):
            response = rag.generate_answer(prompt)
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            save_path = "data_storage"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            target = os.path.join(save_path, "session_history.json")
            save_session_state(
                target, {"messages": st.session_state.get("messages", [])}
            )


if __name__ == "__main__":
    streamlit_app()
