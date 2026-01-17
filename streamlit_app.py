import streamlit as st
from gemini_rag import GeminiRAG
from config.env_loader import load_env
from io import StringIO
import os

load_env()


def streamlit_app():
    st.set_page_config(page_title="RAG Assistant", page_icon="./icon.png")
    col1, col2 = st.columns([1, 8])

    with col1:
        st.image("./icon.png", width=60)

    with col2:
        st.title("Knowledge Assistant")

    @st.cache_resource
    def get_rag_instance():
        return GeminiRAG(index_name="gemini-rag-index")

    rag = get_rag_instance()

    # Sidebar Setup
    with st.sidebar:
        st.header("Setup & Indexing")

        if st.button("Initialize Vector Index"):
            with st.spinner("Creating index on Atlas..."):
                rag.create_vector_index()
                st.success(
                    "Index creation submitted! Wait 2-3 mins for Atlas to build it."
                )

        url = st.text_input("Enter PDF or Web URL")
        # uploaded_file = st.file_uploader("Choose a file")
        uploaded_file = st.file_uploader("Upload a file", type=["csv", "pdf", "txt"])
        if st.button("Index"):
            save_path = "temp_knowledge"
            if uploaded_file is not None:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                    
                file_path = os.path.join(save_path, uploaded_file.name)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                print(f"./{save_path}/{uploaded_file.name}")
                rag.index_pdf(f"./{save_path}/{uploaded_file.name}")
                st.success("Content indexed successfully!")
            else:
                with st.spinner("Fetching and indexing content..."):
                    try:
                        rag.index_pdf(url)
                        st.success("Content indexed successfully!")
                    except Exception as e:
                        st.error("Indexing failed")
                        st.exception(e)

    # Main Chat Interface
    st.subheader("Ask a Question")
    with st.form("query_form"):
        query = st.text_area(
            "Context-based Question:",
            placeholder="e.g., What are the key points of the document?",
        )
        submitted = st.form_submit_button("Ask")

        if submitted:
            if query:
                with st.spinner("Searching knowledge base..."):
                    answer = rag.generate_answer(query)
                    st.markdown("### Answer")
                    st.write(answer)
            else:
                st.warning("Please enter a question first.")


if __name__ == "__main__":
    streamlit_app()
