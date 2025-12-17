import streamlit as st
from gemini_rag import GeminiRAG
from dotenv import load_dotenv
import os

# 1. Page Config & Title
st.set_page_config(page_title="RAG Assistant", page_icon="./Gemini_.png")
col1, col2 = st.columns([1, 8])

with col1:
    st.image("./Gemini_.png", width=60)

with col2:
    st.title("Knowledge Assistant")
# st.title(f"{page_icon='./Gemini_.png'} Gemini Knowledge Assistant")

load_dotenv()


# 2. Initialize RAG Class with Caching
# We use @st.cache_resource so the MongoDB connection only happens once
@st.cache_resource
def get_rag_instance():
    mongo_uri = (
        f"mongodb+srv://{st.secrets['MONGODB_USER']}:"
        f"{st.secrets['MONGODB_PASSWORD']}@rag.gliqall.mongodb.net/"
        "?appName=RAG"
    )
    return GeminiRAG(
        mongo_uri=mongo_uri, db_name="sample_mflix", collection_name="ragpdf"
    )


rag = get_rag_instance()

# 3. Sidebar for Setup
with st.sidebar:
    st.header("Setup & Indexing")

    # Checkbox to trigger index creation (Only run this once!)
    if st.button("Initialize Vector Index"):
        with st.spinner("Creating index on Atlas..."):
            rag.create_vector_index()
            st.success("Index creation submitted! Wait 2-3 mins for Atlas to build it.")

    # Local PDF Upload
    uploaded_file = st.file_uploader("Upload a PDF for your knowledge base", type="pdf")
    if uploaded_file is not None:
        # Save temporary file to disk
        temp_path = "temp_knowledge.pdf"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("Index Document"):
            with st.spinner("Processing PDF and generating embeddings..."):
                rag.index_pdf(temp_path)
                st.success("Document indexed successfully!")

# 4. Main Chat Interface
st.subheader("Ask a Question")
with st.form("query_form"):
    query = st.text_area(
        "Context-based Question:",
        placeholder="e.g., What are the four things to avoid according to the document?",
    )
    submitted = st.form_submit_button("Ask")

    if submitted:
        if query:
            with st.spinner("Searching knowledge base..."):
                # Use your refined prompt inside this method
                answer = rag.generate_answer(query)
                st.markdown("### Answer")
                st.write(answer)
        else:
            st.warning("Please enter a question first.")
