import os
import requests
from io import BytesIO
from pypdf import PdfReader
from langfuse import observe, get_client
import streamlit as st

LANGFUSE_SECRET_KEY = st.secrets["LANGFUSE_SECRET_KEY"]
LANGFUSE_PUBLIC_KEY = st.secrets["LANGFUSE_PUBLIC_KEY"]
LANGFUSE_BASE_URL = st.secrets["LANGFUSE_BASE_URL"]

os.environ["LANGFUSE_SECRET_KEY"] = LANGFUSE_SECRET_KEY
os.environ["LANGFUSE_PUBLIC_KEY"] = LANGFUSE_PUBLIC_KEY
os.environ["LANGFUSE_BASE_URL"] = LANGFUSE_BASE_URL

langfuse = get_client()


class Document:
    """Standard document structure for the RAG pipeline."""

    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata


class UniversalPDFLoader:
    @staticmethod
    @observe(name="pdf_loading")
    def load(source: str) -> list[Document]:
        is_remote = source.startswith(("http://", "https://"))

        try:
            if is_remote:
                print(f"Fetching remote PDF: {source}")
                headers = {
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
                }
                response = requests.get(
                    source, headers=headers, stream=True, timeout=30
                )
                response.raise_for_status()
                pdf_data = BytesIO(response.content)
            else:
                print(f"Loading local PDF: {source}")
                pdf_data = open(source, "rb")

            reader = PdfReader(pdf_data)
            docs = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                docs.append(
                    Document(
                        page_content=text, metadata={"source": source, "page": i + 1}
                    )
                )

            if not is_remote:
                pdf_data.close()
            return docs

        except Exception as e:
            print(f"Error loading {source}: {e}")
            return []
