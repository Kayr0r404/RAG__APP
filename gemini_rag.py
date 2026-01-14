import streamlit as st
from typing import List, Dict, Any
import requests
from io import BytesIO
from google import genai
from google.genai import types
from google.genai import errors
from langchain_community.document_loaders.pdf import BaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
from pymongo.errors import OperationFailure
from pinecone import Pinecone

import os
import requests
from io import BytesIO
from pypdf import PdfReader
from langfuse import observe, get_client

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
    @observe(name="pdf_loading")  # Langfuse will track how long loading takes
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

            # Update Langfuse with the number of pages loaded
            # langfuse.update_current_observation(metadata={"pages_loaded": len(docs)})

            if not is_remote:
                pdf_data.close()
            return docs

        except Exception as e:
            print(f"Error loading {source}: {e}")
            return []


# --- RAG Class Implementation ---


class GeminiRAG:
    """
    - Gemini embeddings
    - Gemini generation
    - MongoDB Atlas Vector Search
    """

    def __init__(
        self,
        mongo_uri: str,
        db_name: str,
        collection_name: str,
        embedding_model="models/gemini-embedding-001",
        chat_model: str = "gemini-3-flash-preview",
        embedding_dim: int = 3072,
        index_name: str = "vector_index",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

        # PINECONE
        pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
        if not pc.has_index(index_name):
            # pc.create_index()
            ...
            l = ""

        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.embedding_dim = embedding_dim
        self.index_name = index_name

        self.mongo_client = MongoClient(mongo_uri)
        db = self.mongo_client[db_name]
        try:
            db.drop_collection(collection_name)
        except Exception:
            ...

        try:
            db.create_collection(collection_name)
            self.collection = db[collection_name]
        except OperationFailure:
            ...

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        self.llm = self.client.models.get(model=self.chat_model)

    def _embed(self, text: str, task_type: str) -> List[float]:
        try:
            response = self.client.models.embed_content(
                model=self.embedding_model,
                contents=[text],
                config=types.EmbedContentConfig(task_type=task_type),
            )

            if not response.embeddings:
                raise ValueError("No embeddings returned from Gemini")

            embedding = response.embeddings[0].values

            if len(embedding) != self.embedding_dim:
                raise ValueError(
                    f"Invalid embedding size: expected {self.embedding_dim}, got {len(embedding)}"
                )

            return embedding

        except errors.APIError as e:
            print("\n--- GEMINI API ERROR ---")
            print(e)
            raise

        except Exception as e:
            print("\n--- EMBEDDING ERROR ---")
            print(type(e), e)
            raise

    def embed_document(self, text: str) -> List[float]:
        return self._embed(text, task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, query: str) -> List[float]:
        return self._embed(query, task_type="RETRIEVAL_QUERY")

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def index_pdf(self, pdf_url: str) -> None:
        """
        Ingest a PDF into MongoDB Atlas Vector Search.
        Safe, idempotent, and batch-embedded.
        """

        # 1. Prevent duplicate indexing (document-level)
        if self.collection.count_documents({"metadata.source": pdf_url}) > 0:
            print(f"PDF already indexed: {pdf_url}")
            return

        print(f"Indexing PDF: {pdf_url}")

        # 2. Load document
        try:
            loader = RemoteURLLoader(pdf_url)
            documents = loader.load()
        except Exception as e:
            raise RuntimeError(f"Failed to load PDF: {e}")

        if not documents:
            print("No documents loaded. Indexing aborted.")
            return

        # 3. Split into chunks
        chunks = self.splitter.split_documents(documents)

        if not chunks:
            print("No chunks generated. Indexing aborted.")
            return

        print(f"Split document into {len(chunks)} chunks")

        # 4. Prepare texts for batch embedding
        texts = [chunk.page_content for chunk in chunks]

        # 5. Generate embeddings (batch)
        try:
            embeddings = self.embed_document(texts)
        except Exception as e:
            raise RuntimeError(f"Embedding failed: {e}")

        if len(embeddings) != len(texts):
            raise ValueError("Mismatch between texts and embeddings")

        # 6. Build records
        records = []

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            if not embedding or len(embedding) != self.embedding_dim:
                raise ValueError(
                    f"Invalid embedding at chunk {i}: "
                    f"expected {self.embedding_dim}, got {len(embedding) if embedding else 'None'}"
                )

            page = chunk.metadata.get("page", "N/A")

            record = {
                "_id": f"{pdf_url}::page_{page}::chunk_{i}",
                "text": chunk.page_content,
                "embedding": embedding,
                "metadata": {"source": pdf_url, "page": page, "chunk_index": i},
            }

            records.append(record)

        # 7. Insert into MongoDB
        try:
            self.collection.insert_many(records, ordered=False)
        except Exception as e:
            raise RuntimeError(f"Failed to insert records: {e}")

        print(f"Indexing complete. {len(records)} chunks stored.")

    def create_vector_index(self):
        print(f"Attempting to create vector index '{self.index_name}'...")

        index_definition = {
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": 3072,
                    "similarity": "cosine",
                }
            ]
        }

        try:
            search_index_model = SearchIndexModel(
                definition=index_definition, name=self.index_name, type="vectorSearch"
            )

            self.collection.create_search_index(model=search_index_model)
            print(f"Vector index creation submitted. PLEASE WAIT 2-3 MINUTES.")
            print("Atlas needs time to build the index before you can query it.")

        except Exception as e:
            print(f"Index creation failed: {e}")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves relevant document chunks from MongoDB Atlas Vector Search.
        Returns a list of dicts: [{"text": ..., "score": ...}, ...]
        """

        print(f"Generating embedding for query...")
        query_embedding = self.embed_query(query)

        pipeline = [
            {
                "$vectorSearch": {
                    "index": self.index_name,
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": 100,
                    "limit": limit,
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "text": 1,
                    "source": 1,
                    "page": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]

        print(f"Retrieving top {limit} chunks...")
        return list(self.collection.aggregate(pipeline))

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate_answer(self, query: str, k: int = 5) -> str:
        """Retrieves context and generates an answer using Gemini."""

        docs = self.retrieve(query, k)
        formatted_context = []
        for doc in docs:
            context_chunk = f"### Source: {doc.get('source', 'Unknown')} (Page: {doc.get('page', 'N/A')})\n{doc['text']}"
            formatted_context.append(context_chunk)

        context = "\n\n---\n\n".join(formatted_context)

        system_instruction = (
            "You are an expert AI partner with a sophisticated, conversational, and deeply knowledgeable tone. "
            "Your goal is to provide insightful answers that feel like they come from a highly experienced human specialist.\n\n"
            "CRITICAL RULES:\n"
            "1. STRICT GROUNDING: Use ONLY the information provided in the <CONTEXT_DOCUMENTS>. "
            "Do not use external knowledge or facts not present in the text.\n"
            "2. UNCERTAINTY: If the answer is not contained within the context, you must say: "
            "'I’m sorry, but based on the specific documents I have access to, I don’t have enough information to answer that accurately.'\n"
            "3. STYLE: Avoid robotic phrasing like 'According to the text...' Instead, weave the facts "
            "naturally into a helpful conversation. Use clear explanations and professional vocabulary.\n"
        )

        prompt_template = f"""
                <CONTEXT_DOCUMENTS>
                {context}
                </CONTEXT_DOCUMENTS>
                <QUESTION>
                {query}
                </QUESTION>

                Please generate an accurate answer based strictly on the content in the <CONTEXT_DOCUMENTS> section.
                """

        print("Generating answer with Gemini...")

        response = self.client.models.generate_content(
            model=self.chat_model,
            contents=[prompt_template],
            config=genai.types.GenerateContentConfig(
                system_instruction=system_instruction
            ),
        )

        return response.text
