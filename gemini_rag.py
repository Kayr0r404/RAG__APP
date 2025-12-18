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


class RemoteURLLoader(BaseLoader):
    """Robust loader for remote PDFs with validation."""

    def __init__(self, url: str, timeout: int = 30):
        self.url = url
        self.timeout = timeout

    def load(self):
        try:
            print(f"Fetching PDF from {self.url}...")

            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0 Safari/537.36"
                )
            }

            response = requests.get(
                self.url,
                headers=headers,
                stream=True,
                allow_redirects=True,
                timeout=self.timeout,
            )
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "").lower()

            if "application/pdf" not in content_type:
                raise ValueError(
                    f"URL did not return a PDF. " f"Content-Type='{content_type}'"
                )

            from pypdf import PdfReader

            pdf_stream = BytesIO(response.content)
            reader = PdfReader(pdf_stream)

            documents = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                documents.append(
                    {
                        "page_content": text,
                        "metadata": {
                            "source": self.url,
                            "page": i + 1,
                        },
                    }
                )

            class SimpleDocument:
                def __init__(self, page_content, metadata):
                    self.page_content = page_content
                    self.metadata = metadata

            return [SimpleDocument(d["page_content"], d["metadata"]) for d in documents]

        except Exception as e:
            print(f"Error processing remote PDF: {e}")
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
        chat_model: str = "gemini-2.5-flash",
        embedding_dim: int = 3072,
        index_name: str = "vector_index",
        chunk_size: int = 400,
        chunk_overlap: int = 20,
    ):
        self.client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

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
        """Load PDF, chunk, embed, and store in MongoDB"""

        if self.collection.count_documents({}) > 0:
            print("Collection is not empty. Skipping indexing.")
            return

        loader = RemoteURLLoader(pdf_url)
        documents = loader.load()

        if not documents:
            print("No documents loaded. Indexing aborted.")
            return

        chunks = self.splitter.split_documents(documents)
        print(f"Split document into {len(chunks)} chunks.")

        records = []
        for chunk in chunks:
            record = {
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", pdf_url),
                "page": chunk.metadata.get("page", "N/A"),
                "embedding": self.embed_document(chunk.page_content),
            }
            records.append(record)

        print(f"Inserting {len(records)} records into MongoDB...")
        self.collection.insert_many(records)
        print("Indexing complete.")

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
