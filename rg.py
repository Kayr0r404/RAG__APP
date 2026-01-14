from typing import List
from pinecone import Pinecone
from google import genai
from google.genai import types
from langfuse import observe, get_client
from os import environ
from gemini_rag import UniversalPDFLoader
import streamlit as st

LANGFUSE_SECRET_KEY = st.secrets["LANGFUSE_SECRET_KEY"]
LANGFUSE_PUBLIC_KEY = st.secrets["LANGFUSE_PUBLIC_KEY"]
LANGFUSE_BASE_URL = st.secrets["LANGFUSE_BASE_URL"]

environ["LANGFUSE_SECRET_KEY"] = LANGFUSE_SECRET_KEY
environ["LANGFUSE_PUBLIC_KEY"] = LANGFUSE_PUBLIC_KEY
environ["LANGFUSE_BASE_URL"] = LANGFUSE_BASE_URL

langfuse = get_client()


class GeminiRAG:
    def __init__(self, index_name: str):
        self.client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        self.pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
        self.index = self.pc.Index(index_name)
        self.model_id = "gemini-3-flash-preview"
        self.embed_model = "text-embedding-004"
        self.embedding_dim = 768

    @observe(as_type="generation")  # Tracks this as an embedding generation
    def embed_text(self, text: str) -> List[float]:
        # Langfuse will automatically capture the input and output
        res = self.client.models.embed_content(
            model=self.embed_model,
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=self.embedding_dim),
        )
        return res.embeddings[0].values

    @observe()  # Tracks the overall indexing process
    def index_pdf(self, path_or_url: str):
        docs = UniversalPDFLoader.load(path_or_url)
        # docs = RemoteURLLoader.load(path_or_url)
        if not docs:
            return

        to_upsert = []
        for doc in docs:
            chunks = [c for c in doc.page_content.split("\n\n") if len(c) > 30]
            for i, chunk in enumerate(chunks):
                embedding = self.embed_text(chunk)  # Nested trace
                to_upsert.append(
                    {
                        "id": f"{doc.metadata['source']}_{i}",
                        "values": embedding,
                        "metadata": {"text": chunk, "source": doc.metadata["source"]},
                    }
                )

        self.index.upsert(vectors=to_upsert)

    @observe()
    def generate_answer(self, query: str):
        query_vector = self.embed_text(query)

        # In v3, we use get_client() or the local instance to update context
        results = self.index.query(vector=query_vector, top_k=5, include_metadata=True)
        context_chunks = [match["metadata"]["text"] for match in results["matches"]]
        context = "\n---\n".join(context_chunks)

        # UPDATE CONTEXT: Use langfuse.update_current_observation
        # langfuse.update_current_observation(
        #     input=query, metadata={"chunks_retrieved": len(context_chunks)}
        # )

        prompt = f"Expert Context:\n{context}\n\nUser Question: {query}"
        response = self.client.models.generate_content(
            model=self.model_id, contents=prompt
        )

        answer = response.text

        # FINAL TRACE UPDATE
        langfuse.update_current_trace(output=answer)

        return answer
