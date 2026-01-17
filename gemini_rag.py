from typing import List
from pinecone import Pinecone
from google import genai
from google.genai import types
from langfuse import observe, get_client
from universal_pdf_loader import UniversalPDFLoader
import streamlit as st

langfuse = get_client()


class GeminiRAG:
    @observe()
    def __init__(self, index_name: str):
        self.client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        self.pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
        self.index = self.pc.Index(index_name)
        self.model_id = "gemini-3-flash-preview"
        self.embed_model = "text-embedding-004"
        self.embedding_dim = 768

    @observe(as_type="generation")
    def embed_text(self, text: str) -> List[float]:
        res = self.client.models.embed_content(
            model=self.embed_model,
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=self.embedding_dim),
        )
        return res.embeddings[0].values

    @observe()
    def index_pdf(self, path_or_url: str):
        docs = UniversalPDFLoader.load(path_or_url)
        if not docs:
            return

        to_upsert = []
        for doc in docs:
            chunks = [c for c in doc.page_content.split("\n\n") if len(c) > 30]
            for i, chunk in enumerate(chunks):
                embedding = self.embed_text(chunk)
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

        results = self.index.query(vector=query_vector, top_k=5, include_metadata=True)
        context_chunks = [match["metadata"]["text"] for match in results["matches"]]
        context = "\n---\n".join(context_chunks)

        prompt = f"Expert Context:\n{context}\n\nUser Question: {query}"
        response = self.client.models.generate_content(
            model=self.model_id, contents=prompt
        )

        answer = response.text
        langfuse.update_current_trace(output=answer)

        return answer
