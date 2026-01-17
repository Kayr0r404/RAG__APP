from os import environ
import streamlit as st


def load_env():
    environ["LANGFUSE_SECRET_KEY"] = st.secrets["LANGFUSE_SECRET_KEY"]
    environ["LANGFUSE_PUBLIC_KEY"] = st.secrets["LANGFUSE_PUBLIC_KEY"]
    environ["LANGFUSE_BASE_URL"] = st.secrets["LANGFUSE_BASE_URL"]
