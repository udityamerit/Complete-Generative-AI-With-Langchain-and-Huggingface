import validators, streamlit as st
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_classic.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

## Streamlit app
st.set_page_config(page_title="LangChain: Summarize Text From YT or Websites", page_icon="🦜")
st.title("🦜LangChain: Summarize Text From YT or Websites")
st.subheader("Summarize URL")


