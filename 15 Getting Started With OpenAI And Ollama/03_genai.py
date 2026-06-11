import os
from dotenv import load_dotenv

from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ['LANGSMITH_TRACING'] = 'True'
os.environ['LANGSMITH_PROJECT'] = os.getenv("LANGSMITH_PROJECT")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a helpful assistant. Please respond to the question asked"),
        ("user", "Question: {question}")
    ]
)

st.title("Langchain Demo With Gamma Model")
input_text = st.text_input("What question you have in mind")

llm = Ollama(model = "gemma:2b")
output_parser=StrOutputParser()
chain = prompt|llm|output_parser

if input_text: 
    st.write(chain.invoke({"question":input_text}))
