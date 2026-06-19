from langchain_groq import ChatGroq
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os 
from dotenv import load_dotenv
load_dotenv()

## Langsmith Tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot With Langchain"


## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","you are a helpful assistant. Please response to the user queries"),
        ("user", "Question:{question}")
    ]
)

def generate_response(question, api_key, llm, temperature, max_tokens):
    llm = ChatGroq(model = llm)
    output_parser = StrOutputParser()
    chain = prompt|llm|output_parser
    answer = chain.invoke({"question":question})
    return answer


## Title of the app
st.title("Enhanced Q&A Chatbot")

st.sidebar.title("Setting")
api_key = st.sidebar.text_input("Enter your API KEY: ", type="password")

llm = st.sidebar.selectbox("Select an AI Model", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama-3.2-1b-preview"])

temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

## main interface
st.write("Go ahead and ask any question")
user_input = st.text_input("you: ")
if user_input:
    response = generate_response(user_input, api_key, llm, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please provide the query")
