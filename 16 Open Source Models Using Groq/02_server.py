from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes
import os
from dotenv import load_dotenv
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
model = ChatGroq(model= "llama-3.1-8b-instant", groq_api_key=groq_api_key)
 

## 1. Prompt templates

generic_template = "Translate the following into {languages}:"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",generic_template),
        ("user","{text}")
    ]
)

parser = StrOutputParser()

## create chain 
chain = prompt|model|parser

## App definition
app = FastAPI(title = "langchain Server",
              version="1.0",
              description = "A Simple API server using Langchain runnable interface")

## Adding chain routes
add_routes(
    app,
    chain,
    path = "/chain"

)


if __name__ =="__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

