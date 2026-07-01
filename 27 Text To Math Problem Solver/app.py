import streamlit as st

from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq

from langchain_core.tools import tool
from langchain.agents import create_agent

from langchain_community.utilities import WikipediaAPIWrapper

import numexpr

load_dotenv()

st.set_page_config(
    page_title="AI Math Solver",
    page_icon="🧮",
    layout="wide"
)

st.title("🧮 AI Text to Math Solver")

groq_api_key = st.sidebar.text_input(
    "Enter Groq API Key",
    type="password"
)

if not groq_api_key:
    st.info("Please enter your Groq API Key.")
    st.stop()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=groq_api_key,
    temperature=0
)

wiki = WikipediaAPIWrapper()


@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for factual information."""
    return wiki.run(query)


@tool
def calculator(expression: str) -> str:
    """
    Evaluate mathematical expressions.
    """

    try:
        return str(numexpr.evaluate(expression))
    except Exception as e:
        return str(e)


@tool
def reasoning(question: str) -> str:
    """Solve logical and mathematical reasoning questions."""

    prompt = f"""
        You are an expert mathematician.

        Solve the following question.

        Question:
        {question}

        Explain step-by-step.

        If calculation is required, explain clearly.
        """

    return llm.invoke(prompt).content

agent = create_agent(
    model=llm,
    tools=[
        calculator,
        wikipedia_search,
        reasoning
    ]
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask any Math Question...")

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        tool_box = st.expander("Tool Calls", expanded=True)

        final_answer = ""

        for event in agent.stream(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": question
                    }
                ]
            },
            stream_mode="values",
        ):

            message = event["messages"][-1]

            # AI requesting tool
            if hasattr(message, "tool_calls") and message.tool_calls:

                for tool in message.tool_calls:

                    with tool_box:
                        st.write(f"### Calling Tool")
                        st.write(f"**Tool:** {tool['name']}")
                        st.code(tool["args"])

            # Final response
            if hasattr(message, "content") and message.content:
                final_answer = message.content

        st.markdown(final_answer)


