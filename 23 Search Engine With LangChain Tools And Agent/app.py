import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool
import arxiv

# --------------------------------------------------
# Load Environment Variables
# --------------------------------------------------
load_dotenv()

# --------------------------------------------------
# Streamlit Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="AI Search Agent",
    page_icon="🔎",
    layout="wide"
)

# --------------------------------------------------
# Custom Arxiv Tool (Compatible with arxiv==4.x)
# --------------------------------------------------
@tool
def arxiv_search(query: str) -> str:
    """
    Search research papers from arXiv.
    """

    try:
        client = arxiv.Client()

        search = arxiv.Search(
            query=query,
            max_results=3
        )

        results = []

        for paper in client.results(search):
            results.append(
                f"""
Title: {paper.title}

Authors: {", ".join([author.name for author in paper.authors])}

Published: {paper.published}

Summary:
{paper.summary[:800]}
"""
            )

        if not results:
            return "No papers found."

        return "\n\n" + "=" * 100 + "\n\n".join(results)

    except Exception as e:
        return f"Arxiv Search Error: {str(e)}"


# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.title("🔎 LangChain Search Agent")

st.markdown(
    """
    Search the web and research papers using:
    - DuckDuckGo Search
    - Arxiv Research Papers
    - Groq LLM
    """
)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("Settings")

api_key = st.sidebar.text_input(
    "Enter Groq API Key",
    type="password"
)

model_name = st.sidebar.selectbox(
    "Select Model",
    [
        "llama-3.3-70b-versatile",
        "llama3-8b-8192",
        "llama3-70b-8192"
    ]
)

# --------------------------------------------------
# Chat History
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I can search the web and research papers. Ask me anything."
        }
    ]

# Display History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --------------------------------------------------
# User Input
# --------------------------------------------------
prompt = st.chat_input(
    "Ask anything..."
)

if prompt:

    # Display User Message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.write(prompt)

    # Validate API Key
    if not api_key:
        st.error("Please enter your Groq API Key in the sidebar.")
        st.stop()

    try:

        # --------------------------------------------------
        # LLM
        # --------------------------------------------------
        llm = ChatGroq(
            groq_api_key=api_key,
            model=model_name,
            temperature=0
        )

        # --------------------------------------------------
        # Tools
        # --------------------------------------------------
        search_tool = DuckDuckGoSearchRun(
            name="web_search"
        )

        tools = [
            search_tool,
            arxiv_search
        ]

        # --------------------------------------------------
        # Agent
        # --------------------------------------------------
        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt="""
You are an AI Research Assistant.

You have access to:
1. DuckDuckGo web search
2. Arxiv research paper search

Use tools whenever needed.

For research-related questions:
- Prefer arxiv_search.

For current events or general information:
- Prefer web_search.

Always provide detailed answers.
"""
        )

        # --------------------------------------------------
        # Invoke Agent
        # --------------------------------------------------
        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                result = agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    }
                )

                response = result["messages"][-1].content

                st.write(response)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response
                    }
                )

    except Exception as e:
        st.error(f"Error: {str(e)}")