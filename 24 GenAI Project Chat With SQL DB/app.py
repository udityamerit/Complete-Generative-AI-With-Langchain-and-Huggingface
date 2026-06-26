## https://reference.langchain.com/python/langchain-community/agent_toolkits/sql/base/create_sql_agent

import sqlite3
from pathlib import Path

import streamlit as st
from sqlalchemy import create_engine

from langchain_groq import ChatGroq

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from sqlalchemy.engine import URL

st.set_page_config(page_title= "Langchain: chat with SQL DB", page_icon="🦜")

st.title("🦜LangChain: Chat with SQL DB")

# INJECTION_WARNING = """
# SQL agent can be vulnerable to prompt injection. Use the DB role with limts"""

LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"

radio_opt = ["Use SQLITE3 Database schoolDB.db", "connect to you SQL Database"]

selected_opt = st.sidebar.radio(label = "Choose the DB which you want to chat", options=radio_opt)

if radio_opt.index(selected_opt)==1:
    db_uri=MYSQL
    mysql_host = st.sidebar.text_input("Provide my SQL Host")
    mysql_user = st.sidebar.text_input("MYSQL USER")
    mysql_password = st.sidebar.text_input("MYSQL password", type="password")
    mysql_db = st.sidebar.text_input("My SQL database")
else:
    db_uri = LOCALDB


if not db_uri:
    st.info("please enter the database information and uri")

api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password"
)

if not api_key:
    st.info("Please enter your Groq API Key.")
    st.stop()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=api_key,
    temperature=0,
    verbose=True,
    streaming=True,
)

@st.cache_resource(ttl = "2h")
def configure_db(db_uri, mysql_host=None, mysql_user=None, mysql_password = None, mysql_db=None):
    if db_uri==LOCALDB:
        dbfilepath = (Path(__file__).parent / "student.db").resolve()
        # print(dbfilepath)
        engine = create_engine(f"sqlite:///{dbfilepath}")
        return SQLDatabase(engine)
    elif db_uri == MYSQL:
        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            st.error("Please provide all MySQL connection details.")
            st.stop()

        # Split host and port
        if ":" in mysql_host:
            host, port = mysql_host.split(":")
            port = int(port)
        else:
            host = mysql_host
            port = 3306

        connection_url = URL.create(
            drivername="mysql+mysqlconnector",
            username=mysql_user,
            password=mysql_password,
            host=host,
            port=port,
            database=mysql_db,
        )

        engine = create_engine(connection_url)

        return SQLDatabase(engine) 

if db_uri == MYSQL:
    db = configure_db(db_uri, mysql_host, mysql_user, mysql_password, mysql_db)
else:
    db=configure_db(db_uri)

## toolkit
toolkit = SQLDatabaseToolkit(
    db=db,
    llm=llm,
)

agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True,
)

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role":"assistant", "content":"How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
user_query = st.chat_input(placeholder="Ask anything from the database")

if user_query:
    st.session_state.messages.append(
        {"role": "user", "content": user_query}
    )
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):

        # Container for intermediate agent reasoning
        thought_container = st.container()

        st_cb = StreamlitCallbackHandler(
            thought_container,
            expand_new_thoughts=True,
            collapse_completed_thoughts=False,
        )

        response = agent_executor.invoke(
            {"input": user_query},
            config={
                "callbacks": [st_cb]
            },
        )

        answer = response["output"]

        st.write(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )