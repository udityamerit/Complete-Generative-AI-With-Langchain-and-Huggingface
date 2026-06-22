import streamlit as st
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)

# Load environment variables
load_dotenv()

# Page Config
st.set_page_config(page_title="Conversational RAG", page_icon="📚")
st.title("📚 Conversational RAG with PDF Uploads")
st.write("Upload PDFs and chat with their contents.")

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Groq API Key
api_key = st.text_input(
    "Enter your Groq API Key",
    type="password"
)

if not api_key:
    st.warning("Please enter your Groq API Key.")
    st.stop()

# LLM
llm = ChatGroq(
    groq_api_key=api_key,
    model="llama-3.3-70b-versatile"
)

# Session ID
session_id = st.text_input(
    "Session ID",
    value="default_session"
)

# Initialize chat history store
if "store" not in st.session_state:
    st.session_state.store = {}

# Upload PDFs
uploaded_files = st.file_uploader(
    "Upload PDF Files",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    documents = []

    with st.spinner("Processing PDFs..."):

        for uploaded_file in uploaded_files:

            temp_pdf = f"temp_{uploaded_file.name}"

            with open(temp_pdf, "wb") as f:
                f.write(uploaded_file.getvalue())

            loader = PyPDFLoader(temp_pdf)
            docs = loader.load()

            documents.extend(docs)

            os.remove(temp_pdf)

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        splits = text_splitter.split_documents(documents)

        # Vector Store
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings
        )

        retriever = vectorstore.as_retriever()

    st.success("PDFs processed successfully!")

    # Contextualization Prompt
    contextualize_q_system_prompt = """
    Given a chat history and the latest user question
    which might reference context in the chat history,
    formulate a standalone question that can be understood
    without the chat history.

    Do NOT answer the question.
    Simply reformulate it if needed.
    """

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_q_prompt,
    )

    # QA Prompt
    system_prompt = """
    You are an assistant for question-answering tasks.

    Use the following retrieved context to answer the question.

    If you do not know the answer, say:
    "I don't know based on the provided documents."

    Keep the answer concise and limited to three sentences.

    Context:
    {context}
    """

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(
        llm,
        qa_prompt
    )

    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        question_answer_chain
    )

    # Session History Function
    def get_session_history(
        session: str,
    ) -> BaseChatMessageHistory:

        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()

        return st.session_state.store[session]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    # User Question
    user_input = st.text_input(
        "Ask a question about your PDFs:"
    )

    if user_input:

        with st.spinner("Thinking..."):

            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {
                        "session_id": session_id
                    }
                },
            )

        st.subheader("Answer")
        st.write(response["answer"])

        # Show Chat History
        session_history = get_session_history(session_id)

        with st.expander("Chat History"):
            for msg in session_history.messages:
                st.write(f"**{msg.type.upper()}**: {msg.content}")