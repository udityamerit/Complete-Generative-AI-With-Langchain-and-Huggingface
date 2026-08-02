import streamlit as st
import os
import time

from dotenv import load_dotenv

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

# -------------------------------
# Load Environment Variables
# -------------------------------

load_dotenv()

api_key = os.getenv("NVIDIA_NIM_API_KEY")

if not api_key:
    st.error("❌ NVIDIA_NIM_API_KEY not found in .env file.")
    st.stop()

os.environ["NVIDIA_NIM_API_KEY"] = api_key

# -------------------------------
# Initialize LLM
# -------------------------------

llm = ChatNVIDIA(
    model="nvidia/nemotron-3-ultra-550b-a55b"
)

# -------------------------------
# Create Vector Embeddings
# -------------------------------

def vector_embedding():

    if "vectors" not in st.session_state:

        with st.spinner("Creating Vector Store..."):

            st.session_state.embeddings = NVIDIAEmbeddings()

            st.session_state.loader = PyPDFDirectoryLoader("../us_census")

            st.session_state.docs = st.session_state.loader.load()

            st.session_state.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=700,
                chunk_overlap=50
            )

            # ✅ FIXED
            st.session_state.final_documents = (
                st.session_state.text_splitter.split_documents(
                    st.session_state.docs[:30]
                )
            )

            st.session_state.vectors = FAISS.from_documents(
                st.session_state.final_documents,
                st.session_state.embeddings
            )

# -------------------------------
# UI
# -------------------------------

st.title("ChatNVIDIA NIM Demo")

prompt = ChatPromptTemplate.from_template(
    """
Answer the question based only on the provided context.

If the answer is not found in the context,
say "I couldn't find the answer in the provided documents."

<context>
{context}
</context>

Question:
{input}

Answer:
"""
)

prompt1 = st.text_input(
    "Enter your question from the document"
)

# -------------------------------
# Create Embeddings Button
# -------------------------------

if st.button("Document Embedding"):

    vector_embedding()

    st.success("✅ FAISS Vector Store DB is Ready using NVIDIA Embeddings")

# -------------------------------
# Question Answering
# -------------------------------

if prompt1:

    # Prevent error if embeddings are not created
    if "vectors" not in st.session_state:
        st.warning("⚠ Please click 'Document Embedding' first.")
        st.stop()

    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    retriever = st.session_state.vectors.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10
        }
    )

    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    start = time.perf_counter()

    response = retrieval_chain.invoke(
        {
            "input": prompt1
        }
    )

    end = time.perf_counter()

    st.subheader("Answer")

    st.write(response["answer"])

    st.info(f"⏱ Response Time: {end-start:.2f} seconds")

    with st.expander("Document Similarity Search"):

        for i, doc in enumerate(response["context"], start=1):

            st.markdown(f"### Document {i}")

            st.write(doc.page_content)

            st.write("--------------------------------")