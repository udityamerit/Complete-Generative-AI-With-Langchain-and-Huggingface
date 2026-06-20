import streamlit as st
import os
import time

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Load environment variables
load_dotenv()

# Set Groq API Key
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = groq_api_key

# Initialize LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="qwen/qwen3-32b"
)

# Prompt Template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based only on the provided context.

    <context>
    {context}
    </context>

    Question: {input}
    """
)

# Streamlit UI
st.title("📚 RAG Document Q&A with Groq Qwen3-32B")

# Function to create embeddings and vector store
def create_vector_embedding():
    if "vectors" not in st.session_state:

        with st.spinner("Loading documents and creating embeddings..."):

            # Embedding Model
            st.session_state.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            # Load PDFs
            loader = PyPDFDirectoryLoader("Resource")
            docs = loader.load()

            if not docs:
                st.error("No PDF files found inside Resource folder.")
                return

            # Split Documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            final_documents = text_splitter.split_documents(docs)

            # Create Vector Store
            st.session_state.vectors = FAISS.from_documents(
                final_documents,
                st.session_state.embeddings
            )

            st.success(
                f"Vector Database Ready! Created {len(final_documents)} chunks."
            )

# Button to create embeddings
if st.button("Create Document Embeddings"):
    create_vector_embedding()

# User Query
user_prompt = st.text_input(
    "Enter your question from the research paper"
)

# Question Answering
if user_prompt:

    # Check if embeddings exist
    if "vectors" not in st.session_state:
        st.warning(
            "Please click 'Create Document Embeddings' first."
        )
        st.stop()

    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    retriever = st.session_state.vectors.as_retriever()

    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    start = time.process_time()

    response = retrieval_chain.invoke(
        {"input": user_prompt}
    )

    elapsed_time = time.process_time() - start

    st.subheader("Answer")
    st.write(response["answer"])

    st.write(f"⏱ Response Time: {elapsed_time:.2f} seconds")

    # Show Retrieved Chunks
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"], start=1):
            st.markdown(f"### Chunk {i}")
            st.write(doc.page_content)
            st.write("---")