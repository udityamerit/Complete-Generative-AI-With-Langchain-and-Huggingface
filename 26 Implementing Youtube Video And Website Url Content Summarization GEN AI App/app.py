import validators, streamlit as st
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_classic.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

## Streamlit app
st.set_page_config(page_title="LangChain: Summarize Text From YT or Websites", page_icon="🦜")
st.title("🦜LangChain: Summarize Text From YT or Websites")
st.subheader("Summarize URL")


with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

if st.button("Summarize the content from YT or Website"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information about the api key")
elif not validators.url(generic_url):
    st.error("Please enter a validator Url. It can may be a YT video URL or Website URL")

else:
    try:
        with st.spinner("Waiting..."):
            if "youtube.com" in generic_url:
                loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
            else:
                loader = UnstructuredURLLoader()
                