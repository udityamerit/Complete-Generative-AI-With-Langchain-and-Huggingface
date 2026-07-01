import streamlit as st
import validators

from urllib.parse import urlparse, parse_qs

from youtube_transcript_api import YouTubeTranscriptApi

from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredURLLoader

from langchain_classic.prompts import PromptTemplate
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_huggingface import HuggingFaceEndpoint

st.set_page_config(page_title="YT & Website Summarizer")

st.title("YouTube & Website Summarizer")

with st.sidebar:
    hf_api_key = st.text_input("Huggingface  Key", type="password")

url = st.text_input("Enter URL")


def get_video_id(url):
    parsed = urlparse(url)

    if parsed.hostname == "youtu.be":
        return parsed.path[1:]

    if parsed.hostname in (
        "www.youtube.com",
        "youtube.com",
        "m.youtube.com",
    ):
        if parsed.path == "/watch":
            return parse_qs(parsed.query)["v"][0]

        if parsed.path.startswith("/shorts/"):
            return parsed.path.split("/")[2]

    return None


repo_id = "Qwen/Qwen2.5-7B-Instruct-1M"

if st.button("Summarize"):

    if not hf_api_key:
        st.error("Enter huggingface API Key")
        st.stop()

    if not validators.url(url):
        st.error("Enter valid URL")
        st.stop()

    llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    task="conversational",  
    huggingfacehub_api_token=hf_api_key,
    max_new_tokens=150, 
    temperature=0.7
)

    prompt = PromptTemplate(
        template="""
Summarize the following content in around 500 words with their important benefits and the points as well.

{text}
""",
        input_variables=["text"],
    )

    try:

        if "youtube.com" in url or "youtu.be" in url:

            video_id = get_video_id(url)

            transcript = YouTubeTranscriptApi.get_transcript(video_id)

            text = " ".join([x["text"] for x in transcript])

            docs = [Document(page_content=text)]

        else:

            loader = UnstructuredURLLoader(
                urls=[url],
                ssl_verify=False,
                headers={
                    "User-Agent": "Mozilla/5.0"
                },
            )

            docs = loader.load()

        chain = load_summarize_chain(
            llm,
            chain_type="stuff",
            prompt=prompt,
        )

        result = chain.invoke(
            {"input_documents": docs}
        )

        st.subheader("Summary")

        if isinstance(result, dict):
            st.write(result["output_text"])
        else:
            st.write(result)

    except Exception as e:
        st.exception(e)