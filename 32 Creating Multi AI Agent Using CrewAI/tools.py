from crewai_tools import YoutubeChannelSearchTool
from crewai_tools.rag.data_types import DataType

# Monkeypatch to fix a bug in crewai_tools 1.15.x where it prepends '@' to URLs
def patched_add(self, youtube_channel_handle: str) -> None:
    # Call the parent RagTool.add directly without modifying the URL
    super(YoutubeChannelSearchTool, self).add(youtube_channel_handle, data_type=DataType.YOUTUBE_CHANNEL)

YoutubeChannelSearchTool.add = patched_add
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from crewai.rag.embeddings.providers.custom.embedding_callable import CustomEmbeddingFunction
import chromadb.api.types

class ChromaNVIDIAEmbeddings(CustomEmbeddingFunction, chromadb.api.types.EmbeddingFunction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lc_embedder = NVIDIAEmbeddings(
            model="nvidia/nv-embedqa-e5-v5",
            api_key=os.environ.get("NVIDIA_API_KEY"),
            truncate="END"
        )
    
    def __call__(self, input):
        # LangChain's embed_documents properly handles the input_type="passage" requirement
        return self.lc_embedder.embed_documents(input)

# Initialize the tool for general YouTube channel searches using the full URL
yt_tool = YoutubeChannelSearchTool(
    youtube_channel_handle='https://www.youtube.com/c/KGPTalkie',
    config={
        "embedding_model": {
            "provider": "custom",
            "config": {
                "embedding_callable": ChromaNVIDIAEmbeddings
            }
        }
    }
)
