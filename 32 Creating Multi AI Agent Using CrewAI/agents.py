import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
from crewai import Agent
from langchain_openai import ChatOpenAI
from tools import yt_tool

# Initialize the NVIDIA NIM LLM via ChatOpenAI
nim_llm = ChatOpenAI(
    model="meta/llama3-70b-instruct",
    api_key=os.environ.get("NVIDIA_API_KEY"),
    base_url="https://integrate.api.nvidia.com/v1"
)

# creating a senior blog content researcher

blog_researcher = Agent(
    role = 'Blog Researcher from Youtube Videos',
    goal = 'get the relavent video content from the topic {topic} from the yt channel',
    verbose = True,
    memory = True,
    backstory = (
        "Expert in understanding Videos in AI Data Science, Machine Learning And Gen AI"
    ),
    tools=[yt_tool],
    allow_delegation=True,
    llm=nim_llm
)

## createing a senior blog writer agent with yt tool

blog_writer = Agent(
    role = 'Blog Writer from Youtube Videos',
    goal = 'Narrate compelling tech stories about the video {topic} from the yt channel',
    verbose = True,
    memory = True,
    backstory = (
        "with a flair for simplifying complex topics you craft"
        "engaging narrative that captivate and educate, bringing new"
        "discoveries to light in the accessible manner"
    ),
    tools=[yt_tool],
    allow_delegation=False,
    llm=nim_llm
)