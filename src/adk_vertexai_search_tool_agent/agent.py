import os
import sys
sys.path.append("..")
import google.cloud.logging
from ..util.callback_logging import log_query_to_model, log_model_response
from dotenv import load_dotenv

from google.adk import Agent
from google.adk.tools import AgentTool

from .tools import get_date

# Add the VertexAiSearchTool import below
from google.adk.tools import VertexAiSearchTool


load_dotenv()
cloud_logging_client = google.cloud.logging.Client()
cloud_logging_client.setup_logging()

# Create your vertexai_search_tool and update its path below
vertexai_search_tool = VertexAiSearchTool(
    search_engine_id="projects/qwiklabs-gcp-03-faa2bf65be78/locations/global/collections/default_collection/engines/planet-search_1770107178777"
)

vertexai_search_agent = Agent(
    name="vertexai_search_agent",
    model=os.getenv("MODEL"),
    instruction="Use your search tool to look up facts.",
    tools=[vertexai_search_tool]
)

root_agent = Agent(
    # A unique name for the agent.
    name="root_agent",
    # The Large Language Model (LLM) that agent will use.
    model=os.getenv("MODEL"),
    # A short description of the agent's purpose, so other agents
    # in a multi-agent system know when to call it.
    description="Answer questions using your data store access.",
    # Instructions to set the agent's behavior.
    instruction="You analyze new planet discoveries and engage with the scientific community on them.",
    # Callbacks to log the request to the agent and its response.
    before_model_callback=log_query_to_model,
    after_model_callback=log_model_response,
    # Add the tools instructed below
    tools=[AgentTool(vertexai_search_agent, skip_summarization=False), get_date]

)
