from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext, Tool
from pydantic_ai.models.openai import OpenAIModel
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

model = OpenAIModel(
    'gpt-4o-mini'
)

class ResponseModel(BaseModel):
    """Automatic Structured response with metadata."""
    leader_name: str
    continent_name: str
    country_name: str
    capital_name: str
    leader_description: str = Field(description="leader description")


if __name__=="__main__":
    agent = Agent(
        model=model,
        result_type=ResponseModel,
        system_prompt="""You are an intelligent research agent. 
        Analyze user request carefully and provide structured responses.""",
        result_retries=3
    )

    agent2 = Agent(
        model=model,
        system_prompt="""You are an intelligent research agent. 
            Analyze input json list carefully and provide markdown table.
            Be concise and don't write anything else except the markdown table.
            use bold tags for headers""",
        result_retries=3
    )

    data_list=[]

    response = agent.run_sync("Tell me about Narendra Modi")
    data_list.append(response.output.model_dump_json(indent=2))

    response = agent.run_sync("tell me about Donald Trump")
    data_list.append(response.output.model_dump_json(indent=2))

    response = agent.run_sync("tell me about Xi Jinping")
    data_list.append(response.output.model_dump_json(indent=2))

    response_table = agent2.run_sync(str(data_list))
    print(response_table.output)


