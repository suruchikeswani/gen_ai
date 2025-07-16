from typing import Dict, Any
import asyncio

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from google.adk.models.lite_llm import LiteLlm
from dotenv import load_dotenv
import pandas as pd
import os

from src.adk_agents.agent import session_service

APP_NAME = "perform_eda"
USER_ID = "skeswani"
SESSION_ID = "123"

load_dotenv()

def eda_tasks(csv_path :str) -> Dict[str,Any]:
    """Method to perform EDA on the CSV. This method has the following Input/Output parameters.
    Input:
    - path_to_csv: Relative path of the CSV containing the data to be analyzed
    Output:
    Dictionary containing the following:
    - Shape: Shape of the data
    - Summary: 5 point summary
    - Null_Counts: Null counts by column
    - value_counts: Unique valur counts for categorical columns"""
    data = pd.read_csv(csv_path)
    result = {}
    result["Shape"] = data.shape
    result["Summary"] = data.describe(include='all').to_string()
    result["Null_Counts"] = data.isnull().sum().to_string()
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    value_counts = {}
    for col in categorical_cols:
        value_counts[col] = data[col].value_counts().head().to_string()
    result['value_counts'] = value_counts
    return result

eda_agent = Agent(
    model=LiteLlm("openai/gpt-3.5-turbo"),
    name='eda_agent',
    instruction= 'As an agent, you will perform EDA on the received data',
    description='This agent will perform exploratory data analysis as suggested by the provided tool. '
                'Additionally, it will also suggest the different charts that can be used to get'
                'good insights on the data. Provide the Python code for any one of the charts',
    tools=[eda_tasks]
)

session_service = InMemorySessionService()
session = asyncio.run(
    session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID))
runner = Runner(agent=eda_agent,app_name=APP_NAME,session_service=session_service)

def run_agent(path_to_csv):
    """Calls EDA agent with path to CSV"""
    query = f"Perform EDA on the CSV located at: {path_to_csv}"
    content = types.Content(role='user', parts=[types.Part(text=query)])
    events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)
    for event in events:
        if event.is_final_response():
            final_response = event.content.parts[0].text
            print("Agent Response: ", final_response)

if __name__=="__main__":
    path_to_csv="sample.csv"
    if not os.path.exists(path_to_csv):
        print("FILE NOT FOUND!")
    else:
        run_agent(path_to_csv)



