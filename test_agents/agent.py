import os
import sys
import logging

sys.path.append("..")
from dotenv import load_dotenv
import google.cloud.logging
from google.adk import Agent
from google.genai import types
from typing import Optional, List, Dict

from google.adk.tools.tool_context import ToolContext
from google.adk.tools.load_artifacts_tool import load_artifacts_tool

import pdfplumber


load_dotenv()


# Tools (add the tool here when instructed)
def save_pdf_content_to_state(
        tool_context: ToolContext,
        file_name: str
) -> str:
    """Saves the PDF text to state.

    Args:
        pdf_text str: a list of strings String containing PDF content

    Returns:
        None
    """
    print("!!! File name",file_name)
    text = ""
    with pdfplumber.open(file_name) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""

    tool_context.state["pdf_text"] = text

    # A best practice for tools is to return a status message in a return dict
    return text


# Agents

info_extractor = Agent(
    name="cover_letter_writer",
    model=os.getenv("MODEL"),
    description="Extracts specific information from the input text",
    instruction="""
        - From the input PDF content text extract the following information:
            - Name: Name of the person in the text
            - Core Skills: Core skills of the person
            - Latest Title: Latest title held by the person
        """,


)

root_agent = Agent(
    name="steering",
    model=os.getenv("MODEL"),
    description="Read a PDF file and extract specific information.",
    instruction="""
        - Accept as input name of a PDF file
        - Use your tools to read the PDF and store it in tool context
        - Hand off to 'info_extractor' and print the information extracted by this agent as bullet points
        """,
    generate_content_config=types.GenerateContentConfig(
        temperature=0.9,
    ),
    # Add the sub_agents parameter when instructed below this line
    tools=[save_pdf_content_to_state],
    sub_agents=[info_extractor]

)

#test_agents/Suruchi_Keswani.pdf