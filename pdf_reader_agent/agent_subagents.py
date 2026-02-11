import os

import pdfplumber
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.tools.tool_context import ToolContext

load_dotenv()


def save_pdf_content_to_state(
        tool_context: ToolContext,
        file_name: str
) -> dict[str, str]:
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
    return {
        "status":"Success"
    }

reader_agent = Agent(
    name="reader_agent",
    description="Extract the text from an input PDF",
    instruction="""
    - You are an assistant that reads a PDF file and extracts its text for further use
    - Use the 'save_pdf_content_to_state' tool to extract the text from the PDF and save it to tool_context
    - Remember to pass the entire file name as received in single quotes to this tool
    """,
    model=os.getenv("MODEL"),
    tools=[save_pdf_content_to_state]
)

info_extractor_agent = Agent(
    name="info_extractor_agent",
    description="Extract the specific information from the text",
    instruction="""
    - You are an assistant that can extract the specified information from a given text
    - Read through the entire text from {pdf_text? } and use this information to extract the following information:
        - Name: name of the person mentioned in the document
        - Contact Details: contact details mentioned
        - Latest Title: latest title held by the person
    - Return this information as your response
    """,
    model=os.getenv("MODEL")
)


root_agent = Agent(
    name="pdf_query_agent",
    description="Agent to take as input a user document and extract specific info from it",
    instruction="""
        - You are an assistant that accepts a file path of a document from the user 
        and extracts specific info from it
        - Call your sub_agents to load the file from the entered file path and 
        the extract required information from it
        - Remember to pass the entire file name as received in single quotes
        - Once you have the extracted information from 'info_extractor_agent', 
        present it as a well-formatted response
        """,
    model=os.getenv("MODEL"),
    sub_agents=[reader_agent,info_extractor_agent]

)