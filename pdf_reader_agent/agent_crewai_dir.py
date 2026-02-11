import os

from crewai_tools import DirectorySearchTool
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.tools.crewai_tool import CrewaiTool

load_dotenv()

root_agent = Agent(
    name="pdf_query_agent",
    description="Agent to answer users questions based on input document",
    instruction="""
        - You are an assistant that has the capability to answer the users questions based on 
        contents of a PDF file
        - Ask the user what they want to know about the PDF file
        - Use your tool to search for the answers to the users question in the PDF file given to it
        """,
    model=os.getenv("MODEL"),
    tools = [
            CrewaiTool(
                name="search_info_in_PDF",
                description=(
                    """Searches for the answers to the user's question in all PDFs present in
                    the provided directory"""
                ),
                tool = DirectorySearchTool(directory='pdf_reader_agent')
            )
        ]

)