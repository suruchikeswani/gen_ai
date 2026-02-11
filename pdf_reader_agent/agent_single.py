from typing import Any, Coroutine

from google.adk.agents import Agent
from google.adk.tools.tool_context import ToolContext
from google.adk.tools import load_artifacts
from google.genai import types
import os
from pathlib import Path


async def save_report_artifacts(tool_context: ToolContext, file_path: str):
    """
    Tool that takes as input a PDF file path or folder path and saves the files as
    artifacts to the tool context
    :param tool_context:
    :param file_path:
    :param filename:
    :return:
    """
    print("!!!!Entered save_report_artifacts: ", file_path)
    print("Is Dir?? ",os.path.isdir(file_path))
    # Check if file_path is a directory
    if os.path.isdir(file_path):
        # Read all PDF files from the directory
        pdf_files = list(Path(file_path).glob('*.pdf'))

        if not pdf_files:
            raise ValueError(f"No PDF files found in directory: {file_path}")

        # Process each PDF file
        for pdf_file in pdf_files:
            with open(pdf_file, 'rb') as f:
                pdf_bytes = f.read()

            # Save as a PDF artifact
            artifact_part = types.Part(
                inline_data=types.Blob(mime_type='application/pdf', data=pdf_bytes)
            )
            await tool_context.save_artifact(pdf_file.name, artifact_part)
    else:
        # Read the single file if directory doesn't exist
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File or directory not found: {file_path}")

        with open(file_path, 'rb') as f:
            pdf_bytes = f.read()

        # Save as a PDF artifact
        print("!!! Getting artifact")
        artifact_part = types.Part(
            inline_data=types.Blob(mime_type='application/pdf', data=pdf_bytes)
        )
        print("!!! Saving artifact ", file_path)
        await tool_context.save_artifact(file_path, artifact_part)


# 1. Define the tool to query the artifact
async def query_pdf_artifact(filename: str, tool_context: ToolContext) -> (
        dict[str, str]):
    """Retrieves content from a stored PDF artifact to answer questions."""

    # Load the artifact from the context
    artifact = await tool_context.load_artifact(filename=filename)

    if not artifact or not artifact.inline_data:
        return f"Error: Could not find artifact '{filename}'."

    # For simple cases, you can return the data or process it
    # Note: Gemini can process 'application/pdf' bytes directly if passed back
    return {
        "content_type": artifact.inline_data.mime_type,
        "data": artifact.inline_data.data  # Binary PDF data
    }



root_agent = Agent(
    name="pdf_reader",
    description="This is my first agent",
    instruction="""
        - You are an assistant that manages artifacts and answers questions based on them.
        - If the user asks to save a document, call 'save_report_artifacts' to save the files as an artifacts.
        - You don't need to ask for the filename from the user and please keep the filename as the original file while saving.
        - Once the file is saved as an artifact use 'load_artifacts_tool' to load the file name
        - You can now read the artifact from tool_context and answer the users questions
        """,
    model="gemini-2.0-flash",
    tools=[save_report_artifacts,load_artifacts]

)