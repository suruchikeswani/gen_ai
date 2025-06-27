from agents import Agent,InputGuardrail,GuardrailFunctionOutput,Runner
from marshmallow.fields import Boolean
from pydantic import BaseModel
import streamlit as st
import time
import os
from dotenv import load_dotenv
from agents.tracing.setup import GLOBAL_TRACE_PROVIDER
from agents import set_tracing_export_api_key
import asyncio
import nest_asyncio
nest_asyncio.apply()

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

st.title("Locally deployed Open AI Agents")
st.write("This is a locally deployed AI agent that can help you with your research questions.")

add_selectbox = st.sidebar.selectbox(
    "Select the LLM provider",
    (["OpenAI"])
)


class ResearchOutput(BaseModel):
    is_research:bool
    reasoning:str

guardrail_agent = Agent(
    name="GuardrailCheck",
    instructions="Check if user is asking about research",
    output_type=ResearchOutput
)

cs_agent = Agent(
    name="CSResearcher",
    handoff_description="Specialist agent for Computer Science Research",
    instructions="You provide help with CS research. Explain your reasoning at each step and include examples"
)

bio_agent = Agent(
    name="BioResearcher",
    handoff_description="Specialist agent for research in Biology",
    instructions="You assist with biological research. Explain important events and context clearly."
)

async def research_guardrail(ctx,agent,input_data):
    print("Entered research_guardrail")
    result = await Runner.run(guardrail_agent,input_data,context=ctx.context)
    final_output=result.final_output_as(ResearchOutput)
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_research
    )

triage_agent=Agent(
    name="TriageAgent",
    instructions="You determine which agent to use based on the user's research question",
    handoffs=[cs_agent,bio_agent],
    input_guardrails=[
        InputGuardrail(guardrail_function=research_guardrail)
    ]
)

set_tracing_export_api_key(os.getenv("OPENAI_API_KEY"))
GLOBAL_TRACE_PROVIDER._multi_processor.force_flush()

async def main():
    user_input=st.text_input("Enter your research question:")
    if st.button("Submit"):
        with st.spinner("Processing..."):
            time.sleep(10)
        result = await Runner.run(triage_agent, user_input)
        st.write(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())