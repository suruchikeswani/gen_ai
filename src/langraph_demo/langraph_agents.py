import os
from dotenv import load_dotenv
import random
from typing_extensions import Literal
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.func import task
from langgraph.func import entrypoint
from langgraph.graph import add_messages
from langgraph.types import interrupt, Command
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
import uuid

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

@tool
def get_career_paths():
    """Suggest career paths based on general user interest"""
    return random.choice(["data science", "product management", "cybersecurity"])

@tool
def get_learning_resources(career: Literal["data science", "product management", "cybersecurity"]):
    """Suggest learning resources based on selected career path"""
    return {
        "data science": ["Coursera: IBM Data Science", "edX: Harvard's Data Science Series"],
        "product management": ["Udemy: Become a Product Manager", "Reforge Programs"],
        "cybersecurity": ["Cybrary", "CompTIA Security+ Certification"],
    }[career]

@tool(return_direct=True)
def transfer_to_education_advisor():
    """Ask education advisor for help"""
    return "Successfully transferred to education advisor."

@tool(return_direct=True)
def transfer_to_career_advisor():
    """Ask career advisor for help"""
    return "Successfully transferred to career advisor."

def create_agents():
    model = ChatOpenAI(model="gpt-4o")

    career_advisor = create_react_agent(model,tools=[get_career_paths, transfer_to_education_advisor],
                                        prompt=(
                                            "You are a career expert. Help users explore career options. "
            "If they ask about courses or education, transfer to the education advisor. "
            "Always explain your reasoning before transferring."))


    education_advisor = create_react_agent(model,tools=[get_learning_resources, transfer_to_career_advisor],
                                        prompt=("You are an education expert. Recommend learning paths for specific careers. "
            "If the user changes their career preference, transfer back to the career advisor. "
            "Always explain your reasoning before transferring."))
    return career_advisor, education_advisor

career_advisor, education_advisor = create_agents()

@task
def call_career_advisor(messages):
    return career_advisor.invoke({"messages": messages})["messages"]

@task
def call_education_advisor(messages):
    return education_advisor.invoke({"messages": messages})["messages"]



checkpointer = MemorySaver()

def string_to_uuid(text):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, text))

@entrypoint(checkpointer=checkpointer)
def multi_turn_graph(messages, previous):
    previous = previous or []
    messages = add_messages(previous, messages)

    call_active_agent = call_career_advisor  # start with career advisor

    while True:
        agent_messages = call_active_agent(messages).result()
        messages = add_messages(messages, agent_messages)

        # Find the last AI message
        ai_msg = next(m for m in reversed(agent_messages) if isinstance(m, AIMessage))

        # If no tool was called, wait for user input
        if not ai_msg.tool_calls:
            user_input = interrupt(value="Ready for user input.")
            messages = add_messages(messages, [{
                "role": "user",
                "content": user_input,
                "id": string_to_uuid(user_input),
            }])
            continue

        # Check if the agent is transferring to the other advisor
        tool_call = ai_msg.tool_calls[-1]
        if tool_call["name"] == "transfer_to_education_advisor":
            call_active_agent = call_education_advisor
        elif tool_call["name"] == "transfer_to_career_advisor":
            call_active_agent = call_career_advisor
        else:
            raise ValueError(f"Unexpected tool: {tool_call['name']}")

if __name__ == '__main__':
    thread_config = {"configurable": {"thread_id": uuid.uuid4()}}

    inputs = [
        {
            "role": "user",
            "content": "I'm interested in technology but not sure what career fits me.",
            "id": str(uuid.uuid4()),
        },
        Command(resume="That sounds good. What courses should I take to get started?"),
        Command(resume="Awesome! Are these resources beginner friendly?"),
    ]

    for idx, user_input in enumerate(inputs):
        print(f"\n--- Conversation Turn {idx + 1} ---\n")
        print(f"User: {user_input}\n")
        for update in multi_turn_graph.stream(
                user_input,
                config=thread_config,
                stream_mode="updates",
        ):
            for node_id, value in update.items():
                if isinstance(value, list) and value:
                    last_msg = value[-1]
                    if isinstance(last_msg, dict) or last_msg.type != "ai":
                        continue
                    print(f"{node_id}: {last_msg.content}")

