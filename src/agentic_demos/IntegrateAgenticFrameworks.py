from dotenv import load_dotenv
import os
import autogen
from langchain_core.messages import convert_to_openai_messages, BaseMessage
from langgraph.func import entrypoint, task
from langgraph.graph import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

config_list = [{"model": "gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]}]
llm_config = {
    "timeout": 600,
    "cache_seed": 42,
    "config_list": config_list,
    "temperature": 0,
}

autogen_agent = autogen.AssistantAgent(
    name="Assistant",
    llm_config=llm_config
)

user_proxy_agent = autogen.UserProxyAgent(
    name="User_Proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "work_dir": "web",
        "use_docker": False,
    },
    llm_config=llm_config,
    system_message="Reply TERMINATE if the task has been solved at full satisfaction. Otherwise, reply CONTINUE, or the reason why the task is not solved yet.",
)

@task
def call_autogen_agent(messages: list[BaseMessage]):
    # convert to openai-style messages
    messages = convert_to_openai_messages(messages)
    response = user_proxy_agent.initiate_chat(
        autogen_agent,
        message=messages[-1],
        # pass previous message history as context
        carryover=messages[:-1],
    )
    # get the final response from the agent
    content = response.chat_history[-1]["content"]
    return {"role": "assistant", "content": content}


# add short-term memory for storing conversation history
checkpointer = MemorySaver()


@entrypoint(checkpointer=checkpointer)
def workflow(messages: list[BaseMessage], previous: list[BaseMessage]):
    messages = add_messages(previous or [], messages)
    response = call_autogen_agent(messages).result()
    return entrypoint.final(value=response, save=add_messages(messages, response))

# pass the thread ID to persist agent outputs for future interactions
# highlight-next-line
config = {"configurable": {"thread_id": "2"}}

for chunk in workflow.stream(
[
        {
            "role": "user",
            "content": "Which numbers between 1 and 50 are divisible by 7?",
        }
    ],
    # highlight-next-line
    config,
):
    print(chunk)




