import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import autogen

# Load the environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

llm_config = {"model": "gpt-3.5-turbo"}

def reflection_message(recipient, messages, sender, config):
    return f'''Review the following content. 
            \n\n {recipient.chat_messages_for_summary(sender)[-1]['content']}'''


def create_nested_agents():
    legal_reviewer = autogen.AssistantAgent(
        name="LegalReviewer",
        llm_config=llm_config,
        system_message="You are a legal reviewer, known for "
                       "your ability to ensure that content is legally compliant "
                       "and free from any potential legal issues. "
                       "Make sure your suggestion is concise (within 3 bullet points), "
                       "concrete and to the point. "
                       "Begin the review by stating your role.",
    )
    ethics_reviewer = autogen.AssistantAgent(
        name="EthicsReviewer",
        llm_config=llm_config,
        system_message="You are an ethics reviewer, known for "
                       "your ability to ensure that content is ethically sound "
                       "and free from any potential ethical issues. "
                       "Make sure your suggestion is concise (within 3 bullet points), "
                       "concrete and to the point. "
                       "Begin the review by stating your role. ",
    )
    meta_reviewer = autogen.AssistantAgent(
        name="MetaReviewer",
        llm_config=llm_config,
        system_message="You are a meta reviewer, you aggregate and review "
                       "the work of other reviewers and give a final suggestion on the content.",
    )

    return legal_reviewer,ethics_reviewer,meta_reviewer


if __name__ == "__main__":
    task = """How Generative AI can help heal mental health. Make sure the blogpost is under 100 words"""

    writer = autogen.AssistantAgent(
        name="Writer",
        system_message="""You are a blog writer. You write reliable and concise blogs with title about a given topic.
        You must polish your blog based on the feedback you receive and give a refined version.
        Only return your final work without additional comments""",
        llm_config=llm_config
    )

    reply = writer.generate_reply(messages = [{"content": task, "role": "user"}])
    print("ORIGINAL REPLY: ",reply)


    critic = autogen.AssistantAgent(
        name="Critic",
        is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
        llm_config=llm_config,
        system_message="You are a critic. You review the work of "
                       "the writer and provide constructive "
                       "feedback to help improve the quality of the content.",
    )

    # response_critic = critic.initiate_chat(recipient=writer,
    #                                        message=task,
    #                                        max_turns=2,
    #                                        summary_method="last_msg")

    legal_reviewer,ethics_reviewer,meta_reviewer = create_nested_agents()

    review_chats = [
        {
            "recipient": legal_reviewer, "message": reflection_message,
            "summary_method": "reflection_with_llm",
            "summary_args": {"summary_prompt":
                                 "Return review into as JSON object only:"
                                 "{'Reviewer': '', 'Review': ''}.", },
            "max_turns": 1
        },
        {
            "recipient": ethics_reviewer, "message": reflection_message,
         "summary_method": "reflection_with_llm",
         "summary_args": {"summary_prompt":
                              "Return review into as JSON object only:"
                              "{'reviewer': '', 'review': ''}", },
         "max_turns": 1
        },
        {
            "recipient": meta_reviewer,
         "message": "Aggregate feedback from all reviewers and give final suggestions on the writing.",
         "max_turns": 1
        }
    ]

    critic.register_nested_chats(review_chats, trigger=writer)

    res = critic.initiate_chat(recipient=writer,
                               message=task,
                               max_turns=2,
                               summary_method="last_msg")

    print("######### PRINTING REVIEWED BLOG ##########")
    print(res.summary)

