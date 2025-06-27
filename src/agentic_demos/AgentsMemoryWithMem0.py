import os
from dotenv import load_dotenv
import autogen
from autogen import ConversableAgent
from mem0 import MemoryClient

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ['MEM0_API_KEY'] = os.getenv('MEM0_API_KEY')

agent = ConversableAgent(name = "PatientAgent", llm_config={"config_list": [{"model": "gpt-4o",
                                                      "api_key": os.environ.get("OPENAI_API_KEY")}]},
                         code_execution_config=False,
                         function_map=None,
                         human_input_mode="NEVER",
                         )
memory = MemoryClient()

conversation = [
  {
    "role": "assistant",
    "content": "Hello, I’m MedCare's virtual assistant.\n\nThank you for trusting MedCare Health System. How can I assist you today?"
  },
  {
    "role": "user",
    "content": "I've been having recurring chest pain and would like to get admitted for observation."
  },
  {
    "role": "assistant",
    "content": "I’m really sorry to hear that you’re experiencing this. Your health is our priority, and I’m here to help you every step of the way.\n\nTo get started, may I have your full name and date of birth, so I can access your patient profile?"
  },
  {
    "role": "user",
    "content": "Name: John Doe, DOB: 03/14/1985"
  },
  {
    "role": "assistant",
    "content": "Thank you, John. I’ve located your record in our system.\n\nBefore proceeding, may I also have the email address or phone number associated with your MedCare account for verification?"
  },
  {
    "role": "user",
    "content": "john.doe@email.com"
  },
  {
    "role": "assistant",
    "content": "Perfect, thank you for confirming your details.\n\nJohn, we take symptoms like chest pain very seriously. I can help you schedule a direct admission to our observation unit.\n\nMay I ask when you first began experiencing these symptoms and whether you have any history of heart-related conditions?"
  },
  {
    "role": "user",
    "content": "It started about three days ago. No previous heart conditions that I know of."
  },
  {
    "role": "assistant",
    "content": "Thanks for the information. I’ll include this in your intake form for the clinical team.\n\nSince this is a recurring issue and not an emergency at the moment, I recommend we schedule a priority admission tomorrow for observation and testing.\n\nWould you like me to go ahead and confirm the admission for tomorrow?"
  },
  {
    "role": "user",
    "content": "Yes, please."
  },
  {
    "role": "assistant",
    "content": "Thank you. I’m confirming an admission for tomorrow.\n\nOur patient coordinator will contact you within the next 2 hours with further instructions and to assist with pre-admission paperwork.\n\nIs there anything else I can help you with right now, John?"
  },
  {
    "role": "user",
    "content": "No, that's all. Thanks a lot!"
  },
  {
    "role": "assistant",
    "content": "You’re very welcome. Wishing you a quick recovery and we're here for you whenever you need us.\n\nTake care, John."
  }
]

user_id="patient_admission_bot"
memory.add(messages=conversation,user_id=user_id)

question = "What did the assistant advise me when I told him I have recurring chest pain?"

relevant_memories = memory.search(query=question,user_id=user_id)
flatten_relevant_memories="\n".join([m["memory"] for m in relevant_memories])

prompt = f"""Answer the user question considering the memories. Keep answers clear and concise.
Memories:
{flatten_relevant_memories}
\n\n
Question: {question}
"""

reply = agent.generate_reply(messages=[{"content": prompt, "role": "user"}])
print(reply)
