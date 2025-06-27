import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import prompt_template

# Load the environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI Chat model with LangChain
chat = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo", temperature=0.7)


# Initialize LangChain with the prompt template and Chat model
prompt = PromptTemplate(input_variables=["desc","merch"], template=prompt_template.prompt)
llm_chain = LLMChain(prompt=prompt, llm=chat)

# Example user input
desc="Hospital fee of  INR 61057.61 for emergency services via MedPlus on 16-Mar-2025, Bangalore."
merch="MedPlus"

# Generate the response
response = llm_chain.run({
    "desc": desc,
    "merch": merch
})

# Print the response
print(response)
