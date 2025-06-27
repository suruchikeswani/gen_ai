from openai import OpenAI
from dotenv import load_dotenv
import os
import static_prompt

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",  # or "gpt-4",
    messages=[
        {"role": "system", "content": "You are an banking expert and "
                                      "your task is to classify bank transactions based on their context"},
        {"role": "user", "content": static_prompt.prompt}
    ],
    max_tokens=150,
    temperature=0.0,
)

# Print the response content
print(response.choices[0].message.content.strip())
