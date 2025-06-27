import os

import streamlit as st
from dotenv import load_dotenv
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
# {faithfulness, answer_relevancy, context_precision, context_recall}
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# Load the environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )

embeddings = OpenAIEmbeddings()
index_name="pinecone-demo-index"

vector_store = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
retriever = vector_store.as_retriever()

llm = ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo", api_key=openai_api_key)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
)

initial_prompt = ("""You are a career counseller. Your job is to read the provided resume and 
provide the requested insights.""")

query = "Generate a LinkedIn headline for this candidate under 50 words."
result = qa({"query": query, "prompt": initial_prompt})

print("Answer:", result['result'])



# To run: streamlit run src/rag/RAG_LlamaIndex_Pinecone_Retriever.py



st.title("RAG Retriever Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("You:", "")

if st.button("Send") and user_input:
    result = qa({"query": user_input, "prompt": initial_prompt})
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", result['result']))

for sender, message in st.session_state.chat_history:
    st.markdown(f"**{sender}:** {message}")

