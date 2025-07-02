import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Step 1: Load documents (replace with your data source)
documents = [
    "The capital of France is Paris.",
    "Python is a popular programming language.",
    "The Moon orbits the Earth every 27.3 days.",
    "OpenAI develops advanced AI models like GPT-4.",
]

# Step 2: Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)
chunks = text_splitter.create_documents(documents)

# Step 3: Initialize OpenAI Embeddings (Ada-002)
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=openai_api_key
)

# Step 4: Create Chroma vector store
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"  # Saves locally
)
# (To load an existing store: `vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)`)

# Step 5: Set up retrieval-based QA chain
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=openai_api_key
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True
)

# Step 6: Query the RAG system
query = "What is the capital of France?"
result = qa_chain({"query": query})

print("Answer:", result["result"])
print("Sources:", [doc.page_content for doc in result["source_documents"]])