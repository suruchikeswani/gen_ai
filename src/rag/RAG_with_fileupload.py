import streamlit as st
import tempfile
import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

# Load the environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

st.set_page_config(page_title="PDF Chat with LlamaIndex", layout="wide")
st.title("📄 Chat with Your PDF Document")

# Step 1: Upload a PDF file
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
print("UPLOADED FILE: ",uploaded_file)

if uploaded_file:
    print("Saving file temporarily")
    # Save file temporarily
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load and parse the document
        st.info("Loading document...")
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()

        splitter = SentenceSplitter(chunk_size=1024)
        nodes = splitter.get_nodes_from_documents(documents)

        Settings.llm = OpenAI(model="gpt-3.5-turbo")
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

        summary_index = SummaryIndex(nodes)
        vector_index = VectorStoreIndex(nodes)

        print("Created indexes")

        summary_query_engine = summary_index.as_query_engine(
            response_mode="tree_summarize",
            use_async=True,
        )
        vector_query_engine = vector_index.as_query_engine()

        summary_tool = QueryEngineTool.from_defaults(
            query_engine=summary_query_engine,
            description=(
                "Useful for summarization questions related to MetaGPT"
            ),
        )

        vector_tool = QueryEngineTool.from_defaults(
            query_engine=vector_query_engine,
            description=(
                "Useful for retrieving specific context from the MetaGPT paper."
            ),
        )

        query_engine = RouterQueryEngine(
            selector=LLMSingleSelector.from_defaults(),
            query_engine_tools=[
                summary_tool,
                vector_tool,
            ],
            verbose=True
        )
        st.session_state.index = vector_index
        st.success("✅ Document indexed and ready for Q&A!")

if "index" in st.session_state:
    print("!!Entered Session state")
    query_engine = st.session_state.index.as_query_engine()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Input + Button
    with st.container():
        user_input = st.text_input("💬 Ask a question about your document:")

        if st.button("Ask") and user_input:
            response = query_engine.query(user_input)
            st.session_state.chat_history.append((user_input, str(response)))

    # Display chat history
    st.markdown("### 🧠 Chat History")
    for question, answer in reversed(st.session_state.chat_history):
        st.markdown(f"**You:** {question}")
        st.markdown(f"**Bot:** {answer}")
        st.markdown("---")
else:
    print("Not entered")
    print("!!!!!!!!!!",st.session_state)