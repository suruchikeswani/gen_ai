import os

from dotenv import load_dotenv
# {faithfulness, answer_relevancy, context_precision, context_recall}
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# Load the environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Read data from a URL and vectorize it
url="https://arxiv.org/pdf/1706.03762"  #Attention is all you need paper
# url_loader = UnstructuredURLLoader(urls=[url])
# urls = url_loader.load()

# Read data from a PDF and vectorize it
pdf_loader = PyPDFLoader("Suruchi_Keswani.pdf")
pdfs = pdf_loader.load()

# Combine the documents from both sources
documents = []
# documents.extend(urls)
documents.extend(pdfs)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)
print(f"Total number of documents: {len(texts)}")
print("Sample document content:", texts[0].page_content[:200])  # Print first 200 characters of the first document

pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )

embeddings = OpenAIEmbeddings()
index_name="pinecone-demo-index"
if index_name not in pc.list_indexes().names():
    pc.create_index(name=index_name,
                          dimension=len(embeddings.embed_query("test")),metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),)

vectorstore_from_texts = PineconeVectorStore.from_documents(
        texts,
        index_name=index_name,
        embedding=embeddings
    )


# docsearch = pc.from_documents(texts, embeddings, index_name=index_name)
