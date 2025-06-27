import os
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex
from llama_index.readers.web import SimpleWebPageReader
from ragas import EvaluationDataset,SingleTurnSample
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
# {faithfulness, answer_relevancy, context_precision, context_recall}
from ragas.metrics._aspect_critic import harmfulness
from ragas.integrations.llama_index import evaluate
from langchain.document_loaders import UnstructuredURLLoader, PyPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter


# Load the environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Read data from a URL and vectorize it
url="https://arxiv.org/pdf/1706.03762"  #Attention is all you need paper

data = SimpleWebPageReader(html_to_text=True).load_data([url])

index=VectorStoreIndex.from_documents(data)
#     ,llm=OpenAI(model="gpt-3.5-turbo"),
#     embed_model=OpenAIEmbedding(model="text-embedding-ada-002")
# )

query_engine = index.as_query_engine()
# q1 = query_engine.query("What is the summary of the document?")
# print(q1)


#Evaluation of RAG
eval_questions = ["What is the main contribution of the paper?",
                  # "What is the architecture of the model proposed in the paper?",
                  # "What are the key results of the paper?",
                  "What are the limitations of the paper?"]

eval_answers = ["The main contribution of the paper is the introduction of the Transformer model, which uses self-attention mechanisms to process sequences of data more efficiently than previous models.",
                # "The architecture of the model proposed in the paper consists of an encoder and a decoder, both of which use self-attention mechanisms to process sequences of data.",
                # "The key results of the paper include improved performance on various natural language processing tasks, such as machine translation and language modeling, compared to previous models.",
                "The limitations of the paper include the need for large amounts of data to train the model and the difficulty in interpreting the self-attention mechanisms used in the model."]


sample1 = SingleTurnSample(
    user_input="What is the main contribution of the paper?",
    reference="The main contribution of the paper is the introduction of the Transformer model, which uses self-attention mechanisms to process sequences of data more efficiently than previous models."

)

sample2 = SingleTurnSample(
    user_input="What are the limitations of the paper?",
    reference="The limitations of the paper include the need for large amounts of data to train the model and the difficulty in interpreting the self-attention mechanisms used in the model."

)
dataset = EvaluationDataset(samples=[sample1, sample2])
results = evaluate(
    query_engine=query_engine,
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)

# metrics = [answer_relevancy, context_precision, context_recall, faithfulness,harmfulness]
# result = evaluate(query_engine,)

print("Evaluation Results:", results)