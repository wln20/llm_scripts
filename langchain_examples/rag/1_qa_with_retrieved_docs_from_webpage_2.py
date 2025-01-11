"""
Another example of QA with retrieved documents from a webpage (https://wln20.github.io)
Reference: https://python.langchain.com/docs/tutorials/rag/
"""
import os
import bs4 # for parsing HTML
# model
from langchain_mistralai import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
# RAG components
from langchain_community.document_loaders import WebBaseLoader # document loader
from langchain_text_splitters import RecursiveCharacterTextSplitter # text splitter
from langchain_core.vectorstores import InMemoryVectorStore # vector store
# RAG application (required by langgraph)
from langchain_core.prompts import PromptTemplate   # prompt template
from langchain_core.documents import Document   # used for type hinting
from typing_extensions import List, TypedDict   # used for type hinting
from langgraph.graph import START, StateGraph   # used for constructing the graph

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = # TODO
os.environ["MISTRAL_API_KEY"] = # TODO
os.environ["HF_TOKEN"] = # TODO
os.environ["TOKENIZERS_PARALLELISM"] = "false" 

# ----------------------------------------------------------------------- #

# Create model
model = ChatMistralAI(model="mistral-large-latest")

# ----------------------------------------------------------------------- #
# Create document loader and load a document
target_url = "https://wln20.github.io/"
bs4_strainer = bs4.SoupStrainer(class_=("page__title", "page__content"))
loader = WebBaseLoader(
    web_paths=(target_url,),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

# ----------------------------------------------------------------------- #

# Create text splitter and split the document, using downloaded mistral tokenizer
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,  # chunk size (characters)
    chunk_overlap=50,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)


# ----------------------------------------------------------------------- #

# Create embedding model and vector store, and store(index) the document chunks
embeddings = MistralAIEmbeddings(model="mistral-embed")
vector_store = InMemoryVectorStore(embedding=embeddings)
document_ids = vector_store.add_documents(documents=all_splits)

# ----------------------------------------------------------------------- #

# Define the prompt template
raw_template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Context: {context}

Question: {question}

Answer:"""


prompt_template = PromptTemplate.from_template(raw_template)

# ----------------------------------------------------------------------- #

# Build the application with langgraph
# Define these three things: State, Nodes, Workflow

# 1. State
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# 2. Nodes
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"], k=5) 
    return {"context": retrieved_docs}

def call_model(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    prompt = prompt_template.invoke({"question": state["question"], "context": docs_content})
    response = model.invoke(prompt)
    return {"answer": response.content}

# 3. Workflow
workflow = StateGraph(state_schema=State)
workflow.add_edge(START, "retrieve")
workflow.add_sequence([retrieve, call_model])
app = workflow.compile()

# ----------------------------------------------------------------------- #

# Run the app
result = app.invoke({"question": "What's Luning Wang's University?"})

print(result["answer"])
