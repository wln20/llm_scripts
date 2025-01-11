"""
QA with retrieved documents from a webpage (https://lilianweng.github.io/posts/2023-06-23-agent/), with detailed comments
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
# Use HF_TOKEN for downloading mistral tokenizer from huggingface (for splitting the text)
# Must request access before running: https://huggingface.co/mistralai/Mixtral-8x7B-v0.1
os.environ["HF_TOKEN"] = # TODO
os.environ["TOKENIZERS_PARALLELISM"] = "false" 

# ----------------------------------------------------------------------- #

# Create model
model = ChatMistralAI(model="mistral-large-latest")

# ----------------------------------------------------------------------- #
# Create document loader and load a document
# Only keep post title, headers, and content from the full HTML.
target_url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=(target_url,),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()
"""
len(docs) == 1
docs[0] is a Document object
docs[0].page_content is the raw text data
len(docs[0].page_content) == 43131
"""

# ----------------------------------------------------------------------- #

# Create text splitter and split the document, using downloaded mistral tokenizer
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)
"""
all_splits is a list of Document objects, each containing a chunk of the original document
len(all_splits) == 66
"""

# ----------------------------------------------------------------------- #

# Create embedding model and vector store, and store(index) the document chunks
embeddings = MistralAIEmbeddings(model="mistral-embed")
vector_store = InMemoryVectorStore(embedding=embeddings)
document_ids = vector_store.add_documents(documents=all_splits)
"""
document_ids is a list of document IDs, each is a the index of a document chunk in the vector store
document_ids[0] == 'c867997e-2271-4690-bbd0-95ea6db9b59c'
"""

# ----------------------------------------------------------------------- #

# Could use prompt templates from hub: https://python.langchain.com/docs/tutorials/rag/#orchestration
# Here we define prompt template manually
raw_template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Context: {context}

Question: {question}

Answer:"""

# Build the prompt template object with `PromptTemplate.from_template` method
prompt_template = PromptTemplate.from_template(raw_template)
# Use: prompt_template.invoke({"context": "This is context", "question": "This is question"})
# return: StringPromptValue(text="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\n\nContext: This is context\n\nQuestion: This is question\n\nHelpful Answer:")

# ----------------------------------------------------------------------- #

# Build the application with langgraph
# Define these three things: State, Nodes, Workflow

# 1. State: the state of the application controls what data is input to the application, transferred between steps, and output by the application. It is typically a TypedDict.
# Here for the single round RAG application, we can just keep track of the input question, retrieved context and generated answer:
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
# Remember that in chatbot, we use langgraph.graph.MessageState, which only has a `message` field (a list of messages recording the context):
# ```
# class MessagesState(TypedDict):
#     messages: Annotated[list[AnyMessage], add_messages]
# ```


# 2. Nodes: the application nodes are the steps in the application. Each node is a function that takes the current state and returns the updated state.
# Here we define 2 steps (i.e. 2 functions): retrieval and generation (call model)
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"], k=3) 
    # The returned `retrieved_docs` is a list of Document objects
    return {"context": retrieved_docs}

def call_model(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    # join the retrieved docs, and put them into the "context" field of the prompt template
    # also inject the "question" field with state["question"]
    prompt = prompt_template.invoke({"question": state["question"], "context": docs_content})
    response = model.invoke(prompt)
    return {"answer": response.content}

# The returned value of each node function must be a key-value pair used to fill a field in the State object
# Remember that in chatbot, we use langgraph.graph.MessageState, which only has a `message` field
# So the call_model function is:
# ```
# def call_model(state: MessagesState):
#     prompt = prompt_template.invoke(state)  # use prompt_template.invoke() to incorporate the template
#     response = model.invoke(prompt)
#     return {"messages": response}
# ```
# Here the `response` variable should be a list of Message objects, such as `[HumanMessage(prompt), AIMessage(...)]`


# 3. Workflow: compile the application into a single `graph` object
# Here we just connect the retrieval and generation steps into a single sequence
workflow = StateGraph(state_schema=State)
# Remember that in chatbot, we use `workflow = StateGraph(state_schema=MessageState)`, while here we use our customized State object
# Join the (virtual) START node with the first node "retrieve"
workflow.add_edge(START, "retrieve")
# Add the two nodes to the workflow, and connect them in a sequence
workflow.add_sequence([retrieve, call_model])
# Remember that in chatbot, we use `workflow.add_node("model", call_model)`, while here we add two nodes with `workflow.add_sequence([,])`
app = workflow.compile()

"""
We could also visualize the graph with:
`app.get_graph().draw_mermaid_png(output_file_path='./graph.png')`

The graph would look like:
    _start_
       |
    retrieve
       |
    call_model
"""

# ----------------------------------------------------------------------- #

# Run the app
result = app.invoke({"question": "What is Task Decomposition?"})
# `app.invoke({"question": "What is Task Decomposition?"})` would fill the current state's "question" field
# `result` is the updated state (in this app, it would be a state with "context" and "answer" fields filled)
# `result.keys == dict_keys(['question', 'context', 'answer'])`

"""
print(result['question'])
# 'What is Task Decomposition?'

print(result["context"])
# (A list of three retrieved Document objects)
"""

print(result["answer"])
# Task Decomposition is a strategy used to break down complex tasks into smaller, manageable parts. This is often done using the Chain of Thought (CoT) prompting technique, where a model is instructed to "think step by step" to handle each component of the task more effectively. It helps in understanding the model's thinking process by dividing big tasks into simpler steps.
