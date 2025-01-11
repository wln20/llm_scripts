"""
Multi round dialogue with RAG from webpage
Reference: https://python.langchain.com/docs/tutorials/qa_chat_history/
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
from langchain_core.tools import tool   # tool call
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage   # message types
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition   # tool node

from langgraph.graph import START, END, StateGraph, MessagesState   # used for constructing the graph

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

# Build the application with langgraph

# Node: Entry point of the workflow, first judge if we need tool calls
# Generate an AIMessage that may include a tool-call to be sent.
# If the returned AIMessage does not include a tool-call, then this function would directly respond to user prompt, and the workflow will end.
# If the returned AIMessage includes a tool-call, then the workflow will proceed to the next node (call_tools).
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    model_with_tools = model.bind_tools([retrieve])
    response = model_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    # The response here is an AIMessage (may include tool calls), so make it a list to be consistent with the return type
    return {"messages": [response]}


# Make the retrieve function a tool, so that it can be called in the graph
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# Node: Wrap the tool in a ToolNode to be used in the graph
call_tools = ToolNode([retrieve])

# Node: Generate a response using the retrieved content.
def call_model(state: MessagesState):
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if isinstance(message, ToolMessage):
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if isinstance(message, HumanMessage) or isinstance(message, SystemMessage)
        or (isinstance(message, AIMessage) and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = model.invoke(prompt)
    return {"messages": [response]}

# 3. Workflow
workflow = StateGraph(state_schema=MessagesState)

workflow.add_node("query_or_respond", query_or_respond)
workflow.add_node("call_tools", call_tools)
workflow.add_node("call_model", call_model)

workflow.add_edge(START, "query_or_respond")
# Use in the conditional_edge to route to the ToolNode if the last message has tool calls. Otherwise, route to the end.
workflow.add_conditional_edges("query_or_respond", tools_condition, {END: END, "tools": "call_tools"})
workflow.add_edge("call_tools", "call_model")
workflow.add_edge("call_model", END)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

"""
Visualize the graph:
`app.get_graph().draw_mermaid_png(output_file_path='./graph.png')`

The graph would look like:
        _start_
            |
    query_or_respond
     /          \
call_tools      |
    |           |
call_model      |
    \           /
        _end_
"""
# ----------------------------------------------------------------------- #

# Run the app

# We have to incorporate a thread_id to enable memory saving, although we might not need multi conversations here
thread_id = "1" 
config = {"configurable": {"thread_id": thread_id}}

while True:
    prompt = input("Human: ")
    if prompt == 'q': 
        break
    
    input_messages = [HumanMessage(prompt)]
    result = app.invoke({"messages": input_messages}, config)
    print(f"AI: {result['messages'][-1].content}")





    
