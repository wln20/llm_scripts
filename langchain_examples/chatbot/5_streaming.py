"""
Multi round dialogue with langgraph, with streaming output (Note: each chunk has several tokens. It streams per chunk, not per token).
Reference: https://python.langchain.com/docs/tutorials/chatbot/
Dependencies:
```
pip install langchain-core, langgraph>0.2.27
pip install -qU langchain-mistralai
```
"""
import os
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# langgraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = # TODO
os.environ["MISTRAL_API_KEY"] = # TODO


model = ChatMistralAI(model="mistral-large-latest")

# Define prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You talk like a pirate. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Define a new graph
workflow = StateGraph(state_schema=MessagesState)


# Define the function that calls the model
def call_model(state: MessagesState):
    prompt = prompt_template.invoke(state)  # use prompt_template.invoke() to incorporate the template
    response = model.invoke(prompt)
    return {"messages": response}



# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Use thread_id to identify which conversation is we on
thread_id = "1"

while True:
    prompt = input("Human: ")
    if prompt == 'q': 
        break

    config = {"configurable": {"thread_id": thread_id}}
    input_messages = [HumanMessage(prompt)]
    # streaming, each chunk contains a dozen of tokens (not really streaing per token)
    print("AI: ", end='')
    for chunk, metadata in app.stream(
        {"messages": input_messages},
        config,
        stream_mode="messages",
    ):
        if isinstance(chunk, AIMessage):  # Filter to just model responses
            print(chunk.content, end="")
    print('\n')
