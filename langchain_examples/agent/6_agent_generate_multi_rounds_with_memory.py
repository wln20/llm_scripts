"""
Use an agent with a search engine tool to do multi round conversation with memory.
Reference: https://python.langchain.com/docs/tutorials/agents/
Dependencies:
```
pip install langchain-core, langgraph>0.2.27
pip install -qU langchain-mistralai
pip install langchain-community
pip install tavily-python
```
"""
import os
# model
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage
# tools
from langchain_community.tools.tavily_search import TavilySearchResults
# agent & memory
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = # TODO
os.environ["MISTRAL_API_KEY"] = # TODO
os.environ["TAVILY_API_KEY"] = # TODO

# Create model
model = ChatMistralAI(model="mistral-large-latest")

# Creat tool (this time it's a Tavily search engine)
search = TavilySearchResults(max_results=2)
# Put all tools in a list for reference
tools = [search]   

# Creat memory saver
memory = MemorySaver()

# Create agent (this time it's a ReAct agent)
# The agent would bind the model with the tools automatically, we don't need to manually bind it
# It also has a checkpointer param as memory
agent_executor = create_react_agent(model, tools, checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}

while True:
    prompt = input("Human: ")
    if prompt == 'q': 
        break

    message = HumanMessage(content=prompt)
    response = agent_executor.invoke({"messages": [message]}, config)
    # It seems that the first response's format is not exactly the same as the subsequent ones, so consider multiple situations
    try:    
        print("AI: " + response["messages"][-1].content[0]['text'])
    except:
        print("AI: " + response["messages"][-1].content)


    # You can also incorporate streaming in multi round dialogues by putting the following snippet into the loop:
    # for chunk in agent_executor.stream({"messages": [message]}):
    #     print(chunk)
    #     print("----")


