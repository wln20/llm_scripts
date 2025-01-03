"""
TODO: This script seems not to be able to run correctly, skip it for now.
"""


"""
Use an agent with a search engine tool to generate responses, in a token-level streaming manner.
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
import asyncio
# model
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage
# tools
from langchain_community.tools.tavily_search import TavilySearchResults
# agent
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

# Create agent (this time it's a ReAct agent)
# The agent would bind the model with the tools automatically, we don't need to manually bind it
agent_executor = create_react_agent(model, tools)

# Generate with the agent
# Output each message once it's generated
message = HumanMessage(content="Who is Luning Wang (Tsinghua EE 2024)?")

# for chunk in agent_executor.stream({"messages": [message]}):
#     print(chunk)
#     print("----")


async def run_agent():
    async for event in agent_executor.astream_events(
        {"messages": [message]}, version="v1"
    ):
        event_kind = event["event"]
        if event_kind == "on_chain_start":
            if (
                event["name"] == "Agent"
            ):  # Was assigned when creating the agent with `.with_config({"run_name": "Agent"})`
                print(
                    f"Starting agent: {event['name']} with input: {event['data'].get('input')}"
                )
        elif event_kind == "on_chain_end":
            if (
                event["name"] == "Agent"
            ):  # Was assigned when creating the agent with `.with_config({"run_name": "Agent"})`
                print()
                print("--")
                print(
                    f"Done agent: {event['name']} with output: {event['data'].get('output')['output']}"
                )
        if event_kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                # Empty content in the context of OpenAI means
                # that the model is asking for a tool to be invoked.
                # So we only print non-empty content
                print(content, end="|")
        elif event_kind == "on_tool_start":
            print("--")
            print(
                f"Starting tool: {event['name']} with inputs: {event['data'].get('input')}"
            )
        elif event_kind == "on_tool_end":
            print(f"Done tool: {event['name']}")
            print(f"Tool output was: {event['data'].get('output')}")
            print("--")

asyncio.run(run_agent())
