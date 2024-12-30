"""
Combine the LLM and the tool. Here we only shows how to bind the LLM with the tool and let LLM propose what tools to call, without really using the tool to do generation (to further generate with the tool's assistance, we need to construct an agent, see in later sections)
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

# Bind the model with the tools (give the language model knowledge of these tools)
model_with_tools = model.bind_tools(tools)

# Use the model to do tool calls
# Case 1: The model do not need to call the tools, it just generate with itself
message_1 = HumanMessage(content="Hi!")
response = model_with_tools.invoke([message_1])

print(f"ContentString: {response.content}")
"""
Hello! How can I assist you today?
"""
print(f"ToolCalls: {response.tool_calls}")
"""
[]
"""

# Case 2: The model needs to do tool call
# But, here it's only the model telling us that it wants to use the search engine with an organized query, and it has not really used the tool to do generation yet
# So, the returned content string is None, but we can see tool calls
# In the next section of using an agent to generate, we can see that this is the first step for the agent execution.
message_2 = HumanMessage(content="Who is Luning Wang (Tsinghua)?")
response = model_with_tools.invoke([message_2])

print(f"ContentString: {response.content}")
"""

"""
print(f"ToolCalls: {response.tool_calls}")
"""
[
    {
        'name': 'tavily_search_results_json', 
        'args': {
                    'query': 'Luning Wang Tsinghua'
                }, 
        'id': 'gBha822Dl', 
        'type': 'tool_call'
    }
]
"""
