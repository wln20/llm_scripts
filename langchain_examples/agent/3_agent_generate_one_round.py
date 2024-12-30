"""
Use an agent with a search engine tool to generate responses (one round conversation).
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
# Case 1: The model doesn't call a tool, it merely generates with itself.
message_1 = HumanMessage(content="Hi!")
response = agent_executor.invoke({"messages": [message_1]})
print(response["messages"])
"""
[
    HumanMessage(content='Hi!', additional_kwargs={}, response_metadata={}, id='9107a181-c614-4ab5-b4ce-7970ddd26ba2'), 
    AIMessage(content='Hello! How can I assist you today?', additional_kwargs={}, response_metadata={'token_usage': {'prompt_tokens': 109, 'total_tokens': 118, 'completion_tokens': 9}, 'model': 'mistral-large-latest', 'finish_reason': 'stop'}, id='run-8d751604-edc6-44ee-a677-37d99b47a1f1-0', usage_metadata={'input_tokens': 109, 'output_tokens': 9, 'total_tokens': 118})]
"""

# Case 2: The model calls a tool
# Given the initial human message, the model firstly replys with a request to call a tool
# then the tool would return a tool message with the retrieved contents
# finally the model would give response with the information given by the tool
message_2 = HumanMessage(content="Who is Luning Wang (Tsinghua EE 2024)?")
response = agent_executor.invoke({"messages": [message_2]})
print(response["messages"])
"""
[
    HumanMessage(content='Who is Luning Wang (Tsinghua EE 2024)?', additional_kwargs={}, response_metadata={}, id='9fc4ad4e-1142-42a5-83d0-bb54058d0c09'), 
    AIMessage(content='', additional_kwargs={'tool_calls': [{'id': '4Cn1aEYgV', 'type': 'function', 'function': {'name': 'tavily_search_results_json', 'arguments': '{"query": "Luning Wang Tsinghua EE 2024"}'}}]}, response_metadata={'token_usage': {'prompt_tokens': 125, 'total_tokens': 165, 'completion_tokens': 40}, 'model': 'mistral-large-latest', 'finish_reason': 'tool_calls'}, id='run-9bb9d995-79e4-4f61-a0f5-661ca1092da1-0', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'Luning Wang Tsinghua EE 2024'}, 'id': '4Cn1aEYgV', 'type': 'tool_call'}], usage_metadata={'input_tokens': 125, 'output_tokens': 40, 'total_tokens': 165}), 
    ToolMessage(content='[{"url": "https://wln20.github.io/files/CV-Luning+Wang.pdf", "content": "Luning Wang EDUCATION Tsinghua University Beijing, China Department of Electronic Engineering B.E. in Electronic Information Science and Technology 08/2020- 06/2024 ⚫ Overall GPA: 3.76/4.0 ⚫ GRE: Verbal 155/ Quantity 170, Total 325, AW 4.0 ⚫ TOEFL: Reading 29/ Listening 26/ Speaking 22/ Writing 25/ Total 102 ⚫ Relevant ourses"}, {"url": "https://github.com/wln20/", "content": "wln20 (Luning Wang) · GitHub GitHub Copilot Write better code with AI Code Search Find more, search less GitHub Sponsors Fund open source developers Enterprise platform AI-powered developer platform Advanced Security Enterprise-grade security features GitHub Copilot Enterprise-grade AI features Search code, repositories, users, issues, pull requests... Search Saved searches wln20 Follow Overview Repositories 14 Projects 0 Packages 0 Stars 33 wln20 @ THU EE | Seeking 2025\'summer intern position on AI (LLM / MLSys) https://wln20.github.io/ Block or report wln20 Learn more about blocking users. Report abuseContact GitHub support about this user’s behavior. Overview Repositories 14 Projects 0 Packages 0 Stars 33 If the problem persists, check the GitHub status page or contact support. © 2024 GitHub,\xa0Inc. Footer navigation"}]', name='tavily_search_results_json', id='ad2a755c-5fa1-49cf-861d-fcdb26d7c158', tool_call_id='4Cn1aEYgV', artifact={'query': 'Luning Wang Tsinghua EE 2024', 'follow_up_questions': None, 'answer': None, 'images': [], 'results': [{'title': 'PDF', 'url': 'https://wln20.github.io/files/CV-Luning+Wang.pdf', 'content': 'Luning Wang EDUCATION Tsinghua University Beijing, China Department of Electronic Engineering B.E. in Electronic Information Science and Technology 08/2020- 06/2024 ⚫ Overall GPA: 3.76/4.0 ⚫ GRE: Verbal 155/ Quantity 170, Total 325, AW 4.0 ⚫ TOEFL: Reading 29/ Listening 26/ Speaking 22/ Writing 25/ Total 102 ⚫ Relevant ourses', 'score': 0.6601948, 'raw_content': None}, {'title': 'wln20 (Luning Wang) - GitHub', 'url': 'https://github.com/wln20/', 'content': "wln20 (Luning Wang) · GitHub GitHub Copilot Write better code with AI Code Search Find more, search less GitHub Sponsors Fund open source developers Enterprise platform AI-powered developer platform Advanced Security Enterprise-grade security features GitHub Copilot Enterprise-grade AI features Search code, repositories, users, issues, pull requests... Search Saved searches wln20 Follow Overview Repositories 14 Projects 0 Packages 0 Stars 33 wln20 @ THU EE | Seeking 2025'summer intern position on AI (LLM / MLSys) https://wln20.github.io/ Block or report wln20 Learn more about blocking users. Report abuseContact GitHub support about this user’s behavior. Overview Repositories 14 Projects 0 Packages 0 Stars 33 If the problem persists, check the GitHub status page or contact support. © 2024 GitHub,\xa0Inc. Footer navigation", 'score': 0.5911811, 'raw_content': None}], 'response_time': 1.54}), 
    AIMessage(content=[{'type': 'text', 'text': "Luning Wang is a student at Tsinghua University in Beijing, China, pursuing a Bachelor's degree in Electronic Engineering with a focus on Electronic Information Science and Technology. He is expected to graduate in June 2024. His academic achievements include an overall GPA of 3.76/4.0, a GRE score of 325 (Verbal 155, Quantitative 170, Analytical Writing 4.0), and a TOEFL score of 102 (Reading 29, Listening 26, Speaking 22, Writing 25) "}, {'type': 'reference', 'reference_ids': [1]}, {'type': 'text', 'text': '.\n\nLuning Wang is also active on GitHub under the username "wln20" and is seeking a summer internship position for 2025 in the field of AI, specifically in Large Language Models (LLM) or Machine Learning Systems (MLSys) '}, {'type': 'reference', 'reference_ids': [2]}, {'type': 'text', 'text': '.'}], additional_kwargs={}, response_metadata={'token_usage': {'prompt_tokens': 613, 'total_tokens': 820, 'completion_tokens': 207}, 'model': 'mistral-large-latest', 'finish_reason': 'stop'}, id='run-96171e08-3878-4adc-be47-ce725adbb2f4-0', usage_metadata={'input_tokens': 613, 'output_tokens': 207, 'total_tokens': 820})]
"""
print(response["messages"][-1].content[0]['text'])
"""
Luning Wang is a student at Tsinghua University in Beijing, China, pursuing a Bachelor's degree in Electronic Engineering with a focus on Electronic Information Science and Technology. He is expected to graduate in June 2024. His academic achievements include an overall GPA of 3.76/4.0, a GRE score of 325 (Verbal 155, Quantitative 170, Analytical Writing 4.0), and a TOEFL score of 102 (Reading 29, Listening 26, Speaking 22, Writing 25) 
"""

