"""
Use an agent with a search engine tool to generate responses, in a message-level streaming manner (output each message once it's generated, rather than output all messages all at once).
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
# Output each message once it's generated
message = HumanMessage(content="Who is Luning Wang (Tsinghua EE 2024)?")

for chunk in agent_executor.stream({"messages": [message]}):
    print(chunk)
    print("----")
"""
{'agent': {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': '0B6tipiYD', 'type': 'function', 'function': {'name': 'tavily_search_results_json', 'arguments': '{"query": "Luning Wang Tsinghua EE 2024"}'}}]}, response_metadata={'token_usage': {'prompt_tokens': 125, 'total_tokens': 165, 'completion_tokens': 40}, 'model': 'mistral-large-latest', 'finish_reason': 'tool_calls'}, id='run-edff927a-4072-43f8-ad55-6cbbfa4e738e-0', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'Luning Wang Tsinghua EE 2024'}, 'id': '0B6tipiYD', 'type': 'tool_call'}], usage_metadata={'input_tokens': 125, 'output_tokens': 40, 'total_tokens': 165})]}}
----
{'tools': {'messages': [ToolMessage(content='[{"url": "https://wln20.github.io/files/CV-Luning+Wang.pdf", "content": "Luning Wang EDUCATION Tsinghua University Beijing, China Department of Electronic Engineering B.E. in Electronic Information Science and Technology 08/2020- 06/2024 ⚫ Overall GPA: 3.76/4.0 ⚫ GRE: Verbal 155/ Quantity 170, Total 325, AW 4.0 ⚫ TOEFL: Reading 29/ Listening 26/ Speaking 22/ Writing 25/ Total 102 ⚫ Relevant ourses"}, {"url": "https://github.com/wln20/", "content": "wln20 (Luning Wang) · GitHub GitHub Copilot Write better code with AI Code Search Find more, search less GitHub Sponsors Fund open source developers Enterprise platform AI-powered developer platform Advanced Security Enterprise-grade security features GitHub Copilot Enterprise-grade AI features Search code, repositories, users, issues, pull requests... Search Saved searches wln20 Follow Overview Repositories 14 Projects 0 Packages 0 Stars 33 wln20 @ THU EE | Seeking 2025\'summer intern position on AI (LLM / MLSys) https://wln20.github.io/ Block or report wln20 Learn more about blocking users. Report abuseContact GitHub support about this user’s behavior. Overview Repositories 14 Projects 0 Packages 0 Stars 33 If the problem persists, check the GitHub status page or contact support. © 2024 GitHub,\xa0Inc. Footer navigation"}]', name='tavily_search_results_json', id='872d38ff-8943-4a9b-8df3-f3e27356a1ac', tool_call_id='0B6tipiYD', artifact={'query': 'Luning Wang Tsinghua EE 2024', 'follow_up_questions': None, 'answer': None, 'images': [], 'results': [{'title': 'PDF', 'url': 'https://wln20.github.io/files/CV-Luning+Wang.pdf', 'content': 'Luning Wang EDUCATION Tsinghua University Beijing, China Department of Electronic Engineering B.E. in Electronic Information Science and Technology 08/2020- 06/2024 ⚫ Overall GPA: 3.76/4.0 ⚫ GRE: Verbal 155/ Quantity 170, Total 325, AW 4.0 ⚫ TOEFL: Reading 29/ Listening 26/ Speaking 22/ Writing 25/ Total 102 ⚫ Relevant ourses', 'score': 0.6601948, 'raw_content': None}, {'title': 'wln20 (Luning Wang) - GitHub', 'url': 'https://github.com/wln20/', 'content': "wln20 (Luning Wang) · GitHub GitHub Copilot Write better code with AI Code Search Find more, search less GitHub Sponsors Fund open source developers Enterprise platform AI-powered developer platform Advanced Security Enterprise-grade security features GitHub Copilot Enterprise-grade AI features Search code, repositories, users, issues, pull requests... Search Saved searches wln20 Follow Overview Repositories 14 Projects 0 Packages 0 Stars 33 wln20 @ THU EE | Seeking 2025'summer intern position on AI (LLM / MLSys) https://wln20.github.io/ Block or report wln20 Learn more about blocking users. Report abuseContact GitHub support about this user’s behavior. Overview Repositories 14 Projects 0 Packages 0 Stars 33 If the problem persists, check the GitHub status page or contact support. © 2024 GitHub,\xa0Inc. Footer navigation", 'score': 0.5911811, 'raw_content': None}], 'response_time': 1.59})]}}
----
{'agent': {'messages': [AIMessage(content=[{'type': 'text', 'text': 'Luning Wang is a student at Tsinghua University in Beijing, China. He is pursuing a Bachelor of Engineering (B.E.) degree in Electronic Information Science and Technology from the Department of Electronic Engineering. His expected graduation date is June 2024. Luning Wang has an overall GPA of 3.76/4.0 and has taken relevant courses in his field of study. Additionally, he has scored well on standardized tests, with a GRE score of Verbal 155, Quantitative 170, and Total 325, and a TOEFL score of Reading 29, Listening 26, Speaking 22, Writing 25, and Total 102 '}, {'type': 'reference', 'reference_ids': [0]}, {'type': 'text', 'text': '.\n\nLuning Wang is also active on GitHub under the username `wln20` and is seeking a summer intern position for 2025 in the field of AI, particularly in LLM or MLSys '}, {'type': 'reference', 'reference_ids': [1]}, {'type': 'text', 'text': '.'}], additional_kwargs={}, response_metadata={'token_usage': {'prompt_tokens': 611, 'total_tokens': 829, 'completion_tokens': 218}, 'model': 'mistral-large-latest', 'finish_reason': 'stop'}, id='run-18a69d14-8da7-4a6a-909a-6122b2cd989d-0', usage_metadata={'input_tokens': 611, 'output_tokens': 218, 'total_tokens': 829})]}}
----
"""


