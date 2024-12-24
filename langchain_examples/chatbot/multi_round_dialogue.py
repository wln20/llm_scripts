"""
Multi round dialogue with basic longchain. 
Reference: https://python.langchain.com/docs/tutorials/chatbot/
Dependencies (Use Mistral API as an example):
```
pip install langchain-core
pip install -qU langchain-mistralai
```
"""
import os
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "XXX" # TODO
os.environ["MISTRAL_API_KEY"] = "XXX" # TODO


model = ChatMistralAI(model="mistral-large-latest")


# model.invoke(
#     [
#         HumanMessage(content="Hi! I'm Bob"),
#         AIMessage(content="Hello Bob! How can I assist you today?"),
#         HumanMessage(content="What's my name?"),
#     ]
# )

input_ctx = []
while True:
    prompt = input("Human: ")
    if prompt == 'q': 
        break

    input_ctx += [HumanMessage(content=prompt)]
    response = model.invoke(input_ctx)
    response_content = response.content
    
    print("AI: " + response_content, end='\n')
    input_ctx += [AIMessage(content=response_content)]
