"""
Use Azure's openai api to chat, with CLI or GUI.
"""
import os
from openai import AzureOpenAI
import gradio as gr

os.environ["AZURE_OPENAI_API_KEY"] = 'XXX'
os.environ["AZURE_OPENAI_ENDPOINT"] = 'XXX'


client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="XXX",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

def chat_with_cli():
    while True:
        input_ = input('Prompt:')
        if input_ == 'q':
            print('Bye.')
            exit()
        message = {"role": "user", "content": input_}
        messages_hist.append(message)
        response = client.chat.completions.create(
            model="gpt-4-0613",
            messages=messages_hist
            )
        output = response.choices[0].message.content
        messages_hist.append({"role": "assistant", "content": output})
        print(f'Response: {output}')

def chat_with_gui():
    def response(message, history):
        message = {"role": "user", "content": message}
        messages_hist = [{"role": "system", "content": "You are a helpful assistant."}]
        for session in history: # session = [user input, bot response]
            messages_hist.append({"role": "user", "content": session[0]})
            messages_hist.append({"role": "assistant", "content": session[1]})
        messages_hist.append(message)
        response = client.chat.completions.create(
            model="gpt-4-0613",
            messages=messages_hist
            )
        output = response.choices[0].message.content
        return output
      
    gr.ChatInterface(
      response,
      title="My GPT-4",
      description="By Wang Luning",
    ).launch(share=True)

if __name__ == '__main__':
    mode = "XXX" # ‘cli’ or 'gui'
    if mode == 'cli':
        chat_with_cli()
    else:
        chat_with_gui()

        
      

                                                                                                                                51,1          Bot
