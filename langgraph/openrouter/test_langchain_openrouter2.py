# from langchain_openai import ChatOpenAI
# gpt4o_chat = ChatOpenAI(model="gpt-4o", temperature=0)
# gpt35_chat = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)


from langchain_openai import ChatOpenAI
import os

api_key = os.getenv("OPENROUTER_API_KEY")  # 确保已 export
grok4 = ChatOpenAI(
    model="x-ai/grok-4-fast:free",
    temperature=0,
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

# resp = grok4.invoke("用一句话总结：LangChain 的作用是什么？")
# print(resp.content)

from langchain_core.messages import HumanMessage

# Create a message
msg = HumanMessage(content="Hello world", name="Lance")

# Message list
messages = [msg]

# Invoke the model with a list of messages 
# resp = grok4.invoke(messages)
# print(resp)








