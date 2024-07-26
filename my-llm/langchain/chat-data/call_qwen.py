

from openai import OpenAI
import os

class CallQwen:

    def __init__(self):
        self.api_key = "sk-4d0ed669937c4e5bb13b5b55e41bc186"


    def single_chat(self, messages):
        client = OpenAI(
            # api_key=os.getenv("DASHSCOPE_API_KEY"), # 如果您没有配置环境变量，请在此处用您的API Key进行替换
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope服务的base_url
        )
        completion = client.chat.completions.create(            
            model="qwen-turbo",
            messages=messages,
            temperature=0.8,
            top_p=0.8
        )
        print(completion.model_dump_json())

    


if __name__ == '__main__':
    # print("Hello, World!")
    callQwen = CallQwen()

    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        # {'role': 'user', 'content': '你是谁？'}
        # {'role': 'user', 'content': '1 两 是多少克？'}
        {'role': 'user', 'content': 'AI PC 是指什么？'}
    ]

    callQwen.single_chat(messages=messages)
    


