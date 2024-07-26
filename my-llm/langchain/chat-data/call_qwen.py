
import os
import json
import random
from datetime import datetime
from openai import OpenAI


class CallQwen:

    def __init__(self):
        self.api_key = "sk-4d0ed669937c4e5bb13b5b55e41bc186"


    def get_response(self, messages):
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
        return completion

    
    def single_chat(self, messages):
        self.get_response(messages=messages)

    
    def multi_turn_chat(self):
        messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
        # 您可以自定义设置对话轮数，当前为3
        for i in range(3):
            user_input = input("请输入：")
            # 将用户问题信息添加到messages列表中
            messages.append({'role': 'user', 'content': user_input})
            assistant_output = self.get_response(messages).choices[0].message.content
            # 将大模型的回复信息添加到messages列表中
            messages.append({'role': 'assistant', 'content': assistant_output})
            print(f'用户输入：{user_input}')
            print(f'模型输出：{assistant_output}')
            print('\n')

    
    def get_response_stream(self, messages):
        client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope服务的base_url
        )
        completion = client.chat.completions.create(            
            model="qwen-turbo",
            messages=messages,
            stream=True,
            # 可选，配置以后会在流式输出的最后一行展示token使用信息
            stream_options={"include_usage": True}
        )
        for chunk in completion:
            print(chunk.model_dump_json())


    # 模拟天气查询工具。返回结果示例：“北京今天是晴天。”
    def get_current_weather(self, location):
        return f"{location}今天是晴天。 "


    # 查询当前时间的工具。返回结果示例：“当前时间：2024-04-15 17:15:18。“
    def get_current_time(self):
        # 获取当前日期和时间
        current_datetime = datetime.now()
        # 格式化当前日期和时间
        formatted_time = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        # 返回格式化后的当前时间
        return f"当前时间：{formatted_time}。"


    def qwen_function_call(self): 
        client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        
        # 定义工具列表，模型在选择使用哪个工具时会参考工具的name和description
        tools = [
            # 工具1 获取当前时刻的时间
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "当你想知道现在的时间时非常有用。",
                    "parameters": {}  # 因为获取当前时间无需输入参数，因此parameters为空字典
                }
            },
            # 工具2 获取指定城市的天气
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "当你想查询指定城市的天气时非常有用。",
                    "parameters": {  # 查询天气时需要提供位置，因此参数设置为location
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "城市或县区，比如北京市、杭州市、余杭区等。"
                            }
                        }
                    },
                    "required": [
                        "location"
                    ]
                }
            }
        ]

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input("请输入：")}
        ]

        # The first invocation of the model
        first_response = client.chat.completions.create(
            model="qwen-plus",
            messages=messages,
            tools=tools
        )

        
        assistant_output = first_response.choices[0].message
        print(f"\noutput of First round: {first_response}\n")
        # If the model determines that there is no need to invoke the tool, then print out the assistant"s response directly without making a second call to the model.

        # TODO：目前这里是有问题的
        # 将大模型的回复信息添加到messages列表中
        # messages.append({'role': 'tool_calls', 'content': assistant_output.content})


        if assistant_output.tool_calls is None:
            print(f"Final response：{assistant_output.content}")
            return
        else:
            function_name = assistant_output.tool_calls[0].function.name
            # If the model chooses the tool get_current_weather
            if function_name == "get_current_weather":
                tool_info = {"name": "get_current_weather", "role": "tool"}
                location = json.loads(assistant_output.tool_calls[0]["function"]["arguments"])["properties"]["location"]
                tool_info["content"] = self.get_current_weather(location)
            # If the model chooses the tool get_current_time
            elif function_name == "get_current_time":
                tool_info = {"name": "get_current_time", "role": "tool"}
                tool_info["content"] = self.get_current_time()

        print(f"Tool info：{tool_info['content']}\n")

        messages.append({
            "role": "tool",
            "name": function_name,
            "content": tool_info["content"]
        })

        print(messages)

        # TODO：目前这里是有问题的
        # The second round of the model"s invocation involves summarizing the output of the tool
        second_response = client.chat.completions.create(
            model="qwen-max",
            messages=messages,
            tools=tools
        )
        print(f"Output of second round: {second_response}\n")
        print(f"Final response: {second_response.choices[0].message.content}")




if __name__ == '__main__':
    # print("Hello, World!")
    callQwen = CallQwen()

    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        # {'role': 'user', 'content': '你是谁？'}
        # {'role': 'user', 'content': '1 两 是多少克？'}
        {'role': 'user', 'content': 'AI PC 是指什么？'}
    ]

    # callQwen.single_chat(messages=messages)
    # 测试问题：堺雅人 他的代表作
    # callQwen.multi_turn_chat()
    # callQwen.get_response_stream(messages=messages)
    # 测试问题：几点了？
    callQwen.qwen_function_call()
    


