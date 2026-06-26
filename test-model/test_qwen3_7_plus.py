import os
from openai import OpenAI

from dotenv import load_dotenv


load_dotenv()

client = OpenAI(
    api_key=os.getenv("QWEN_API_KEY"),
    base_url=os.getenv("QWEN_ENDPOINT"),
)


instructions = """简要分析这张图片"""
img_url = os.getenv("TEST_IMG_URL")  # 从环境变量中获取测试图片 URL

# messages = [{"role": "user", "content": "3+2=?"}]
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": img_url
                }
            },
            {
                "type": "text",
                "text": instructions
            }
        ]
    }
]


completion = client.chat.completions.create(
    model="qwen3.7-plus",  # 您可以按需更换为其它深度思考模型
    messages=messages,
    extra_body={"enable_thinking": True},  # 是否启用思考过程
    stream=True,
    # stream_options = {
    #     "include_usage": True,  # 是否包含使用量信息
    # }
)

is_answering = False  # 是否进入回复阶段
print("\n" + "=" * 20 + "思考过程" + "=" * 20)
for chunk in completion:
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta
    if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
        if not is_answering:
            print(delta.reasoning_content, end="", flush=True)
    if hasattr(delta, "content") and delta.content:
        if not is_answering:
            print("\n" + "=" * 20 + "回答内容" + "=" * 20)
            is_answering = True
        print(delta.content, end="", flush=True)





