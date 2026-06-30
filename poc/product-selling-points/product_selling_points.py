import os
import json
from openai import OpenAI

from dotenv import load_dotenv


load_dotenv()

client = OpenAI(
    api_key=os.getenv("QWEN_API_KEY"),
    base_url=os.getenv("QWEN_ENDPOINT"),
)


instructions = """你是一个电商商品详情页分析师，擅长从图文长图中提取产品结构化的卖点信息。

## 输入
一张电子产品的商品详情页长图（如手机/电脑/平板等），内容为竖向长图，包含多屏卖点模块。

## 任务
1. 按「从上到下的出现顺序」+「视觉占比（字号/面积/是否首屏）」两个维度，综合判断优先级。
2. 识别出所有**主要产品卖点**（忽略促销、物流、售后、赠品等非产品本身信息）。
3. 每个主要卖点下，提取其支撑的 2–n 个子项（参数 / 功能 / 场景 / 文案要点）。
4. 将所有卖点分为 3 级：

   - 🥇 一级卖点：出现最靠前 或 视觉占比最大，通常是该产品最核心的 1–3 个差异化主打点
   - 🥈 二级卖点：中间模块，出现顺序中等、占比中等，属于重要但非最核心
   - 🥉 三级卖点：靠后模块或占比小，属于补充性/细节性卖点

   排序规则：
   - 同级别内严格按「出现顺序」排
   - 若某卖点同时"出现早 + 占比大"，优先升到更高级

## 输出格式（严格 JSON）

{
  "product_type": "识别出的产品类别",
  "selling_points": [
    {
      "level": 1,
      "title": "卖点主标题（原文或近义提炼）",
      "sub_items": ["子项1", "子项2", "子项3"]
    },
    {
      "level": 2,
      "title": "...",
      "sub_items": [...]
    },
    {
      "level": 3,
      "title": "...",
      "sub_items": [...]
    }
  ]
}

## 额外要求
- sub_items 尽量保留原文关键词，不要过度润色
- 如果某卖点下子项不明显，可写 []
- 不要输出解释性文字，只要 JSON"""



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


enable_thinking = False  # 是否启用思考过程
reasoning_content = ""  # 定义完整思考过程
answer_content = ""     # 定义完整回复
is_answering = False   # 判断是否结束思考过程并开始回复

completion = client.chat.completions.create(
    model="qwen3.7-plus",  # 您可以按需更换为其它深度思考模型
    messages=messages,
    # 是否启用思考过程
    extra_body = {
        "enable_thinking": enable_thinking
    },  
    stream=True
)

if enable_thinking:
    print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")

for chunk in completion:
    # 如果chunk.choices为空，则打印usage
    if not chunk.choices:
        print("\n" + "=" * 20 + "Token Usage" + "=" * 20 + "\n")
        # print("\nUsage:")
        print(json.dumps(chunk.usage.model_dump(), ensure_ascii=False, indent=2))
    else:
        delta = chunk.choices[0].delta
        # 打印思考过程
        if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
            print(delta.reasoning_content, end='', flush=True)
            reasoning_content += delta.reasoning_content
        else:
            # 开始回复
            if delta.content != "" and is_answering is False:
                print("\n" + "=" * 20 + "回答内容" + "=" * 20 + "\n")
                is_answering = True
            # 打印回复过程
            print(delta.content, end='', flush=True)
            answer_content += delta.content



# print("=" * 20 + "完整思考过程" + "=" * 20 + "\n")
# print(reasoning_content)
# print("=" * 20 + "完整回答内容" + "=" * 20 + "\n")
# print(answer_content)

