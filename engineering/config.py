# vLLM Qwen3 流式调用配置
VLLM_CONFIG = {
    "base_url": "http://localhost:8000/v1",
    "api_key": "EMPTY",
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "timeout": 60.0
}

# 默认生成参数
DEFAULT_PARAMS = {
    "temperature": 0.7,
    "max_tokens": 2048,
    "top_p": 0.9,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
}

# 系统提示词模板
SYSTEM_PROMPTS = {
    "default": "你是一个有用的AI助手。请用中文回答问题，保持友好和专业。",
    "coding": "你是一个专业的编程助手。请提供准确、详细的代码解答和技术指导。",
    "academic": "你是一个学术研究助手。请提供准确、客观的学术信息和分析。",
    "creative": "你是一个创意写作助手。请发挥创造力，提供有趣和富有想象力的内容。"
}
