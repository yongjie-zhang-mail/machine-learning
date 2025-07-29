#!/usr/bin/env python3
"""
vLLM Qwen3 模型流式调用使用示例
"""

import asyncio
import time
from stream_chat import VLLMStreamChat
from config import VLLM_CONFIG, DEFAULT_PARAMS, SYSTEM_PROMPTS


def basic_usage_example():
    """基础使用示例"""
    print("=== 基础使用示例 ===")
    
    # 创建客户端
    client = VLLMStreamChat(**VLLM_CONFIG)
    
    # 简单对话
    question = "什么是机器学习？请简要介绍一下。"
    print(f"问题: {question}")
    print("回答: ", end="", flush=True)
    
    for chunk in client.stream_chat(
        message=question,
        system_prompt=SYSTEM_PROMPTS["academic"],
        **DEFAULT_PARAMS
    ):
        print(chunk, end="", flush=True)
    print("\n")


def conversation_example():
    """连续对话示例"""
    print("=== 连续对话示例 ===")
    
    client = VLLMStreamChat(**VLLM_CONFIG)
    
    # 对话序列
    conversations = [
        "请介绍一下 Python 编程语言",
        "Python 在机器学习领域有哪些优势？",
        "可以推荐一些 Python 机器学习的学习资源吗？"
    ]
    
    for i, message in enumerate(conversations, 1):
        print(f"\n轮次 {i}")
        print(f"问题: {message}")
        print("回答: ", end="", flush=True)
        
        for chunk in client.stream_chat(
            message=message,
            system_prompt=SYSTEM_PROMPTS["coding"] if i == 1 else None,
            temperature=0.7,
            max_tokens=1024
        ):
            print(chunk, end="", flush=True)
        print()


def code_generation_example():
    """代码生成示例"""
    print("=== 代码生成示例 ===")
    
    client = VLLMStreamChat(**VLLM_CONFIG)
    
    prompt = """
请写一个 Python 函数，实现以下功能：
1. 读取 CSV 文件
2. 对数据进行基本的清洗（去除空值）
3. 计算数值列的基本统计信息
4. 返回处理后的数据和统计信息

请包含完整的代码和注释。
"""
    
    print("代码生成请求:")
    print(prompt)
    print("\n生成的代码:")
    print("-" * 50)
    
    for chunk in client.stream_chat(
        message=prompt,
        system_prompt=SYSTEM_PROMPTS["coding"],
        temperature=0.3,  # 较低的温度以获得更准确的代码
        max_tokens=2048
    ):
        print(chunk, end="", flush=True)
    print("\n" + "-" * 50)


def performance_test():
    """性能测试"""
    print("=== 性能测试 ===")
    
    client = VLLMStreamChat(**VLLM_CONFIG)
    
    test_questions = [
        "请解释深度学习的基本概念",
        "什么是神经网络？",
        "介绍一下卷积神经网络的工作原理",
        "什么是注意力机制？",
        "解释一下 Transformer 架构的优势"
    ]
    
    total_time = 0
    total_tokens = 0
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n测试 {i}: {question}")
        
        start_time = time.time()
        token_count = 0
        
        print("回答: ", end="", flush=True)
        for chunk in client.stream_chat(
            message=question,
            system_prompt=SYSTEM_PROMPTS["academic"] if i == 1 else None,
            temperature=0.7,
            max_tokens=512,
            keep_history=False  # 不保持历史以确保独立测试
        ):
            print(chunk, end="", flush=True)
            token_count += len(chunk)
        
        end_time = time.time()
        elapsed = end_time - start_time
        total_time += elapsed
        total_tokens += token_count
        
        print(f"\n[耗时: {elapsed:.2f}s, 字符数: {token_count}]")
    
    print(f"\n=== 性能统计 ===")
    print(f"总耗时: {total_time:.2f}s")
    print(f"平均耗时: {total_time/len(test_questions):.2f}s")
    print(f"总字符数: {total_tokens}")
    print(f"平均字符数: {total_tokens/len(test_questions):.0f}")
    print(f"生成速度: {total_tokens/total_time:.2f} 字符/秒")


def custom_parameters_example():
    """自定义参数示例"""
    print("=== 自定义参数示例 ===")
    
    client = VLLMStreamChat(**VLLM_CONFIG)
    
    question = "请写一个创意小故事，主题是未来的AI世界"
    
    # 不同参数设置的对比
    parameter_sets = [
        {"name": "保守设置", "temperature": 0.3, "top_p": 0.8},
        {"name": "平衡设置", "temperature": 0.7, "top_p": 0.9},
        {"name": "创意设置", "temperature": 1.0, "top_p": 0.95}
    ]
    
    for params in parameter_sets:
        print(f"\n--- {params['name']} ---")
        print(f"Temperature: {params['temperature']}, Top_p: {params['top_p']}")
        print("回答: ", end="", flush=True)
        
        for chunk in client.stream_chat(
            message=question,
            system_prompt=SYSTEM_PROMPTS["creative"],
            temperature=params["temperature"],
            top_p=params["top_p"],
            max_tokens=800,
            keep_history=False
        ):
            print(chunk, end="", flush=True)
        print("\n")


def error_handling_example():
    """错误处理示例"""
    print("=== 错误处理示例 ===")
    
    # 测试无效的配置
    invalid_config = VLLM_CONFIG.copy()
    invalid_config["base_url"] = "http://invalid-url:8000/v1"
    
    client = VLLMStreamChat(**invalid_config)
    
    print("测试无效连接...")
    try:
        response = ""
        for chunk in client.stream_chat(
            message="测试消息",
            max_tokens=100,
            keep_history=False
        ):
            response += chunk
        print(f"意外成功: {response}")
    except Exception as e:
        print(f"预期的错误: {e}")
    
    # 测试正常配置
    print("\n测试正常连接...")
    normal_client = VLLMStreamChat(**VLLM_CONFIG)
    try:
        response = normal_client.chat(
            message="简单测试",
            max_tokens=50,
            keep_history=False
        )
        print(f"连接正常: {response[:100]}...")
    except Exception as e:
        print(f"连接错误: {e}")


if __name__ == "__main__":
    import sys
    
    examples = {
        "basic": basic_usage_example,
        "conversation": conversation_example,
        "code": code_generation_example,
        "performance": performance_test,
        "parameters": custom_parameters_example,
        "error": error_handling_example
    }
    
    if len(sys.argv) > 1 and sys.argv[1] in examples:
        examples[sys.argv[1]]()
    else:
        print("=== vLLM Qwen3 流式调用示例 ===")
        print("可用示例:")
        for name, func in examples.items():
            print(f"  python examples.py {name}  # {func.__doc__}")
        
        print("\n运行所有示例...")
        for name, func in examples.items():
            try:
                func()
                print("\n" + "="*80 + "\n")
            except KeyboardInterrupt:
                print("\n用户中断")
                break
            except Exception as e:
                print(f"\n示例 {name} 出错: {e}")
                print("\n" + "="*80 + "\n")
