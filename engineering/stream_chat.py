#!/usr/bin/env python3
"""
vLLM 部署的 Qwen3 模型流式调用示例
使用 OpenAI Python 包进行流式对话
"""

import os
import sys
import json
import time
from typing import Iterator, Dict, Any, Optional, List
from openai import OpenAI


class VLLMStreamChat:
    """vLLM 流式聊天客户端"""
    
    def __init__(
        self, 
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        timeout: float = 60.0
    ):
        """
        初始化 vLLM 流式聊天客户端
        
        Args:
            base_url: vLLM 服务的基础 URL
            api_key: API 密钥（vLLM 本地部署通常使用 "EMPTY"）
            model: 模型名称
            timeout: 请求超时时间
        """
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout
        )
        self.model = model
        self.conversation_history: List[Dict[str, str]] = []
    
    def stream_chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        keep_history: bool = True
    ) -> Iterator[str]:
        """
        进行流式对话
        
        Args:
            message: 用户消息
            system_prompt: 系统提示词
            temperature: 温度参数
            max_tokens: 最大生成 token 数
            top_p: top_p 参数
            frequency_penalty: 频率惩罚
            presence_penalty: 存在惩罚
            keep_history: 是否保持对话历史
            
        Yields:
            str: 流式返回的文本片段
        """
        try:
            # 构建消息列表
            messages = []
            
            # 添加系统提示词
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # 添加历史对话
            if keep_history:
                messages.extend(self.conversation_history)
            
            # 添加当前用户消息
            messages.append({"role": "user", "content": message})
            
            # 创建流式聊天完成
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stream=True
            )
            
            # 收集完整回复用于历史记录
            full_response = ""
            
            # 流式输出
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield content
            
            # 更新对话历史
            if keep_history:
                self.conversation_history.append({"role": "user", "content": message})
                self.conversation_history.append({"role": "assistant", "content": full_response})
                
        except Exception as e:
            yield f"错误: {str(e)}"
    
    def chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        keep_history: bool = True
    ) -> str:
        """
        非流式对话（一次性返回完整回复）
        
        Args:
            message: 用户消息
            system_prompt: 系统提示词
            temperature: 温度参数
            max_tokens: 最大生成 token 数
            top_p: top_p 参数
            frequency_penalty: 频率惩罚
            presence_penalty: 存在惩罚
            keep_history: 是否保持对话历史
            
        Returns:
            str: 完整的回复
        """
        response = ""
        for chunk in self.stream_chat(
            message=message,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            keep_history=keep_history
        ):
            response += chunk
        return response
    
    def clear_history(self):
        """清空对话历史"""
        self.conversation_history.clear()
    
    def get_history(self) -> List[Dict[str, str]]:
        """获取对话历史"""
        return self.conversation_history.copy()
    
    def set_history(self, history: List[Dict[str, str]]):
        """设置对话历史"""
        self.conversation_history = history


def interactive_chat():
    """交互式聊天演示"""
    print("=== vLLM Qwen3 流式聊天演示 ===")
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'clear' 清空对话历史")
    print("输入 'history' 查看对话历史")
    print("-" * 50)
    
    # 创建聊天客户端
    chat_client = VLLMStreamChat(
        base_url="http://localhost:8000/v1",  # 根据实际部署修改
        model="Qwen/Qwen2.5-7B-Instruct"     # 根据实际模型修改
    )
    
    # 系统提示词
    system_prompt = "你是一个有用的AI助手。请用中文回答问题，保持友好和专业。"
    
    while True:
        try:
            # 获取用户输入
            user_input = input("\n用户: ").strip()
            
            # 处理特殊命令
            if user_input.lower() in ['quit', 'exit']:
                print("再见！")
                break
            elif user_input.lower() == 'clear':
                chat_client.clear_history()
                print("对话历史已清空")
                continue
            elif user_input.lower() == 'history':
                history = chat_client.get_history()
                print(f"对话历史 ({len(history)} 条消息):")
                for i, msg in enumerate(history):
                    print(f"{i+1}. {msg['role']}: {msg['content'][:100]}...")
                continue
            elif not user_input:
                continue
            
            # 流式输出回复
            print("助手: ", end="", flush=True)
            for chunk in chat_client.stream_chat(
                message=user_input,
                system_prompt=system_prompt if len(chat_client.get_history()) == 0 else None,
                temperature=0.7,
                max_tokens=2048
            ):
                print(chunk, end="", flush=True)
            print()  # 换行
            
        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            break
        except Exception as e:
            print(f"\n错误: {e}")


def demo_streaming():
    """流式调用演示"""
    print("=== 流式调用演示 ===")
    
    # 创建聊天客户端
    chat_client = VLLMStreamChat(
        base_url="http://localhost:8000/v1",
        model="Qwen/Qwen2.5-7B-Instruct"
    )
    
    # 示例问题
    questions = [
        "请介绍一下人工智能的发展历史",
        "Python 有哪些主要的机器学习库？",
        "解释一下什么是 Transformer 架构"
    ]
    
    system_prompt = "你是一个专业的AI技术专家，请提供准确、详细的技术解答。"
    
    for i, question in enumerate(questions, 1):
        print(f"\n问题 {i}: {question}")
        print("回答: ", end="", flush=True)
        
        start_time = time.time()
        response_length = 0
        
        for chunk in chat_client.stream_chat(
            message=question,
            system_prompt=system_prompt if i == 1 else None,
            temperature=0.7,
            max_tokens=1024
        ):
            print(chunk, end="", flush=True)
            response_length += len(chunk)
        
        end_time = time.time()
        print(f"\n[耗时: {end_time - start_time:.2f}s, 长度: {response_length} 字符]")
        print("-" * 80)


def test_connection():
    """测试连接"""
    print("=== 测试 vLLM 连接 ===")
    
    try:
        chat_client = VLLMStreamChat(
            base_url="http://localhost:8000/v1",
            model="Qwen/Qwen2.5-7B-Instruct"
        )
        
        print("正在测试连接...")
        response = chat_client.chat(
            message="你好，请简单介绍一下自己",
            temperature=0.7,
            max_tokens=200,
            keep_history=False
        )
        
        print("连接成功！")
        print(f"模型回复: {response}")
        
    except Exception as e:
        print(f"连接失败: {e}")
        print("\n请检查:")
        print("1. vLLM 服务是否正在运行")
        print("2. 服务地址是否正确 (默认: http://localhost:8000)")
        print("3. 模型名称是否正确")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            test_connection()
        elif sys.argv[1] == "demo":
            demo_streaming()
        elif sys.argv[1] == "chat":
            interactive_chat()
        else:
            print("使用方法:")
            print("python stream_chat.py test   # 测试连接")
            print("python stream_chat.py demo   # 演示流式调用")
            print("python stream_chat.py chat   # 交互式聊天")
    else:
        print("=== vLLM Qwen3 流式调用工具 ===")
        print("使用方法:")
        print("python stream_chat.py test   # 测试连接")
        print("python stream_chat.py demo   # 演示流式调用")
        print("python stream_chat.py chat   # 交互式聊天")
        print("\n开始交互式聊天...")
        interactive_chat()
