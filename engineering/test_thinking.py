#!/usr/bin/env python3
"""
思考模式测试脚本
"""

from stream_chat import VLLMStreamChat


def test_thinking_mode():
    """测试思考模式功能"""
    print("=== 思考模式测试 ===")
    
    # 创建客户端
    client = VLLMStreamChat(
        base_url="http://127.0.0.1:8002/v1",
        model="Qwen3-0.6B"
    )
    
    # 测试问题
    test_question = "9.9和9.11谁大？请详细分析"
    
    print(f"问题: {test_question}")
    print("\n🤔 思考过程:")
    print("-" * 50)
    
    # 测试思考模式
    try:
        for chunk_data in client.thinking_chat(
            message=test_question,
            system_prompt="你是一个逻辑思维严谨的AI助手，请仔细思考后给出准确的答案。",
            temperature=0.3,
            max_tokens=1024,
            keep_history=False
        ):
            if chunk_data["type"] == "thinking":
                print(chunk_data["content"], end="", flush=True)
            elif chunk_data["type"] == "answer":
                print("\n" + "-" * 50)
                print("💡 最终答案:")
                print(chunk_data["content"], end="", flush=True)
            elif chunk_data["type"] == "error":
                print(f"\n❌ {chunk_data['content']}")
        
        print("\n\n✅ 思考模式测试完成")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")


def test_normal_mode():
    """测试普通模式（带思考输出）"""
    print("\n=== 普通模式测试（显示思考） ===")
    
    client = VLLMStreamChat(
        base_url="http://127.0.0.1:8002/v1",
        model="Qwen3-0.6B"
    )
    
    test_question = "什么是机器学习？"
    
    print(f"问题: {test_question}")
    print("\n回答: ", end="", flush=True)
    
    try:
        for chunk in client.stream_chat(
            message=test_question,
            system_prompt="你是一个专业的AI助手。",
            temperature=0.5,
            max_tokens=800,
            enable_thinking=True,
            show_thinking=True,
            keep_history=False
        ):
            print(chunk, end="", flush=True)
        
        print("\n\n✅ 普通模式测试完成")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")


def test_no_thinking_mode():
    """测试关闭思考模式"""
    print("\n=== 关闭思考模式测试 ===")
    
    client = VLLMStreamChat(
        base_url="http://127.0.0.1:8002/v1",
        model="Qwen3-0.6B"
    )
    
    test_question = "9.9和9.11谁大？"
    
    print(f"问题: {test_question}")
    print("\n回答: ", end="", flush=True)
    
    try:
        for chunk in client.stream_chat(
            message=test_question,
            system_prompt="你是一个专业的AI助手。",
            temperature=0.5,
            max_tokens=500,
            enable_thinking=False,
            show_thinking=False,
            keep_history=False
        ):
            print(chunk, end="", flush=True)
        
        print("\n\n✅ 关闭思考模式测试完成")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "thinking":
            test_thinking_mode()
        elif sys.argv[1] == "normal":
            test_normal_mode()
        elif sys.argv[1] == "no-thinking":
            test_no_thinking_mode()
        else:
            print("使用方法:")
            print("python test_thinking.py thinking     # 测试思考模式")
            print("python test_thinking.py normal       # 测试普通模式（显示思考）")
            print("python test_thinking.py no-thinking  # 测试关闭思考模式")
    else:
        print("=== 思考功能全面测试 ===")
        test_thinking_mode()
        test_normal_mode()
        test_no_thinking_mode()
        print("\n🎉 所有测试完成！")
