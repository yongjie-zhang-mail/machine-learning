"""
Dify 客户端使用示例
演示如何使用 DifyClient 类调用 Dify API
"""

import time
from dify_client import DifyClient
from config import DIFY_CONFIG, INPUT_TEMPLATES


def example_blocking_chat2():
    """阻塞模式对话示例"""
    print("=== 阻塞模式对话示例 ===")
    
    # 初始化客户端
    client = DifyClient(
        api_key=DIFY_CONFIG["api_key"],
        base_url=DIFY_CONFIG["base_url"]
    )
    
    try:
        # 发送第一条消息
        response = client.send_chat_message(
            query=" ",
            user=DIFY_CONFIG["default_user"],
            response_mode="blocking",
            inputs=INPUT_TEMPLATES["fifa_buddy_other_test"]
        )
        
        print(f"助手回答: {response.answer}")
        print(f"会话ID: {response.conversation_id}")
        print(f"消息ID: {response.message_id}")
        
        # 继续对话
        follow_up_response = client.send_chat_message(
            query=" ",
            user=DIFY_CONFIG["default_user"],
            response_mode="blocking",
            conversation_id=response.conversation_id,  # 使用相同的会话ID
            inputs=INPUT_TEMPLATES["fifa_buddy_other_test"]
        )
        
        print(f"\n继续对话回答: {follow_up_response.answer}")
        print(f"会话ID: {response.conversation_id}")
        print(f"消息ID: {response.message_id}")
        
        return response.conversation_id
        
    except Exception as e:
        print(f"阻塞模式对话错误: {e}")
        return None


def example_blocking_chat():
    """阻塞模式对话示例"""
    print("=== 阻塞模式对话示例 ===")
    
    # 初始化客户端
    client = DifyClient(
        api_key=DIFY_CONFIG["api_key"],
        base_url=DIFY_CONFIG["base_url"]
    )
    
    try:
        # 发送第一条消息
        response = client.send_chat_message(
            query="你好，我想了解一下足球比赛分析",
            user=DIFY_CONFIG["default_user"],
            response_mode="blocking",
            inputs=INPUT_TEMPLATES["simple_chat"]
        )
        
        print(f"助手回答: {response.answer}")
        print(f"会话ID: {response.conversation_id}")
        print(f"消息ID: {response.message_id}")
        
        # 继续对话
        follow_up_response = client.send_chat_message(
            query="请具体分析一下巴西队的战术特点",
            user=DIFY_CONFIG["default_user"],
            response_mode="blocking",
            conversation_id=response.conversation_id,  # 使用相同的会话ID
            inputs=INPUT_TEMPLATES["fifa_analysis"]
        )
        
        print(f"\n继续对话回答: {follow_up_response.answer}")
        
        return response.conversation_id
        
    except Exception as e:
        print(f"阻塞模式对话错误: {e}")
        return None


def example_streaming_chat():
    """流式模式对话示例"""
    print("\n=== 流式模式对话示例 ===")
    
    # 初始化客户端
    client = DifyClient(
        api_key=DIFY_CONFIG["api_key"],
        base_url=DIFY_CONFIG["base_url"]
    )
    
    try:
        print("问题: 请详细介绍世界杯的历史和发展")
        print("回答: ", end="")
        
        # 发送流式消息
        for chunk in client.send_chat_message(
            query="请详细介绍世界杯的历史和发展",
            user=DIFY_CONFIG["default_user"],
            response_mode="streaming",
            inputs=INPUT_TEMPLATES["with_context"]
        ):
            # 处理不同类型的事件
            if chunk.get("event") == "message":
                print(chunk.get("answer", ""), end="", flush=True)
            elif chunk.get("event") == "message_end":
                print(f"\n\n消息结束，会话ID: {chunk.get('conversation_id')}")
                break
            elif chunk.get("event") == "error":
                print(f"\n错误: {chunk.get('message')}")
                break
        
    except Exception as e:
        print(f"\n流式模式对话错误: {e}")


def example_with_context():
    """带上下文的对话示例"""
    print("\n=== 带上下文的对话示例 ===")
    
    client = DifyClient(
        api_key=DIFY_CONFIG["api_key"],
        base_url=DIFY_CONFIG["base_url"]
    )
    
    # 自定义输入变量
    custom_inputs = {
        "context": "用户是一个足球教练，正在准备重要比赛",
        "language": "zh-CN",
        "style": "专业、详细"
    }
    
    try:
        response = client.send_chat_message(
            query="我需要制定针对强队的防守战术，请给我一些建议",
            user="coach_user",
            response_mode="blocking",
            inputs=custom_inputs
        )
        
        print(f"专业建议: {response.answer}")
        
    except Exception as e:
        print(f"带上下文对话错误: {e}")


def example_error_handling():
    """错误处理示例"""
    print("\n=== 错误处理示例 ===")
    
    # 使用错误的 API 密钥
    client = DifyClient(api_key="invalid-api-key")
    
    try:
        response = client.send_chat_message(
            query="测试错误处理",
            user="test_user",
            response_mode="blocking"
        )
        print(f"不应该看到这条消息: {response.answer}")
        
    except Exception as e:
        print(f"预期的错误（API 密钥无效）: {e}")


def batch_conversation_example():
    """批量对话示例"""
    print("\n=== 批量对话示例 ===")
    
    client = DifyClient(
        api_key=DIFY_CONFIG["api_key"],
        base_url=DIFY_CONFIG["base_url"]
    )
    
    questions = [
        "什么是越位规则？",
        "如何提高射门精度？",
        "现代足球的战术演变是怎样的？"
    ]
    
    conversation_id = None
    
    for i, question in enumerate(questions, 1):
        try:
            response = client.send_chat_message(
                query=question,
                user="batch_user",
                response_mode="blocking",
                conversation_id=conversation_id
            )
            
            print(f"问题 {i}: {question}")
            print(f"回答 {i}: {response.answer[:100]}...")  # 只显示前100个字符
            print("-" * 50)
            
            # 保存会话ID以继续对话
            if conversation_id is None:
                conversation_id = response.conversation_id
                
        except Exception as e:
            print(f"问题 {i} 处理错误: {e}")


if __name__ == "__main__":
    # 检查配置
    if DIFY_CONFIG["api_key"] == "your-dify-api-key-here":
        print("⚠️  请先在 config.py 中设置正确的 API 密钥！")
        print("您需要：")
        print("1. 注册 Dify 账户")
        print("2. 创建应用并获取 API 密钥")
        print("3. 在 config.py 中替换 'your-dify-api-key-here' 为实际的 API 密钥")
        exit(1)
    
    print("开始运行 Dify 客户端示例...")
    
    # 运行各种示例
    # conversation_id = example_blocking_chat()
    # example_streaming_chat()
    # example_with_context()
    # example_error_handling()
    # batch_conversation_example()
    conversation_id = example_blocking_chat2()

    print("\n✅ 所有示例运行完成！")
