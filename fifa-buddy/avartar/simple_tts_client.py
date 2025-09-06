#!/usr/bin/env python3
"""
精简的TTS API客户端
实现获取session_id并进行语音播报
"""

import requests
import urllib3

# 禁用SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class SimpleTTSClient:
    def __init__(self, base_url="https://82.156.1.74:8003"):
        self.base_url = base_url
        self.session_id = None
    
    def get_session_id(self):
        """获取第一个活跃的session_id"""
        try:
            response = requests.get(f"{self.base_url}/api/sessions", verify=False)
            if response.status_code == 200:
                data = response.json()
                sessions = data.get('sessions', [])
                if sessions:
                    self.session_id = sessions[0]['session_id']
                    print(f"✅ 获取到session_id: {self.session_id}")
                    return self.session_id
                else:
                    print("❌ 没有找到活跃会话")
                    return None
            else:
                print(f"❌ 获取会话失败: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ 请求失败: {e}")
            return None
    
    def speak(self, text):
        """使用获取到的session_id进行语音播报"""
        if not self.session_id:
            print("❌ 没有可用的session_id，请先获取")
            return False
        
        url = f"{self.base_url}/api/tts/speak"
        data = {
            "text": text,
            "session_id": self.session_id
        }
        
        try:
            response = requests.post(url, json=data, verify=False)
            if response.status_code == 200:
                result = response.json()
                print(f"✅ 播报成功: {text}")
                return True
            else:
                print(f"❌ 播报失败: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ 请求失败: {e}")
            return False
    
    def auto_speak(self, message):
        """自动获取session_id并进行语音播报"""
        # 如果没有session_id，先尝试获取
        if not self.session_id:
            if not self.get_session_id():
                print("❌ 无法获取session_id，播报失败")
                return False
        
        # 使用获取到的session_id进行播报
        return self.speak(message)

def test_tts_messages():
    """测试TTS播报功能"""
    # 创建客户端
    client = SimpleTTSClient()
    
    # 1. 获取session_id
    if client.get_session_id():
        # 2. 使用session_id进行语音播报
        test_messages = [
            "你好，这是自动获取会话ID的测试",
            "Hello, welcome to FIFA Avatar!",
            "测试完成，谢谢使用"
        ]
        
        for message in test_messages:
            client.speak(message)
    else:
        print("❌ 无法获取session_id，程序终止")

def test_auto_speak():
    """测试auto_speak方法（自动获取session_id并播报）"""
    print("=== 测试 auto_speak 方法 ===")
    
    # 创建客户端
    client = SimpleTTSClient()
    
    # 测试消息列表
    test_messages = [
        "这是auto_speak方法的第一条测试消息",
        "Auto speak test: Hello world!",
        "自动播报测试：功能正常",
        "Test completed successfully!"
    ]
    
    print(f"准备播报 {len(test_messages)} 条消息...")
    
    # 使用auto_speak方法逐条播报
    for i, message in enumerate(test_messages, 1):
        print(f"\n[{i}/{len(test_messages)}] 播报消息...")
        success = client.auto_speak(message)
        if not success:
            print(f"❌ 第{i}条消息播报失败，停止测试")
            break
    
    print("\n=== auto_speak 测试完成 ===")

def main():
    # 运行原始测试
    test_tts_messages()
    
    print("\n" + "="*50 + "\n")
    
    # 运行auto_speak测试
    test_auto_speak()

if __name__ == "__main__":
    main()
