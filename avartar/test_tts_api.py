#!/usr/bin/env python3
"""
数字人TTS API测试脚本
"""

import requests
import json
import time
import urllib3

# 禁用SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class TTSAPITester:
    def __init__(self, base_url="https://82.156.1.74:8003"):
        self.base_url = base_url
        
    def health_check(self):
        """健康检查"""
        try:
            response = requests.get(f"{self.base_url}/api/health", verify=False)
            if response.status_code == 200:
                result = response.json()
                print(f"✅ 服务健康: {result}")
                return True
            else:
                print(f"❌ 健康检查失败: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ 连接失败: {e}")
            return False
    
    def get_sessions(self):
        """获取活跃会话列表"""
        try:
            response = requests.get(f"{self.base_url}/api/sessions", verify=False)
            if response.status_code == 200:
                result = response.json()
                print(f"📋 活跃会话: {json.dumps(result, indent=2, ensure_ascii=False)}")
                return result
            else:
                print(f"❌ 获取会话失败: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ 请求失败: {e}")
            return None
    
    def send_tts_message(self, text, session_id="direct_tts_session"):
        """发送TTS播报消息"""
        url = f"{self.base_url}/api/tts/speak"
        data = {
            "text": text,
            "session_id": session_id
        }
        
        try:
            print(f"🔊 发送TTS消息: {text}")
            response = requests.post(url, json=data, verify=False)
            if response.status_code == 200:
                result = response.json()
                print(f"✅ 成功: {result['message']}")
                return True
            else:
                print(f"❌ 失败: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"❌ 请求失败: {e}")
            return False
    
    def run_test(self):
        """运行完整测试"""
        print("=" * 50)
        print("🚀 开始TTS API测试")
        print("=" * 50)
        
        # 1. 健康检查
        print("\n1. 健康检查...")
        if not self.health_check():
            print("❌ 服务不可用，测试终止")
            return
        
        # 2. 获取会话列表
        print("\n2. 获取会话列表...")
        sessions = self.get_sessions()
        
        # 3. 发送测试消息
        print("\n3. 发送TTS测试消息...")
        test_messages = [
            "你好，这是第一条测试消息",
            "欢迎使用数字人TTS播报功能",
            "这是一个简单的API测试",
            "测试完成，谢谢使用"
        ]
        
        for i, message in enumerate(test_messages, 1):
            print(f"\n测试消息 {i}/{len(test_messages)}:")
            success = self.send_tts_message(message)
            if success:
                time.sleep(2)  # 等待2秒再发送下一条
            else:
                print("❌ 消息发送失败，跳过后续测试")
                break
        
        print("\n" + "=" * 50)
        print("🎉 测试完成")
        print("=" * 50)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="TTS API测试工具")
    parser.add_argument("--url", default="https://82.156.1.74:8003", help="服务器地址")
    parser.add_argument("--text", help="发送单条消息")
    parser.add_argument("--session", default="direct_tts_session", help="会话ID")
    
    args = parser.parse_args()
    
    tester = TTSAPITester(args.url)
    
    if args.text:
        # 发送单条消息
        tester.send_tts_message(args.text, args.session)
    else:
        # 运行完整测试
        tester.run_test()

if __name__ == "__main__":
    main()