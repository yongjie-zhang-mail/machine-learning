"""
Dify 客户端类
用于调用 Dify 对话消息接口
"""

import requests
import json
import time
from typing import Dict, List, Optional, Union, Iterator
from dataclasses import dataclass


@dataclass
class DifyResponse:
    """Dify 响应数据类"""
    event: str
    task_id: str
    id: str
    message_id: str
    conversation_id: str
    mode: str
    answer: str
    metadata: Dict
    created_at: int


class DifyClient:
    """
    Dify 客户端类
    用于调用 Dify API 接口
    """
    
    def __init__(self, api_key: str, base_url: str = "https://api.dify.ai/v1"):
        """
        初始化 Dify 客户端
        
        Args:
            api_key (str): Dify API 密钥
            base_url (str): API 基础 URL，默认为官方 API 地址
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def send_chat_message(
        self,
        query: str,
        user: str,
        inputs: Optional[Dict] = None,
        response_mode: str = "blocking",
        conversation_id: Optional[str] = None,
        files: Optional[List[Dict]] = None,
        auto_generate_name: bool = True
    ) -> Union[DifyResponse, Iterator[Dict]]:
        """
        发送对话消息
        
        Args:
            query (str): 用户输入/提问内容
            user (str): 用户标识，应用内唯一
            inputs (Optional[Dict]): 允许传入 App 定义的各变量值
            response_mode (str): 响应模式，"streaming" 或 "blocking"，默认 "blocking"
            conversation_id (Optional[str]): 会话 ID，用于继续之前的对话
            files (Optional[List[Dict]]): 文件列表，仅当模型支持 Vision 能力时可用
            auto_generate_name (bool): 自动生成会话标题，默认 True
            
        Returns:
            Union[DifyResponse, Iterator[Dict]]: 
                - blocking 模式返回 DifyResponse 对象
                - streaming 模式返回流式响应迭代器
        """
        url = f"{self.base_url}/chat-messages"
        
        # 构建请求体
        payload = {
            "query": query,
            "user": user,
            "response_mode": response_mode,
            "auto_generate_name": auto_generate_name
        }
        
        # 添加可选参数
        if inputs:
            payload["inputs"] = inputs
        if conversation_id:
            payload["conversation_id"] = conversation_id
        if files:
            payload["files"] = files
        
        try:
            if response_mode == "streaming":
                return self._handle_streaming_response(url, payload)
            else:
                return self._handle_blocking_response(url, payload)
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"请求失败: {str(e)}")
        except Exception as e:
            raise Exception(f"处理响应时出错: {str(e)}")
    
    def _handle_blocking_response(self, url: str, payload: Dict) -> DifyResponse:
        """
        处理阻塞模式响应
        
        Args:
            url (str): 请求 URL
            payload (Dict): 请求体
            
        Returns:
            DifyResponse: 响应数据对象
        """
        response = requests.post(url, headers=self.headers, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            return DifyResponse(
                event=data.get("event", ""),
                task_id=data.get("task_id", ""),
                id=data.get("id", ""),
                message_id=data.get("message_id", ""),
                conversation_id=data.get("conversation_id", ""),
                mode=data.get("mode", ""),
                answer=data.get("answer", ""),
                metadata=data.get("metadata", {}),
                created_at=data.get("created_at", int(time.time()))
            )
        else:
            raise Exception(f"API 请求失败，状态码: {response.status_code}, 响应: {response.text}")
    
    def _handle_streaming_response(self, url: str, payload: Dict) -> Iterator[Dict]:
        """
        处理流式响应
        
        Args:
            url (str): 请求 URL
            payload (Dict): 请求体
            
        Yields:
            Dict: 流式响应数据块
        """
        response = requests.post(
            url, 
            headers=self.headers, 
            json=payload, 
            stream=True
        )
        
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        try:
                            data = json.loads(line[6:])  # 去掉 'data: ' 前缀
                            yield data
                        except json.JSONDecodeError:
                            continue
        else:
            raise Exception(f"API 请求失败，状态码: {response.status_code}, 响应: {response.text}")
    
    def get_conversation_messages(self, conversation_id: str, user: str) -> Dict:
        """
        获取对话消息历史（如果 API 支持）
        
        Args:
            conversation_id (str): 会话 ID
            user (str): 用户标识
            
        Returns:
            Dict: 对话消息历史
        """
        url = f"{self.base_url}/messages"
        params = {
            "conversation_id": conversation_id,
            "user": user
        }
        
        response = requests.get(url, headers=self.headers, params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"获取对话历史失败，状态码: {response.status_code}, 响应: {response.text}")


# 使用示例
if __name__ == "__main__":
    # 初始化客户端
    api_key = "your-api-key-here"
    client = DifyClient(api_key)
    
    # 发送对话消息（阻塞模式）
    try:
        response = client.send_chat_message(
            query="你好，请介绍一下自己",
            user="user123",
            response_mode="blocking"
        )
        print(f"回答: {response.answer}")
        print(f"会话ID: {response.conversation_id}")
        
    except Exception as e:
        print(f"错误: {e}")
    
    # 发送对话消息（流式模式）
    try:
        for chunk in client.send_chat_message(
            query="请详细解释机器学习",
            user="user123",
            response_mode="streaming"
        ):
            if chunk.get("event") == "message":
                print(chunk.get("answer", ""), end="", flush=True)
                
    except Exception as e:
        print(f"流式响应错误: {e}")
