# Simple TTS Client

一个精简的TTS（文本转语音）API客户端，用于与数字人TTS服务进行交互。

## 功能特性

- 🔍 **自动获取会话ID** - 从服务器获取活跃的session_id
- 🔊 **语音播报** - 使用获取到的session_id进行文本转语音
- 🛡️ **错误处理** - 完善的异常处理和状态反馈
- 📦 **精简设计** - 代码简洁，易于集成和使用

## 快速开始

### 环境要求

```bash
pip install requests urllib3
```

### 基本使用

```python
from simple_tts_client import SimpleTTSClient

# 创建客户端
client = SimpleTTSClient()

# 获取session_id
if client.get_session_id():
    # 进行语音播报
    client.speak("你好，这是一条测试消息")
```

### 直接运行

```bash
python simple_tts_client.py
```

## API 说明

### SimpleTTSClient 类

#### 初始化
```python
client = SimpleTTSClient(base_url="https://82.156.1.74:8003")
```

**参数:**
- `base_url` (str): TTS服务器地址，默认为 "https://82.156.1.74:8003"

#### 方法

##### get_session_id()
获取第一个活跃的session_id

**返回值:**
- `str`: 成功时返回session_id
- `None`: 失败时返回None

**示例:**
```python
session_id = client.get_session_id()
if session_id:
    print(f"获取到session_id: {session_id}")
```

##### speak(text)
使用当前session_id进行语音播报

**参数:**
- `text` (str): 要播报的文本内容

**返回值:**
- `bool`: 成功返回True，失败返回False

**示例:**
```python
success = client.speak("Hello, world!")
if success:
    print("播报成功")
```

## 使用示例

### 示例1: 基本使用
```python
#!/usr/bin/env python3
from simple_tts_client import SimpleTTSClient

def main():
    client = SimpleTTSClient()
    
    # 获取session_id
    if client.get_session_id():
        # 播报消息
        client.speak("欢迎使用TTS服务")
    else:
        print("无法连接到TTS服务")

if __name__ == "__main__":
    main()
```

### 示例2: 批量播报
```python
#!/usr/bin/env python3
from simple_tts_client import SimpleTTSClient
import time

def batch_speak():
    client = SimpleTTSClient()
    
    if not client.get_session_id():
        print("获取session_id失败")
        return
    
    messages = [
        "第一条消息",
        "第二条消息", 
        "第三条消息"
    ]
    
    for i, message in enumerate(messages, 1):
        print(f"播报第{i}条消息...")
        if client.speak(message):
            time.sleep(1)  # 等待1秒
        else:
            print(f"第{i}条消息播报失败")
            break

if __name__ == "__main__":
    batch_speak()
```

### 示例3: 自定义服务器地址
```python
#!/usr/bin/env python3
from simple_tts_client import SimpleTTSClient

def custom_server():
    # 使用自定义服务器地址
    client = SimpleTTSClient(base_url="https://your-server.com:8003")
    
    if client.get_session_id():
        client.speak("连接到自定义服务器成功")

if __name__ == "__main__":
    custom_server()
```

## 输出示例

```
✅ 获取到session_id: 104f7g
✅ 播报成功: 你好，这是自动获取会话ID的测试
✅ 播报成功: Hello, welcome to FIFA Avatar!
✅ 播报成功: 测试完成，谢谢使用
```

## 错误处理

客户端包含完善的错误处理机制：

- **网络连接错误**: 当无法连接到服务器时会显示相应错误信息
- **Session获取失败**: 当无法获取到活跃会话时会提示
- **播报失败**: 当TTS播报失败时会显示错误状态码
- **SSL证书**: 自动忽略SSL证书验证警告

## 注意事项

1. **网络连接**: 确保能够访问目标服务器的8003端口
2. **SSL证书**: 代码自动忽略SSL证书验证，适用于开发和测试环境
3. **会话管理**: 客户端会自动获取第一个活跃会话，如需指定特定会话请手动设置session_id
4. **编码支持**: 支持中文和英文文本播报

## 技术实现

- **HTTP客户端**: 使用requests库进行API调用
- **SSL处理**: 使用urllib3禁用SSL警告
- **JSON处理**: 自动处理JSON请求和响应
- **异常处理**: 完善的try-catch错误处理机制

## 相关文件

- `test_tts_api.py` - 完整的TTS API测试脚本
- `tts_api_curl_commands.md` - curl命令参考文档

## 许可证

本项目基于原有的TTS API测试代码开发，遵循相同的许可证条款。

---

*创建时间: 2025年9月3日*  
*基于: test_tts_api.py 和 tts_api_curl_commands.md*
