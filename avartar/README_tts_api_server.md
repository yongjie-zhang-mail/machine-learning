# TTS API Server

基于 FastAPI 构建的TTS（文本转语音）REST API服务器，将 SimpleTTSClient 的功能封装为Web API，提供便捷的语音播报服务。

## 🚀 功能特性

- 🎯 **RESTful API** - 标准的REST API接口
- 🔊 **语音播报** - 支持中英文文本转语音
- 🔍 **自动会话管理** - 自动获取和管理session_id
- 📖 **自动文档** - Swagger UI自动生成API文档
- ⚡ **高性能** - 基于FastAPI异步框架
- 🛡️ **数据验证** - Pydantic模型自动验证请求数据
- 🌐 **跨域支持** - 配置CORS支持前端调用
- 📊 **健康检查** - 提供服务状态监控

## 📋 API 端点

| 方法 | 端点 | 功能 | 描述 |
|------|------|------|------|
| GET | `/` | 根路径 | API信息和端点列表 |
| POST | `/api/speak` | 语音播报 | 核心TTS功能 |
| GET | `/api/health` | 健康检查 | 服务状态监控 |
| GET | `/api/sessions` | 会话列表 | 获取活跃会话 |
| GET | `/docs` | API文档 | Swagger UI文档 |

## 🛠️ 安装和配置

### 环境要求

```bash
python >= 3.7
```

### 安装依赖

```bash
# 安装核心依赖
pip install fastapi uvicorn

# 安装基础依赖
pip install requests urllib3
```

### 文件依赖

确保以下文件在同一目录下：
- `simple_tts_client.py` - TTS客户端
- `tts_api_server.py` - API服务器

## 🚀 启动服务

### 方法1: 直接运行
```bash
python tts_api_server.py
```

### 方法2: 使用uvicorn
```bash
uvicorn tts_api_server:app --host 0.0.0.0 --port 8000 --reload
```

### 启动信息
```
🚀 启动 TTS API 服务器...
📖 API 文档地址: http://localhost:8000/docs
🔗 API 根地址: http://localhost:8000
🔊 语音播报API: POST http://localhost:8000/api/speak
```

## 📡 API 使用说明

### 1. 语音播报 - POST /api/speak

**请求格式:**
```json
{
  "message": "要播报的文本内容",
  "base_url": "https://82.156.1.74:8003"  // 可选
}
```

**响应格式:**
```json
{
  "success": true,
  "message": "播报成功",
  "session_id": "104f7g"
}
```

**curl 示例:**
```bash
curl -X POST "http://localhost:8000/api/speak" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "你好，这是通过FastAPI调用的语音播报"
  }'
```

### 2. 健康检查 - GET /api/health

**响应示例:**
```json
{
  "status": "healthy",
  "message": "TTS服务正常",
  "session_id": "104f7g"
}
```

**curl 示例:**
```bash
curl http://localhost:8000/api/health
```

### 3. 获取会话列表 - GET /api/sessions

**响应示例:**
```json
{
  "sessions": [
    {
      "session_id": "104f7g",
      "handlers": ["LiteAvatar", "SileroVad", "SenseVoice"],
      "active": true
    }
  ]
}
```

## 💻 客户端使用示例

### Python 客户端
```python
import requests

# 创建客户端
base_url = "http://localhost:8000"

def speak_text(message, server_url=None):
    """调用语音播报API"""
    data = {"message": message}
    if server_url:
        data["base_url"] = server_url
    
    response = requests.post(f"{base_url}/api/speak", json=data)
    return response.json()

def check_health():
    """检查服务健康状态"""
    response = requests.get(f"{base_url}/api/health")
    return response.json()

# 使用示例
if __name__ == "__main__":
    # 健康检查
    health = check_health()
    print(f"服务状态: {health}")
    
    # 语音播报
    result = speak_text("Hello, FastAPI TTS!")
    print(f"播报结果: {result}")
```

### JavaScript 客户端
```javascript
// 语音播报函数
async function speakText(message, baseUrl = null) {
    const data = { message };
    if (baseUrl) data.base_url = baseUrl;
    
    try {
        const response = await fetch('http://localhost:8000/api/speak', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        console.log('播报结果:', result);
        return result;
    } catch (error) {
        console.error('API调用失败:', error);
    }
}

// 使用示例
speakText("Hello from JavaScript!");
```

### 批量播报示例
```python
import requests
import time

def batch_speak(messages, delay=1):
    """批量语音播报"""
    base_url = "http://localhost:8000"
    
    for i, message in enumerate(messages, 1):
        print(f"[{i}/{len(messages)}] 播报: {message}")
        
        response = requests.post(f"{base_url}/api/speak", 
                               json={"message": message})
        result = response.json()
        
        if result.get("success"):
            print(f"✅ 播报成功")
        else:
            print(f"❌ 播报失败: {result}")
        
        if i < len(messages):  # 最后一条不需要等待
            time.sleep(delay)

# 使用示例
messages = [
    "欢迎使用TTS API服务",
    "这是第二条测试消息",
    "批量播报测试完成"
]
batch_speak(messages)
```

## 🔧 配置选项

### 服务器配置
```python
# 在 main() 函数中修改以下参数
uvicorn.run(
    "tts_api_server:app",
    host="0.0.0.0",        # 监听地址
    port=8000,             # 端口号
    reload=True,           # 开发模式自动重载
    log_level="info"       # 日志级别
)
```

### TTS服务器配置
```python
# 默认TTS服务器地址
tts_client = SimpleTTSClient(base_url="https://82.156.1.74:8003")

# 也可以在API调用时动态指定
{
  "message": "test",
  "base_url": "https://your-custom-server:8003"
}
```

## 🐳 Docker 部署

### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "tts_api_server.py"]
```

### requirements.txt
```
fastapi==0.104.1
uvicorn==0.24.0
requests==2.31.0
urllib3==2.1.0
```

### 构建和运行
```bash
# 构建镜像
docker build -t tts-api-server .

# 运行容器
docker run -p 8000:8000 tts-api-server
```

## 📊 性能和监控

### 性能特点
- **异步处理**: FastAPI基于Starlette异步框架
- **自动文档**: 零配置Swagger UI文档
- **数据验证**: Pydantic自动验证，减少错误
- **连接复用**: HTTP客户端连接复用

### 监控端点
- **健康检查**: `GET /api/health`
- **服务信息**: `GET /`
- **会话状态**: `GET /api/sessions`

## 🚨 错误处理

### 常见错误码
| 状态码 | 错误 | 描述 |
|--------|------|------|
| 400 | Bad Request | 请求参数错误 |
| 500 | Internal Server Error | 内部服务器错误 |
| 503 | Service Unavailable | TTS服务不可用 |

### 错误响应格式
```json
{
  "detail": "错误描述信息"
}
```

## 🔒 安全注意事项

1. **生产环境**: 修改CORS配置，限制允许的域名
2. **认证授权**: 根据需要添加API密钥或JWT认证
3. **限流控制**: 考虑添加请求频率限制
4. **HTTPS**: 生产环境使用HTTPS协议

## 📝 开发和贡献

### 项目结构
```
avartar/
├── simple_tts_client.py          # TTS客户端
├── tts_api_server.py             # FastAPI服务器
├── test_tts_api.py               # 原始测试脚本
├── tts_api_curl_commands.md      # curl命令文档
└── README_tts_api_server.md      # 本文档
```

### 开发模式
```bash
# 启动开发服务器（自动重载）
uvicorn tts_api_server:app --reload --host 0.0.0.0 --port 8000
```

### 测试
```bash
# 运行基础TTS客户端测试
python simple_tts_client.py

# 测试API服务器
python -c "
import requests
response = requests.post('http://localhost:8000/api/speak', 
                        json={'message': 'API测试'})
print(response.json())
"
```

## 🆘 故障排除

### 常见问题

**1. 无法启动服务器**
```bash
# 检查端口是否被占用
lsof -i :8000

# 使用其他端口
uvicorn tts_api_server:app --port 8001
```

**2. TTS服务连接失败**
- 检查TTS服务器地址和端口
- 验证网络连接
- 查看API响应的错误信息

**3. 导入模块错误**
```bash
# 确保simple_tts_client.py在同一目录
ls -la simple_tts_client.py

# 检查Python路径
python -c "import sys; print(sys.path)"
```

## 📄 许可证

本项目基于原有的TTS客户端代码开发，遵循相同的许可证条款。

---

*创建时间: 2025年9月4日*  
*基于: simple_tts_client.py 和 FastAPI框架*
