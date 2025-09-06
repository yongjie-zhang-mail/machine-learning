# Dify 客户端

这是一个用于调用 Dify 对话消息 API 的 Python 客户端库。

## 文件结构

```
dify/
├── dify_client.py     # 主要的 Dify 客户端类
├── config.py          # 配置文件
├── example_usage.py   # 使用示例
└── README.md          # 说明文档
```

## 功能特性

- ✅ 支持阻塞模式和流式模式的对话
- ✅ 支持会话上下文管理
- ✅ 支持自定义输入变量
- ✅ 支持文件上传（Vision 功能）
- ✅ 完整的错误处理
- ✅ 类型提示支持
- ✅ 详细的使用示例

## 快速开始

### 1. 安装依赖

```bash
pip install requests
```

### 2. 配置 API 密钥

编辑 `config.py` 文件，替换您的 Dify API 密钥：

```python
DIFY_CONFIG = {
    "api_key": "your-actual-dify-api-key",  # 替换为您的实际 API 密钥
    "base_url": "https://api.dify.ai/v1",    # 或您的自部署地址
    # ...
}
```

### 3. 基本使用

```python
from dify_client import DifyClient

# 初始化客户端
client = DifyClient(api_key="your-api-key")

# 发送对话消息（阻塞模式）
response = client.send_chat_message(
    query="你好，请介绍一下自己",
    user="user123",
    response_mode="blocking"
)

print(response.answer)
```

## API 参考

### DifyClient 类

#### 初始化
```python
client = DifyClient(api_key: str, base_url: str = "https://api.dify.ai/v1")
```

#### send_chat_message 方法

```python
def send_chat_message(
    self,
    query: str,                           # 必需：用户输入内容
    user: str,                            # 必需：用户标识
    inputs: Optional[Dict] = None,        # 可选：输入变量
    response_mode: str = "blocking",      # 可选：响应模式
    conversation_id: Optional[str] = None, # 可选：会话 ID
    files: Optional[List[Dict]] = None,   # 可选：文件列表
    auto_generate_name: bool = True       # 可选：自动生成会话标题
) -> Union[DifyResponse, Iterator[Dict]]
```

**参数说明：**

- `query`: 用户的问题或输入内容
- `user`: 用户的唯一标识符
- `inputs`: 传递给应用的变量值字典
- `response_mode`: 
  - `"blocking"`: 等待完整响应后返回
  - `"streaming"`: 流式返回响应
- `conversation_id`: 用于继续之前的对话
- `files`: 文件列表，用于支持 Vision 功能的模型
- `auto_generate_name`: 是否自动生成会话标题

**返回值：**

- 阻塞模式：返回 `DifyResponse` 对象
- 流式模式：返回响应数据的迭代器

### DifyResponse 对象

```python
@dataclass
class DifyResponse:
    event: str              # 事件类型
    task_id: str           # 任务 ID
    id: str                # 唯一 ID
    message_id: str        # 消息 ID
    conversation_id: str   # 会话 ID
    mode: str              # 应用模式
    answer: str            # 回答内容
    metadata: Dict         # 元数据（包含使用情况等）
    created_at: int        # 创建时间戳
```

## 使用示例

### 阻塞模式对话

```python
from dify_client import DifyClient

client = DifyClient(api_key="your-api-key")

response = client.send_chat_message(
    query="什么是机器学习？",
    user="student123",
    response_mode="blocking"
)

print(f"回答: {response.answer}")
print(f"会话ID: {response.conversation_id}")
```

### 流式模式对话

```python
for chunk in client.send_chat_message(
    query="请详细解释深度学习",
    user="student123",
    response_mode="streaming"
):
    if chunk.get("event") == "message":
        print(chunk.get("answer", ""), end="", flush=True)
```

### 带上下文的对话

```python
# 自定义输入变量
inputs = {
    "context": "用户是机器学习初学者",
    "language": "zh-CN",
    "difficulty": "beginner"
}

response = client.send_chat_message(
    query="请解释神经网络",
    user="beginner_user",
    inputs=inputs,
    response_mode="blocking"
)
```

### 继续对话

```python
# 第一轮对话
first_response = client.send_chat_message(
    query="什么是足球战术？",
    user="coach123",
    response_mode="blocking"
)

# 继续对话
second_response = client.send_chat_message(
    query="请详细说明 4-3-3 阵型",
    user="coach123",
    conversation_id=first_response.conversation_id,  # 使用相同会话ID
    response_mode="blocking"
)
```

## 错误处理

```python
try:
    response = client.send_chat_message(
        query="你好",
        user="user123"
    )
    print(response.answer)
    
except Exception as e:
    print(f"请求失败: {e}")
```

## 配置说明

在 `config.py` 中可以配置：

- API 密钥和基础 URL
- 默认用户标识
- 请求超时时间
- 重试次数
- 预设的输入变量模板
- 文件上传配置

## 运行示例

```bash
# 运行完整示例
python example_usage.py
```

**注意：** 运行前请确保在 `config.py` 中设置了正确的 API 密钥。

## 获取 Dify API 密钥

1. 访问 [Dify 官网](https://dify.ai/)
2. 注册并登录账户
3. 创建应用
4. 在应用设置中找到 API 密钥
5. 将密钥复制到 `config.py` 文件中

## 注意事项

- API 密钥请妥善保管，不要提交到公共代码仓库
- 流式模式适合长回答，阻塞模式适合短回答
- 每个用户标识符在应用内应该唯一
- 文件上传功能需要模型支持 Vision 能力

## 许可证

MIT License
