# vLLM Qwen3 流式调用工具

这个工具集提供了对 vLLM 部署的 Qwen3 模型进行流式调用的完整解决方案。

## 功能特性

- ✅ 流式对话支持
- ✅ 对话历史管理
- ✅ 多种系统提示词模板
- ✅ 灵活的参数配置
- ✅ 错误处理和重试机制
- ✅ 性能测试和监控
- ✅ 交互式聊天界面

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 启动 vLLM 服务

首先确保 vLLM 服务正在运行，默认地址为 `http://localhost:8000`：

```bash
# 示例启动命令（根据实际情况调整）
vllm serve Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8000
```

### 2. 测试连接

```bash
python stream_chat.py test
```

### 3. 开始对话

```bash
# 交互式聊天
python stream_chat.py chat

# 或者直接运行
python stream_chat.py
```

### 4. 运行演示

```bash
# 流式调用演示
python stream_chat.py demo

# 各种使用示例
python examples.py
```

## 使用方法

### 基础流式调用

```python
from stream_chat import VLLMStreamChat

# 创建客户端
client = VLLMStreamChat(
    base_url="http://localhost:8000/v1",
    model="Qwen/Qwen2.5-7B-Instruct"
)

# 流式对话
for chunk in client.stream_chat("你好，请介绍一下自己"):
    print(chunk, end="", flush=True)
```

### 非流式调用

```python
# 一次性获取完整回复
response = client.chat("什么是人工智能？")
print(response)
```

### 自定义参数

```python
for chunk in client.stream_chat(
    message="写一个 Python 排序算法",
    system_prompt="你是一个专业的编程助手",
    temperature=0.3,
    max_tokens=1024,
    top_p=0.9
):
    print(chunk, end="", flush=True)
```

### 对话历史管理

```python
# 清空历史
client.clear_history()

# 获取历史
history = client.get_history()

# 设置历史
client.set_history(history)
```

## 配置选项

### 连接配置

在 `config.py` 中修改连接设置：

```python
VLLM_CONFIG = {
    "base_url": "http://localhost:8000/v1",  # vLLM 服务地址
    "api_key": "EMPTY",                      # API 密钥
    "model": "Qwen/Qwen2.5-7B-Instruct",    # 模型名称
    "timeout": 60.0                          # 超时时间
}
```

### 生成参数

```python
DEFAULT_PARAMS = {
    "temperature": 0.7,        # 温度（0-2，越高越随机）
    "max_tokens": 2048,        # 最大生成 token 数
    "top_p": 0.9,             # Top-p 采样
    "frequency_penalty": 0.0,  # 频率惩罚
    "presence_penalty": 0.0    # 存在惩罚
}
```

## 命令行使用

### 主程序

```bash
python stream_chat.py [command]

# 可用命令：
# test  - 测试连接
# demo  - 演示流式调用
# chat  - 交互式聊天
```

### 示例程序

```bash
python examples.py [example]

# 可用示例：
# basic        - 基础使用示例
# conversation - 连续对话示例
# code         - 代码生成示例
# performance  - 性能测试
# parameters   - 自定义参数示例
# error        - 错误处理示例
```

## 高级功能

### 1. 系统提示词模板

预定义了多种系统提示词：

- `default`: 通用助手
- `coding`: 编程助手
- `academic`: 学术助手
- `creative`: 创意助手

### 2. 性能监控

内置性能测试功能，可以监控：

- 响应时间
- 生成速度
- Token 使用量

### 3. 错误处理

完善的错误处理机制：

- 连接错误重试
- 超时处理
- 异常捕获和日志

## 故障排除

### 常见问题

1. **连接失败**
   - 检查 vLLM 服务是否运行
   - 确认服务地址和端口
   - 检查防火墙设置

2. **模型名称错误**
   - 确认模型在 vLLM 中正确加载
   - 检查模型名称拼写

3. **生成速度慢**
   - 检查服务器资源
   - 调整 max_tokens 参数
   - 优化 vLLM 启动参数

### 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 测试连接
client.chat("测试", max_tokens=10, keep_history=False)
```

## 注意事项

1. 确保 vLLM 服务正在运行
2. 根据实际部署修改 `base_url` 和 `model` 参数
3. 合理设置 `max_tokens` 以避免过长等待
4. 在生产环境中考虑添加重试和错误恢复机制
5. 注意 API 调用频率限制（如果有）

## 扩展开发

可以基于此工具进行扩展：

- 添加更多的系统提示词模板
- 实现对话数据的持久化存储
- 集成到 Web 应用或 API 服务
- 添加多模态支持（如图像输入）
- 实现批量处理功能
