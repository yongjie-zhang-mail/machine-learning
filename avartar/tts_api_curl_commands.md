# TTS API Curl 命令集合

基于 `test_tts_api.py` 生成的 curl 调用命令，用于测试数字人TTS API。

## 服务器信息
- **Base URL**: `https://82.156.1.74:8003/`
- **协议**: HTTPS (使用 `-k` 参数忽略SSL证书验证)

## API 端点说明

### 1. 健康检查 - GET /api/health
检查服务状态是否正常

**完整命令:**
```bash
curl -k -X GET "https://82.156.1.74:8003/api/health"
```

**简化命令:**
```bash
curl -k https://82.156.1.74:8003/api/health
```

---

### 2. 获取活跃会话列表 - GET /api/sessions
获取当前系统中的活跃会话信息

**完整命令:**
```bash
curl -k -X GET "https://82.156.1.74:8003/api/sessions"
```

**简化命令:**
```bash
curl -k https://82.156.1.74:8003/api/sessions
```

```json
**示例响应:**
{
    "sessions": [
        {
            "session_id": "104f7g",
            "handlers": [
                "LiteAvatar",
                "SileroVad", 
                "SenseVoice",
                "CosyVoice",
                "Dify",
                "LLMOpenAICompatible",
                "RtcClient"
            ],
            "active": true
        }
    ]
}
```


---

### 3. TTS播报消息 - POST /api/tts/speak
发送文本进行语音播报

**完整命令:**

```bash
curl -k -X POST "https://82.156.1.74:8003/api/tts/speak" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，这是一条测试消息",
    "session_id": "direct_tts_session"
  }'
```

```bash
curl -k -X POST "https://82.156.1.74:8003/api/tts/speak" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，这是一条测试消息",
    "session_id": "104f7g"
  }'
```

**简化命令:**
```bash
curl -k -H "Content-Type: application/json" -d '{"text":"你好，测试消息","session_id":"direct_tts_session"}' https://82.156.1.74:8003/api/tts/speak
```

## 请求参数说明

### TTS播报参数
- `text`: 要播报的文本内容 (必需)
- `session_id`: 会话ID (可选，默认: "direct_tts_session")

## 测试示例

### 快速测试脚本
```bash
#!/bin/bash

echo "=== TTS API 测试 ==="

echo "1. 健康检查..."
curl -k https://82.156.1.74:8003/api/health
echo -e "\n"

echo "2. 获取会话列表..."
curl -k https://82.156.1.74:8003/api/sessions
echo -e "\n"

echo "3. 发送TTS消息..."
curl -k -H "Content-Type: application/json" \
     -d '{"text":"API测试成功，服务运行正常","session_id":"test_session"}' \
     https://82.156.1.74:8003/api/tts/speak
echo -e "\n"

echo "=== 测试完成 ==="
```

## 常用测试消息

### 中文测试
```bash
# 简单问候
curl -k -H "Content-Type: application/json" -d '{"text":"你好，欢迎使用TTS服务"}' https://82.156.1.74:8003/api/tts/speak

# 数字播报
curl -k -H "Content-Type: application/json" -d '{"text":"现在时间是2025年9月3日"}' https://82.156.1.74:8003/api/tts/speak

# 长文本测试
curl -k -H "Content-Type: application/json" -d '{"text":"这是一个较长的文本测试，用于验证TTS系统处理长文本的能力和语音合成效果"}' https://82.156.1.74:8003/api/tts/speak
```

### 英文测试
```bash
# 英文问候
curl -k -H "Content-Type: application/json" -d '{"text":"Hello, welcome to TTS service"}' https://82.156.1.74:8003/api/tts/speak

# 数字英文
curl -k -H "Content-Type: application/json" -d '{"text":"Today is September 3rd, 2025"}' https://82.156.1.74:8003/api/tts/speak
```

```bash
curl -k -X POST "https://82.156.1.74:8003/api/tts/speak" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, welcome to FIFA Avartar~ ",
    "session_id": "104f7g"
  }'
```




## 注意事项

1. **SSL证书**: 使用 `-k` 参数忽略SSL证书验证
2. **Content-Type**: POST请求必须设置 `Content-Type: application/json`
3. **JSON格式**: 请求体必须是有效的JSON格式
4. **中文编码**: 确保终端支持UTF-8编码以正确显示中文
5. **网络连接**: 确保能够访问目标服务器的8003端口

## 响应格式

### 成功响应示例
```json
{
  "status": "success",
  "message": "消息已发送",
  "session_id": "direct_tts_session"
}
```

### 错误响应示例
```json
{
  "status": "error",
  "message": "错误描述",
  "code": "ERROR_CODE"
}
```

---

*文件生成时间: 2025年9月3日*  
*基于: test_tts_api.py*
