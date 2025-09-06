# FIFA Buddy FastAPI Server

这是 FIFA Buddy 项目中基于 FastAPI 的 REST API 服务器，提供简单的 API 接口演示和服务健康检查功能。

## 项目特性

- ✅ 基于 FastAPI 框架构建
- ✅ 支持自动 API 文档生成 (Swagger UI)
- ✅ 完整的日志记录系统
- ✅ Pydantic 数据验证
- ✅ 健康检查接口
- ✅ 支持热重载开发模式

## 功能接口

- **GET /** - 返回 API 基本信息和可用端点
- **GET /hello** - 返回简单的问候消息
- **POST /hello2** - 支持自定义参数的个性化问候
- **GET /health** - 服务健康检查接口

## 环境要求

- Python 3.7+
- FastAPI 0.104.1+
- Uvicorn 0.24.0+

## 安装与运行

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动服务器

#### 方法一：直接运行 Python 文件
```bash
python main.py
```

#### 方法二：使用 uvicorn 命令
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### 方法三：使用启动脚本
```bash
chmod +x run.sh
./run.sh
```

## 服务访问

服务器启动后，可以通过以下方式访问：

- **根接口**: http://localhost:8000/ - API 基本信息
- **Hello 接口**: http://localhost:8000/hello - 简单问候
- **Hello2 接口**: http://localhost:8000/hello2 - 个性化问候 (POST)
- **健康检查**: http://localhost:8000/health - 服务状态检查
- **API 文档**: http://localhost:8000/docs - Swagger UI 交互式文档
- **ReDoc 文档**: http://localhost:8000/redoc - ReDoc 风格文档

## API 接口详情

### GET /
获取 API 基本信息和可用端点列表

**响应示例:**
```json
{
  "message": "欢迎使用 FastAPI Demo Server",
  "version": "1.0.0",
  "endpoints": {
    "hello": "/hello",
    "hello2": "/hello2", 
    "health": "/health",
    "docs": "/docs"
  }
}
```

### GET /hello
返回简单的问候消息

**响应示例:**
```json
{
  "message": "hello fastapi"
}
```

### POST /hello2
支持自定义参数的个性化问候接口

**请求体:**
```json
{
  "name": "World"
}
```

**响应示例:**
```json
{
  "message": "hello World"
}
```

**注意:** `name` 参数为可选，默认值为 "FastAPI"

### GET /health
服务健康检查接口，返回服务状态和时间戳

**响应示例:**
```json
{
  "status": "ok",
  "message": "服务运行正常", 
  "timestamp": "2025-09-06 14:30:25"
}
```

## 接口测试

### 使用 curl 测试

```bash
# 测试根接口 - 获取 API 信息
curl http://localhost:8000/

# 测试简单问候接口
curl http://localhost:8000/hello

# 测试个性化问候接口 (POST)
curl -X POST http://localhost:8000/hello2 \
  -H "Content-Type: application/json" \
  -d '{"name": "FIFA Buddy"}'

# 测试健康检查
curl http://localhost:8000/health
```

### 使用 httpie 测试

```bash
# 安装 httpie: pip install httpie

# 测试各接口
http GET localhost:8000/
http GET localhost:8000/hello
http POST localhost:8000/hello2 name="FIFA Buddy"
http GET localhost:8000/health
```

## 日志记录

服务器运行时会生成详细的日志记录：

- **控制台输出**: 实时显示请求和响应信息
- **日志文件**: `fastapi_server.log` - 持久化保存所有日志

日志包含：
- 接口访问记录
- 请求参数信息
- 响应数据内容
- 服务启动状态
- 错误信息追踪

## 开发说明

- 服务器默认运行在 `0.0.0.0:8000`
- 开发模式支持代码热重载
- 使用 Pydantic 进行数据验证
- 遵循 RESTful API 设计规范

## 项目结构

```
fastapi/
├── main.py              # 主程序文件
├── requirements.txt     # 依赖包列表
├── run.sh              # 启动脚本
├── README.md           # 项目说明文档
├── fastapi_server.log  # 日志文件 (运行时生成)
└── __pycache__/        # Python 缓存目录
```
