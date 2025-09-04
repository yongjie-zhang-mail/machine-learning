# FastAPI Rest API Server

这是一个基于 FastAPI 库的简单 Rest API 服务器。

## 功能

- **GET /** - 返回 API 基本信息和可用端点
- **GET /hello** - 返回 "hello fastapi" 消息
- **GET /health** - 健康检查接口

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行服务器

### 方法一：直接运行 Python 文件
```bash
python main.py
```

### 方法二：使用 uvicorn 命令
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 访问 API

服务器启动后，可以通过以下方式访问：

- **根接口**: http://localhost:8000/ (API 信息)
- **Hello 接口**: http://localhost:8000/hello (返回 hello fastapi)
- **健康检查**: http://localhost:8000/health
- **API 文档**: http://localhost:8000/docs (Swagger UI)
- **ReDoc 文档**: http://localhost:8000/redoc

## API 接口说明

### GET /
返回 API 基本信息和可用端点

**响应示例:**
```json
{
  "message": "欢迎使用 FastAPI Demo Server",
  "version": "1.0.0",
  "endpoints": {
    "hello": "/hello",
    "health": "/health",
    "docs": "/docs"
  }
}
```

### GET /hello
返回 hello fastapi 消息

**响应示例:**
```json
{
  "message": "hello fastapi"
}
```

### GET /health
健康检查接口

**响应示例:**
```json
{
  "status": "ok",
  "message": "服务运行正常"
}
```

## 测试

使用 curl 测试接口：

```bash
# 测试根接口 - API 信息
curl http://localhost:8000/

# 测试 hello 接口
curl http://localhost:8000/hello

# 测试健康检查
curl http://localhost:8000/health
```
