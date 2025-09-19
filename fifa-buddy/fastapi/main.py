from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fastapi_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 定义请求体模型
class Hello2Request(BaseModel):
    name: Optional[str] = "FastAPI"
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "World"
            }
        }

# 创建 FastAPI 实例
app = FastAPI(
    title="FastAPI Demo Server",
    description="一个简单的 FastAPI Rest API 服务器",
    version="1.0.0"
)

# 根路径接口 - 返回 API 信息
@app.get("/")
async def root():
    """
    根路径接口，返回 API 基本信息
    """
    logger.info("访问根路径接口 - 客户端请求 API 信息")
    
    response_data = {
        "message": "欢迎使用 FastAPI Demo Server",
        "version": "1.0.0",
        "endpoints": {
            "hello": "/hello",
            "hello2": "/hello2",
            "health": "/health",
            "docs": "/docs"
        }
    }
    
    logger.info(f"根路径接口响应成功 - 返回数据: {response_data}")
    return response_data

# Hello FastAPI 接口
@app.get("/hello")
async def hello_fastapi():
    """
    Hello FastAPI 接口，返回 hello fastapi
    """
    logger.info("访问 /hello 接口")
    
    response_data = {"message": "hello fastapi"}
    
    logger.info(f"/hello 接口响应成功 - 返回消息: {response_data['message']}")
    return response_data

# Hello2 FastAPI 接口 - 支持自定义名称 (POST 方式)
@app.post("/hello2")
async def hello2_fastapi(request: Hello2Request):
    """
    Hello2 FastAPI 接口，支持自定义名称参数 (POST 方式)
    
    请求体:
    - name (可选): 要问候的名称，默认值为 'FastAPI'
    
    返回:
    - 包含个性化问候消息的 JSON 响应
    """
    logger.info(f"访问 /hello2 接口 (POST) - 参数 name: {request.name}")
    
    response_data = {"message": f"hello {request.name}"}
    
    logger.info(f"/hello2 接口响应成功 - 返回消息: {response_data['message']}")
    return response_data

# 健康检查接口
@app.get("/health")
async def health_check():
    """
    健康检查接口
    """
    logger.info("执行健康检查 - 检查服务状态")
    
    # 模拟健康检查逻辑
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    response_data = {
        "status": "ok", 
        "message": "服务运行正常",
        "timestamp": current_time
    }
    
    logger.info(f"健康检查完成 - 服务状态: {response_data['status']}, 时间: {current_time}")
    return response_data

# 如果直接运行此文件，启动服务器
if __name__ == "__main__":
    import uvicorn
    logger.info("正在启动 FastAPI 服务器...")
    logger.info("服务器配置: host=0.0.0.0, port=8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
