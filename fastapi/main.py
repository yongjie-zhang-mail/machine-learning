from fastapi import FastAPI
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
