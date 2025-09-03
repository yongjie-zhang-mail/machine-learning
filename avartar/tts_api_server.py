#!/usr/bin/env python3
"""
TTS API 服务器
使用 FastAPI 将 SimpleTTSClient 的 auto_speak 方法封装为 REST API
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
from simple_tts_client import SimpleTTSClient

# 创建 FastAPI 应用
app = FastAPI(
    title="TTS API Server",
    description="数字人TTS语音播报API服务",
    version="1.0.0"
)

# 全局 TTS 客户端实例
tts_client = SimpleTTSClient()

# 请求模型
class TTSRequest(BaseModel):
    message: str
    base_url: Optional[str] = None

# 响应模型
class TTSResponse(BaseModel):
    success: bool
    message: str
    session_id: Optional[str] = None

@app.get("/")
async def root():
    """API 根路径"""
    return {
        "message": "TTS API Server is running",
        "version": "1.0.0",
        "endpoints": {
            "speak": "/api/speak (POST)",
            "health": "/api/health (GET)",
            "docs": "/docs"
        }
    }

@app.get("/api/health")
async def health_check():
    """健康检查"""
    try:
        # 尝试获取会话来验证服务可用性
        session_id = tts_client.get_session_id()
        if session_id:
            return {
                "status": "healthy",
                "message": "TTS服务正常",
                "session_id": session_id
            }
        else:
            return {
                "status": "unhealthy",
                "message": "无法连接到TTS服务"
            }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"服务不可用: {str(e)}")

@app.post("/api/speak", response_model=TTSResponse)
async def speak_text(request: TTSRequest):
    """
    语音播报API
    
    Args:
        request: 包含要播报的消息和可选的服务器地址
        
    Returns:
        TTSResponse: 播报结果
    """
    try:
        # 如果提供了新的base_url，创建新的客户端
        if request.base_url:
            client = SimpleTTSClient(base_url=request.base_url)
        else:
            client = tts_client
        
        # 验证消息不为空
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="消息内容不能为空")
        
        # 调用 auto_speak 方法
        success = client.auto_speak(request.message)
        
        if success:
            return TTSResponse(
                success=True,
                message="播报成功",
                session_id=client.session_id
            )
        else:
            raise HTTPException(
                status_code=500, 
                detail="TTS播报失败，请检查服务状态"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")

@app.get("/api/sessions")
async def get_sessions():
    """获取活跃会话列表"""
    try:
        import requests
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        response = requests.get(f"{tts_client.base_url}/api/sessions", verify=False)
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code, 
                detail="获取会话列表失败"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取会话失败: {str(e)}")

# 添加中间件处理跨域
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def main():
    """启动 FastAPI 服务器"""
    print("🚀 启动 TTS API 服务器...")
    print("📖 API 文档地址: http://localhost:8000/docs")
    print("🔗 API 根地址: http://localhost:8000")
    print("🔊 语音播报API: POST http://localhost:8000/api/speak")
    
    uvicorn.run(
        "tts_api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()
