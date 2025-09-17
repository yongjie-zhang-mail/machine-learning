#!/bin/bash

# FastAPI Server 启动脚本
# 用于启动 FastAPI 服务器

echo "正在启动 FastAPI 服务器..."
echo "服务器地址: http://localhost:8000"
echo "API 文档: http://localhost:8000/docs"
echo "按 Ctrl+C 停止服务器"
echo "----------------------------------------"

# 启动 FastAPI 服务器
python /app/main.py
