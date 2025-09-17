# ---------- Base stage: install dependencies ----------
FROM python:3.11-slim AS base

# ------- 可选国内加速配置（通过 --build-arg 覆盖） -------
ARG USE_CN_MIRROR=true
ARG PYPI_MIRROR_PRIMARY=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PYPI_MIRROR_FALLBACK=https://mirrors.aliyun.com/pypi/simple
ARG PIP_EXTRA_INDEX_URL=

# 设置时区与基础环境变量（可按需修改）
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    APP_HOME=/home/app

WORKDIR ${APP_HOME}

# 安装系统依赖（如需其他库可添加）
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# 仅复制 requirements 以便利用缓存
COPY requirements.txt ./
RUN set -eux; \
    pip install --upgrade pip; \
    if [ "$USE_CN_MIRROR" = "true" ]; then \
        echo "使用国内主镜像: $PYPI_MIRROR_PRIMARY"; \
        PIP_INDEX="-i $PYPI_MIRROR_PRIMARY"; \
        if [ -n "$PIP_EXTRA_INDEX_URL" ]; then \
           echo "附加额外索引: $PIP_EXTRA_INDEX_URL"; \
           PIP_INDEX="$PIP_INDEX --extra-index-url $PIP_EXTRA_INDEX_URL"; \
        fi; \
        if pip install $PIP_INDEX -r requirements.txt; then \
            echo "主镜像安装成功"; \
        else \
            echo "主镜像失败，尝试备用镜像: $PYPI_MIRROR_FALLBACK"; \
            pip install -i $PYPI_MIRROR_FALLBACK -r requirements.txt; \
        fi; \
    else \
        echo "使用官方 PyPI"; \
        pip install -r requirements.txt; \
    fi

# ---------- Runtime stage: 最小化镜像 ----------
FROM python:3.11-slim AS runtime
ENV APP_HOME=/home/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR ${APP_HOME}

# 创建非 root 用户提升安全性
RUN useradd -m -u 1000 appuser

# 复制已安装的依赖
COPY --from=base /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=base /usr/local/bin /usr/local/bin

# 复制应用源代码
COPY main.py ./
COPY run.sh ./

# 创建日志目录（若写文件日志）
RUN touch fastapi_server.log && chown appuser:appuser fastapi_server.log \
    && chmod +x run.sh

USER appuser

EXPOSE 8000

# 健康检查（Docker 原生命令，可选）
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# 默认启动命令（使用 uvicorn）
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
