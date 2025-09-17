FROM python:3.10-alpine

# 定义工作目录变量
ARG APP_DIR=/app

LABEL maintainer="yongjie"

# 设置工作目录
WORKDIR ${APP_DIR}

# 复制依赖文件
COPY requirements.txt ${APP_DIR}/

# 安装依赖包
RUN pip install -r ${APP_DIR}/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/ --trusted-host pypi.tuna.tsinghua.edu.cn

# 复制应用程序文件
COPY . ${APP_DIR}/

# 给运行脚本添加执行权限
RUN chmod +x ${APP_DIR}/run.sh

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["sh", "${APP_DIR}/run.sh"]