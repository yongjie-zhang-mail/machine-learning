# VS Code 工作区配置说明

此目录包含 VS Code 工作区的配置文件，用于自动设置 Python 环境和终端配置。

## 配置文件

### settings.json
- **python.defaultInterpreterPath**: 设置默认 Python 解释器为 lab conda 环境
- **terminal.integrated.defaultProfile.linux**: 设置默认终端配置为自动激活 lab 环境
- **python.terminal.activateEnvironment**: 自动在终端中激活 Python 环境

### launch.json  
- **Python: Current File**: 调试当前文件的配置
- **Python: FastAPI**: 专门用于调试 FastAPI 项目的配置

## 功能

✅ 新建终端自动激活 lab conda 环境
✅ Python 代码默认使用 lab 环境解释器  
✅ 调试时自动使用正确的 Python 环境
✅ 支持 FastAPI 项目的专门调试配置

## 使用方法

1. 重新启动 VS Code 或重新加载窗口 (Ctrl+Shift+P -> "Developer: Reload Window")
2. 打开新终端时会自动激活 lab conda 环境
3. 运行 Python 代码时会自动使用 lab 环境的解释器

## 验证设置

在新终端中运行以下命令验证环境：

```bash
# 检查当前 conda 环境
echo $CONDA_DEFAULT_ENV

# 检查 Python 路径
which python

# 检查 Python 版本
python --version
```

预期结果：
- CONDA_DEFAULT_ENV 应该显示 "lab"
- Python 路径应该指向 `/root/anaconda3/envs/lab/bin/python`
