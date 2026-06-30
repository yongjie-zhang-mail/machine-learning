# product-selling-points

从电商商品详情页长图中提取产品结构化卖点信息（基于 Qwen 多模态模型）。

## 环境要求

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/)

## 快速开始

1. 安装依赖并创建虚拟环境：

   ```bash
   uv sync
   ```

2. 配置环境变量，复制示例文件并填写实际值：

   ```bash
   cp .env.example .env
   ```

3. 运行：

   ```bash
   uv run product_selling_points.py
   ```

## 依赖管理

- 新增依赖：`uv add <package>`
- 移除依赖：`uv remove <package>`
- 同步环境：`uv sync`
