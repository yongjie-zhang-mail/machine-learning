# product-selling-points

基于 Qwen 多模态模型，对电商商品详情页长图进行分析，支持三类任务：

- **extract**：提取各产品的结构化卖点
- **compare**：多个产品核心卖点横评
- **ai-ecosystem**：提取产品中 AI 程序与软件生态相关内容并总结

也可用 **all** 命令一次性依次执行以上全部三个任务。

## 环境要求

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/)

## 快速开始

1. 安装依赖并创建虚拟环境：

   ```bash
   uv sync
   ```

2. 配置密钥与模型（复制示例文件并填写实际值）：

   ```bash
   cp .env.example .env
   ```

3. 配置待分析的商品图片（复制示例文件并填写实际 URL）：

   ```bash
   cp products.example.json products.json
   ```

4. 运行指定任务：

   ```bash
   uv run product_selling_points.py extract
   uv run product_selling_points.py compare
   uv run product_selling_points.py ai-ecosystem
   # 依次执行以上全部三个任务
   uv run product_selling_points.py all
   ```

## 配置说明

运行所需参数分两处管理，命令行只接受任务名，其余参数全部来自配置文件或默认值。

### `.env`（密钥与模型）

| 变量 | 说明 | 默认值 |
| --- | --- | --- |
| `QWEN_ENDPOINT` | Qwen 接口地址 | 必填 |
| `QWEN_API_KEY` | Qwen API Key | 必填 |
| `QWEN_MODEL` | 模型名称 | `qwen3.7-plus` |
| `ENABLE_THINKING` | 是否输出思考过程 | `true` |

### `products.json`（待分析的商品图片）

采用结构化数组形式，包含 1 个本品与若干竞品：

```json
{
  "own_product": {
    "name": "本品名称",
    "url": "https://example.com/own-product.jpg"
  },
  "competitors": [
    { "name": "竞品1", "url": "https://example.com/competitor1.jpg" },
    { "name": "竞品2", "url": "https://example.com/competitor2.jpg" }
  ]
}
```

- `extract` / `ai-ecosystem`：对「本品 + 全部竞品」逐个分析。
- `compare`：将「本品 + 全部竞品」一并做横向对比（至少需 2 个产品）。

## 任务说明

- **extract**：逐个产品提取结构化卖点，输出严格 JSON，包含：
  - `product_name`：识别到的产品名称（品牌+型号）
  - `product_type`：产品类别
  - `selling_points[]`：每个卖点含 `level`（1/2/3 级）、`title`、`category`（卖点分类）、`sub_items`
  - 卖点分类枚举：便携、体验、颜值、性能、AI、扩展、续航、品质、输入体验、交互体验、屏幕、视听、安全（无匹配则为空）
- **compare**：跨产品归纳对比维度并给出结论。
- **ai-ecosystem**：提取产品中的 AI 能力与软件生态并总结，输出 JSON 含 `product_name`、`ai_features`、`software_ecosystem`、`summary`。
- **all**：按 `extract` → `compare` → `ai-ecosystem` 顺序依次执行全部任务，控制台会打印任务分节与产品进度标识。

## 输出结果

每个任务在控制台实时打印结果（含思考过程与 Token 用量）的同时，也会将结果写入本地文件。每个任务目录下分为两个子目录：

- `full/`：全量输出（思考过程 + 回答内容 + Token 用量），`.txt` 格式。
- `json/`：从回答中提取的纯 JSON 内容，`.json` 格式（提取失败时跳过）。

目录结构如下：

```
poc-result/
├── extract/
│   ├── full/
│   │   ├── <识别到的产品名称>.txt   # 文件名取自 JSON 的 product_name，识别失败则回退到配置名
│   │   └── ...
│   └── json/
│       ├── <识别到的产品名称>.json
│       └── ...
├── compare/
│   ├── full/
│   │   └── 横评结果.txt
│   └── json/
│       └── 横评结果.json
└── ai-ecosystem/
    ├── full/
    │   ├── <识别到的产品名称>.txt
    │   └── ...
    └── json/
        ├── <识别到的产品名称>.json
        └── ...
```

- 同名文件会被覆写更新。

## 依赖管理

- 新增依赖：`uv add <package>`
- 移除依赖：`uv remove <package>`
- 同步环境：`uv sync`
