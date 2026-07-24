import re
import json
import argparse
from pathlib import Path
from typing import NamedTuple

from openai import OpenAI
from dotenv import load_dotenv

from config import Config, Product
from prompts import (
    EXTRACT_SELLING_POINTS_PROMPT,
    COMPARE_SELLING_POINTS_PROMPT,
    EXTRACT_AI_ECOSYSTEM_PROMPT,
)


# 结果输出根目录
RESULT_ROOT = Path(__file__).parent / "poc-result"


class ChatResult(NamedTuple):
    """一次模型调用的结果。

    - full：控制台的完整输出（思考过程 + 回答内容 + Token 用量），用于写入文件。
    - answer：仅回答内容（非思考部分），用于提取 JSON、产品名等二次处理。
    """

    full: str
    answer: str


class ProductSellingPointsAnalyzer:
    """商品详情页卖点分析器。

    封装了对多模态大模型的调用，支持多种分析任务：
    - 提取单个产品卖点
    - 多个产品核心卖点横评
    - 提取产品中的 AI 程序与软件生态相关内容并总结
    """

    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )

    # ------------------------------------------------------------------
    # 内部通用方法
    # ------------------------------------------------------------------
    def _build_image_messages(self, img_urls, prompt: str):
        """根据图片 URL（单张或多张）和提示词构造 messages。"""
        if isinstance(img_urls, str):
            img_urls = [img_urls]

        content = [
            {"type": "image_url", "image_url": {"url": url}} for url in img_urls
        ]
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def _chat_stream(self, messages) -> ChatResult:
        """流式调用模型，实时打印思考过程与回复。

        返回 ChatResult：full 为控制台完整输出（思考 + 回答 + 用量），
        answer 为纯回答内容（非思考部分），便于二次处理。
        """
        completion = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            extra_body={"enable_thinking": self.config.enable_thinking},
            stream=True,
        )

        full_output = ""      # 控制台完整输出（思考过程 + 回答 + 用量）
        answer_content = ""   # 仅回答内容（非思考部分）
        is_answering = False

        def emit(text: str, end: str = ""):
            """同时打印到控制台并累积到完整输出。"""
            nonlocal full_output
            print(text, end=end, flush=True)
            full_output += text + end

        if self.config.enable_thinking:
            emit("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n", end="\n")

        for chunk in completion:
            # chunk.choices 为空时打印 usage
            if not chunk.choices:
                emit("\n" + "=" * 20 + "Token Usage" + "=" * 20 + "\n", end="\n")
                emit(json.dumps(chunk.usage.model_dump(), ensure_ascii=False, indent=2))
                continue

            delta = chunk.choices[0].delta

            # 打印思考过程（回复阶段 reasoning_content 可能是空字符串，用真值判断跳过）
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                emit(reasoning)

            # 打印回复内容（与思考过程独立判断）
            if delta.content:
                if not is_answering:
                    emit("\n" + "=" * 20 + "回答内容" + "=" * 20 + "\n", end="\n")
                    is_answering = True
                emit(delta.content)
                answer_content += delta.content

        return ChatResult(full=full_output, answer=answer_content)

    # ------------------------------------------------------------------
    # 具体任务
    # ------------------------------------------------------------------
    def extract_selling_points(self, img_url: str) -> ChatResult:
        """提取单张商品详情页图片中的结构化卖点信息。"""
        messages = self._build_image_messages(img_url, EXTRACT_SELLING_POINTS_PROMPT)
        return self._chat_stream(messages)

    def compare_selling_points(self, img_urls) -> ChatResult:
        """多个产品核心卖点横评。"""
        messages = self._build_image_messages(img_urls, COMPARE_SELLING_POINTS_PROMPT)
        return self._chat_stream(messages)

    def extract_ai_ecosystem(self, img_url: str) -> ChatResult:
        """提取某个产品中涉及 AI 程序和软件生态相关的内容并总结。"""
        messages = self._build_image_messages(img_url, EXTRACT_AI_ECOSYSTEM_PROMPT)
        return self._chat_stream(messages)


# ---------------------------------------------------------------------------
# 输出与日志辅助
# ---------------------------------------------------------------------------
def _sanitize_filename(name: str) -> str:
    """将产品名转为安全的文件名。"""
    safe = re.sub(r'[\\/:*?"<>|]', "_", name).strip()
    return safe or "result"


def _extract_product_name(answer: str) -> str | None:
    """从回答内容中解析 JSON 里的 product_name 字段（找不到返回 None）。"""
    match = re.search(r'"product_name"\s*:\s*"([^"]+)"', answer)
    return match.group(1).strip() if match else None


def _extract_json(answer: str) -> str | None:
    """从纯回答内容中提取 JSON 并格式化（找不到或解析失败返回 None）。

    回答可能被 ```json 代码块包裹，去壳后取第一个 { 到最后一个 } 的片段，
    经 json 解析校验后返回格式化文本。
    """
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", answer, re.DOTALL)
    raw = fenced.group(1) if fenced else answer

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.dumps(json.loads(raw[start : end + 1]), ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        return None


def _print_banner(text: str):
    """在控制台打印醒目的产品/任务标识，便于区分不同产品的输出。"""
    print("\n" + "#" * 60)
    print(f"#  {text}")
    print("#" * 60 + "\n", flush=True)


def _print_section(text: str):
    """在控制台打印更醒目的任务分节标识（用于 all 命令区分各任务阶段）。"""
    print("\n" + "=" * 70)
    print(f"||  {text}")
    print("=" * 70 + "\n", flush=True)


def _save_result(task: str, product_name: str, result: ChatResult) -> Path:
    """保存结果，返回全量文件路径。

    - full 目录：poc-result/<任务>/full/<产品名>.txt，保存全量输出。
    - json 目录：poc-result/<任务>/json/<产品名>.json，仅保存纯 JSON 内容。

    若同名文件已存在，则直接覆写更新。
    """
    safe_name = _sanitize_filename(product_name)

    # full：全量输出
    full_dir = RESULT_ROOT / task / "full"
    full_dir.mkdir(parents=True, exist_ok=True)
    full_path = full_dir / f"{safe_name}.txt"
    full_path.write_text(result.full, encoding="utf-8")

    # json：从纯回答内容提取 JSON（提取失败则跳过）
    json_text = _extract_json(result.answer)
    if json_text is not None:
        json_dir = RESULT_ROOT / task / "json"
        json_dir.mkdir(parents=True, exist_ok=True)
        json_path = json_dir / f"{safe_name}.json"
        json_path.write_text(json_text, encoding="utf-8")

    return full_path


# ---------------------------------------------------------------------------
# 各子命令的处理函数
# ---------------------------------------------------------------------------
def _resolve_single_products(config: Config) -> list[Product]:
    """extract / ai-ecosystem 使用：确定待逐个分析的产品列表。

    取 products.json 中的本品 + 全部竞品。
    """
    products = config.all_products
    if not products:
        raise SystemExit("缺少产品：请在 products.json 配置 own_product / competitors")
    return products


def _cmd_extract(analyzer: ProductSellingPointsAnalyzer, config: Config):
    products = _resolve_single_products(config)
    for idx, product in enumerate(products, start=1):
        _print_banner(f"【提取卖点】({idx}/{len(products)}) {product.name}")
        result = analyzer.extract_selling_points(product.url)
        # 文件名使用模型识别出的产品名称，识别失败时回退到配置中的名称
        file_name = _extract_product_name(result.answer) or product.name
        path = _save_result("extract", file_name, result)
        print(f"\n\n[已保存] {file_name} -> {path}\n", flush=True)


def _cmd_compare(analyzer: ProductSellingPointsAnalyzer, config: Config):
    products = config.all_products
    if len(products) < 2:
        raise SystemExit(
            "横评至少需要 2 张图片：请在 products.json 配置 own_product 与 competitors"
        )
    names = "、".join(p.name for p in products)
    _print_banner(f"【核心卖点横评】{names}")
    result = analyzer.compare_selling_points([p.url for p in products])
    path = _save_result("compare", "横评结果", result)
    print(f"\n\n[已保存] 横评结果 -> {path}\n", flush=True)


def _cmd_ai_ecosystem(analyzer: ProductSellingPointsAnalyzer, config: Config):
    products = _resolve_single_products(config)
    for idx, product in enumerate(products, start=1):
        _print_banner(f"【AI与软件生态】({idx}/{len(products)}) {product.name}")
        result = analyzer.extract_ai_ecosystem(product.url)
        # 文件名使用模型识别出的产品名称，识别失败时回退到配置中的名称
        file_name = _extract_product_name(result.answer) or product.name
        path = _save_result("ai-ecosystem", file_name, result)
        print(f"\n\n[已保存] {file_name} -> {path}\n", flush=True)


def _cmd_all(analyzer: ProductSellingPointsAnalyzer, config: Config):
    """依次执行全部三个任务：extract、compare、ai-ecosystem。"""
    steps = [
        ("extract", "提取卖点", _cmd_extract),
        ("compare", "核心卖点横评", _cmd_compare),
        ("ai-ecosystem", "AI与软件生态", _cmd_ai_ecosystem),
    ]
    total = len(steps)
    for idx, (task, label, handler) in enumerate(steps, start=1):
        _print_section(f"任务 {idx}/{total}：{label}（{task}）")
        handler(analyzer, config)
    _print_section("全部任务已完成")


def build_parser() -> argparse.ArgumentParser:
    """构建命令行解析器。仅保留任务项，其余参数均来自配置文件或默认值。"""
    parser = argparse.ArgumentParser(description="商品详情页卖点分析工具")
    subparsers = parser.add_subparsers(dest="task", required=True)

    subparsers.add_parser("extract", help="提取各产品的结构化卖点")
    subparsers.add_parser("compare", help="多个产品核心卖点横评")
    subparsers.add_parser("ai-ecosystem", help="提取 AI 程序与软件生态相关内容并总结")
    subparsers.add_parser("all", help="依次执行以上全部三个任务")

    return parser


# 任务名 -> 处理函数
TASK_HANDLERS = {
    "extract": _cmd_extract,
    "compare": _cmd_compare,
    "ai-ecosystem": _cmd_ai_ecosystem,
    "all": _cmd_all,
}


def main():
    load_dotenv()

    parser = build_parser()
    args = parser.parse_args()

    # 模型、思考开关、图片 URL 等全部来自配置文件或默认值
    config = Config.from_env()

    analyzer = ProductSellingPointsAnalyzer(config)
    TASK_HANDLERS[args.task](analyzer, config)


if __name__ == "__main__":
    main()

