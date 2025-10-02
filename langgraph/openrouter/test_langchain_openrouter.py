"""使用 LangChain 调用 OpenRouter 上的 x-ai/grok-4-fast:free 模型示例。

要点：
1. OpenRouter 提供 OpenAI 兼容接口；在 LangChain 中可通过 ChatOpenAI 指定 base_url。
2. 模型名称使用官方标识："x-ai/grok-4-fast:free"。
3. 需要环境变量：OPENROUTER_API_KEY（推荐），兼容 OPENAI_API_KEY。
4. 这里演示：
   - 基本单轮调用 invoke()
   - 流式输出 stream()
   - 带上下文 / 系统提示
5. 若你的 langchain-openai 版本字段不同，代码已做回退处理。

依赖：requirements 中需包含 langchain-openai / langchain-core / langchain-community。

运行：
  export OPENROUTER_API_KEY="sk-or-v1-xxxx"  # 在 shell 中设置
  python test_langchain_openrouter.py

文档： https://openrouter.ai/docs
"""

from __future__ import annotations

import os
import sys
import time
from typing import Iterable


def ensure_api_key() -> str:
	"""获取 OpenRouter API Key。

	优先级：
	1. OPENROUTER_API_KEY
	2. OPENAI_API_KEY (兼容)
	未找到则提示并退出。
	"""
	key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
	if not key:
		print("[ERROR] 请先在环境变量中设置 OPENROUTER_API_KEY (或 OPENAI_API_KEY)。")
		print("例如: export OPENROUTER_API_KEY=sk-or-v1-xxxxx")
		sys.exit(1)
	return key


def build_client(model: str = "x-ai/grok-4-fast:free", **kwargs):
	"""构建 LangChain ChatOpenAI 客户端。

	兼容 (不同版本字段)：base_url / openai_api_base。
	"""
	from langchain_openai import ChatOpenAI  # 延迟导入方便报错定位

	api_key = ensure_api_key()

	common_init = dict(
		model=model,
		api_key=api_key,  # 新版 langchain-openai 可用
		temperature=kwargs.pop("temperature", 0.7),
		timeout=kwargs.pop("timeout", 120),
		max_retries=kwargs.pop("max_retries", 2),
		# 支持 response_format / seed 等可继续添加
	)

	# OpenRouter OpenAI 兼容地址
	base_url = "https://openrouter.ai/api/v1"

	# 优先尝试新版参数 base_url；失败则回退 openai_api_base
	try:
		llm = ChatOpenAI(base_url=base_url, **common_init)  # type: ignore[arg-type]
	except TypeError:
		# 旧版本语法兼容
		llm = ChatOpenAI(openai_api_base=base_url, openai_api_key=api_key, **{k: v for k, v in common_init.items() if k not in {"api_key"}})  # type: ignore[arg-type]

	return llm


def simple_invoke(prompt: str) -> str:
	"""普通同步调用。"""
	from langchain_core.messages import SystemMessage, HumanMessage

	llm = build_client()
	messages = [
		SystemMessage(content="你是一名简洁、准确且有结构化表达能力的 AI 助手。回答时如有条目尽量使用列表。"),
		HumanMessage(content=prompt),
	]
	resp = llm.invoke(messages)
	# resp.content 可能是 str 或 list[dict]（函数调用等），这里只处理 str
	return getattr(resp, "content", str(resp))


def stream_invoke(prompt: str) -> str:
	"""流式调用，边接收边打印，最后返回完整文本。"""
	from langchain_core.messages import SystemMessage, HumanMessage

	llm = build_client()
	messages = [
		SystemMessage(content="你是一名条理清晰的技术助理。"),
		HumanMessage(content=prompt),
	]
	assembled = []
	print("\n[Streaming]\n", flush=True)
	start = time.time()
	for chunk in _safe_stream(llm.stream(messages)):
		delta = getattr(chunk, "content", "")
		if delta:
			assembled.append(delta)
			print(delta, end="", flush=True)
	print()  # 换行
	print(f"\n[Done] 耗时: {time.time() - start:.2f}s\n")
	return "".join(assembled)


def _safe_stream(stream_iter: Iterable):
	"""兼容不同版本 chunk 类型。"""
	for item in stream_iter:
		yield item


def main():  # pragma: no cover - 简单脚本入口
	import argparse

	parser = argparse.ArgumentParser(description="LangChain + OpenRouter grok-4-fast:free 示例")
	parser.add_argument("prompt", nargs="?", default="解释一下 Transformer 的核心思想，并列出 3 个关键组件。")
	parser.add_argument("--mode", choices=["once", "stream"], default="once")
	args = parser.parse_args()

	if args.mode == "once":
		answer = simple_invoke(args.prompt)
		print("\n[Answer]\n" + answer)
	else:
		final_text = stream_invoke(args.prompt)
		print("\n[Final Accumulated]\n" + final_text)


if __name__ == "__main__":
	main()



