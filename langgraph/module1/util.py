"""LangGraph 图渲染工具 (精简版)。

仅提供一个简单的 `render_graph` 函数，并保留 `Util.render_graph` 兼容旧调用。
"""

from typing import Any
import os

def render_graph(g: Any, outfile: str = "simple_graph.png", overwrite: bool = False) -> str:
    """将已编译的 LangGraph 导出为 Mermaid PNG。

    参数:
        g: builder.compile() 结果对象，需支持 g.get_graph().draw_mermaid_png()
        outfile: 输出 PNG 路径
        overwrite: True 时即使文件存在也重写

    返回: 最终文件路径
    """
    if not overwrite and os.path.isfile(outfile):
        print(f"Skip existing: {outfile} (overwrite=True 以强制重生成)")
        return outfile

    data = g.get_graph().draw_mermaid_png()
    with open(outfile, "wb") as f:
        f.write(data)
    print(f"Saved graph PNG: {outfile}")
    return outfile

# 向后兼容: 仍可使用 Util.render_graph(graph)
class Util:  # noqa: D401  简单兼容壳
    render_graph = staticmethod(render_graph)

__all__ = ["render_graph", "Util"]
