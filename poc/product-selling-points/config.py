"""集中管理运行配置。

- 密钥与模型等运行参数：从环境变量（.env）读取。
- 待处理的商品图片 URL（本品 + 竞品）：从 products.json 读取，
  采用结构化数组形式，方便查看与修改。

业务代码通过 `Config.from_env()` 获取一个不可变的配置对象。
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field


DEFAULT_MODEL = "qwen3.7-plus"
PRODUCTS_FILE = Path(__file__).parent / "products.json"


@dataclass(frozen=True)
class Product:
    """单个商品（本品或竞品）。"""

    name: str
    url: str


def _load_products(path: Path) -> tuple["Product | None", list["Product"]]:
    """从 products.json 加载本品与竞品列表。文件不存在时返回空配置。"""
    if not path.exists():
        return None, []

    data = json.loads(path.read_text(encoding="utf-8"))

    own_raw = data.get("own_product")
    own = Product(name=own_raw["name"], url=own_raw["url"]) if own_raw else None

    competitors = [
        Product(name=item["name"], url=item["url"])
        for item in data.get("competitors", [])
    ]
    return own, competitors


@dataclass(frozen=True)
class Config:
    """运行期配置。"""

    api_key: str | None
    base_url: str | None
    model: str = DEFAULT_MODEL
    enable_thinking: bool = True

    # 本品（单产品任务默认使用）
    own_product: "Product | None" = None
    # 竞品列表
    competitors: list["Product"] = field(default_factory=list)

    @property
    def own_url(self) -> str | None:
        """本品图片 URL。"""
        return self.own_product.url if self.own_product else None

    @property
    def all_products(self) -> list["Product"]:
        """本品 + 竞品的全部产品（含名称，用于逐个分析）。"""
        products: list[Product] = []
        if self.own_product:
            products.append(self.own_product)
        products.extend(self.competitors)
        return products

    @property
    def all_urls(self) -> list[str]:
        """本品 + 竞品的全部图片 URL（用于横评）。"""
        urls: list[str] = []
        if self.own_product:
            urls.append(self.own_product.url)
        urls.extend(c.url for c in self.competitors)
        return urls

    @classmethod
    def from_env(cls, products_file: Path = PRODUCTS_FILE) -> "Config":
        """从环境变量与 products.json 构建配置对象。"""
        own, competitors = _load_products(products_file)
        return cls(
            api_key=os.getenv("QWEN_API_KEY"),
            base_url=os.getenv("QWEN_ENDPOINT"),
            model=os.getenv("QWEN_MODEL", DEFAULT_MODEL),
            enable_thinking=os.getenv("ENABLE_THINKING", "true").lower() != "false",
            own_product=own,
            competitors=competitors,
        )
