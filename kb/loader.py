# -*- coding: utf-8 -*-
"""kb/loader.py —— 知识库文件加载器

把 docs/ 树下的 .md 读成 RAG 用的「块（chunk）」列表。

一个 .md 文件 = 一篇文档（顶部 --- 之间写 doc / version / source 元信息），
文件里用 `## 条款名` 切分出的每一段 = 一条可被检索的「块」。

★ 为什么不再写死 RAW_DOCS？
    写死 = 面试官问「你的知识库多大、文档怎么进来的」只能答「我手敲了 10 条」。
    从文件读 = 加文档只要往 docs/ 扔一个 .md，入库把关照常拦脏数据，
    这才像一个「能接真实文档流」的系统，而不是一个写死的 demo。

★ 加载顺序：docs/kb/ 先于 docs/inbox/（详见 load_documents 里的排序 key）。
    原因：近似重复关是「先来的留、后到的像就拦」。干净库先入，
    待审核的脏副本（旧版 / 话术库改写版）作为重复或旧版被拦下；
    反过来若脏数据先入，它会把干净的条款挤掉（那个被删的恰恰是你真正要用的）。
"""

from __future__ import annotations

import re
from pathlib import Path

REQUIRED_META = ("doc", "version", "source")


def _parse_frontmatter(text: str) -> dict | None:
    """解析文件顶部 --- ... --- 里的元信息。格式不对 / 缺字段 → 返回 None。

    用字符串切分而不是真 YAML：知识库的元信息就三个固定字段，
    没必要引入 PyYAML 依赖，手写解析也更好测。
    """
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None                      # 文件根本没用 --- 包裹元信息
    fm = parts[1]
    meta: dict = {}
    for line in fm.splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" not in line:
            return None                  # 元信息里出现无法识别的行
        k, _, v = line.partition(":")
        k, v = k.strip(), v.strip()
        if not k:
            return None
        meta[k] = v
    if not all(k in meta for k in REQUIRED_META):
        return None                      # 三个必填字段少一个都不收
    return meta


def load_documents(root: str | Path, max_chars: int = 400) -> tuple[list[dict], list[str]]:
    """扫描 root 下所有 .md，返回 (块列表, 解析失败的文件路径列表)。

    root 通常为项目里的 docs/ 目录，里面分 kb/（已审核）和 inbox/（待审核）。

    sorted 的稳定排序 + kb 优先：
        ① sorted 保证「加文件 / 改名」不会让「留谁拦谁」随机抖动，测试才能
           稳定断言「留 7 拦 3」；
        ② key 里 (目录名 != "kb") 让 kb 排在前、inbox 排在后 —— 见模块顶部说明。
    """
    root = Path(root)
    docs: list[dict] = []
    errors: list[str] = []

    entries = sorted(
        root.rglob("*.md"),
        key=lambda p: (p.parent.name != "kb", p),     # kb/ 先，inbox/ 后
    )
    for path in entries:
        try:
            # ★ 必须指定 utf-8：Windows 默认按 cp936（GBK）读，中文会直接乱码抛错
            text = path.read_text(encoding="utf-8")
        except Exception:
            errors.append(str(path))
            continue

        meta = _parse_frontmatter(text)
        if meta is None:
            errors.append(str(path))                   # 元信息不合格 → 当作坏文件跳过
            continue

        body = text.split("---", 2)[-1]                # 取关闭的 --- 之后的正文
        for sec in re.split(r"(?m)^##\s+", body)[1:]:  # 按「行首 ## 」切成一条条条款
            clause, _, content = sec.partition("\n")
            content = content.strip()
            if not content:
                continue
            docs.append({
                "doc": meta["doc"],
                "clause": clause.strip(),
                "version": meta["version"],
                "source": meta["source"],
                "text": content[:max_chars],           # 超长截断，避免单块撑爆向量库
            })
    return docs, errors
