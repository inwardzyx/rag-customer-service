# -*- coding: utf-8 -*-
"""kb/loader 的单元测试 —— 不加载 embedding 模型，纯测「文件 → 块列表」的解析。

和 tests/test_guard.py 的区别：
    test_guard 走的是「真实语料 + 真实模型」的重路径（慢，十几秒起）；
    本文件只测加载器本身，一条都不碰模型，秒级跑完，CI 里也便宜。

跑法（仓库根目录）：
    python -m pytest tests/test_loader.py -v
"""

import os
import sys
import tempfile

# 把仓库根目录加进模块搜索路径，才能 import kb.loader
# （本文件【不】import service —— 那会连带加载 embedding 模型，就不便宜了）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kb.loader import load_documents, _parse_frontmatter   # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(REPO_ROOT, "docs")


# ==================================================================
# ① 元信息解析：三个字段齐全才收，缺一个 / 格式错就拒
# ==================================================================
def test_parse_frontmatter_ok():
    text = "---\ndoc: a.md\nversion: 2026-01-01\nsource: x\n---\n## c\n正文"
    assert _parse_frontmatter(text) == {
        "doc": "a.md", "version": "2026-01-01", "source": "x"}


def test_parse_frontmatter_missing_field_returns_none():
    # 缺 source → 不合格
    text = "---\ndoc: a.md\nversion: 2026-01-01\n---\n## c\n正文"
    assert _parse_frontmatter(text) is None


def test_parse_frontmatter_no_delimiter_returns_none():
    # 根本没用 --- 包裹 → 不合格（这种文件不该进知识库）
    assert _parse_frontmatter("## c\n正文无元信息") is None


# ==================================================================
# ② 真实 docs/ 目录：数量 + schema 必须和入库把关的预期对得上
# ==================================================================
def test_load_documents_counts():
    docs, errors = load_documents(DOCS_DIR)
    # 真实语料（学校公开制度）：纪律处分 61 块 + 请销假 7 块 + inbox 抓取残留 15 块 = 83 条原始块。
    # 注意：这里返回的是「解析出的全部原始块」，还没过入库把关；
    # 过完关后才会变成 68 留 15 拦（那条断言在 test_guard.py::test_kept_count_matches_real_corpus）。
    assert len(docs) == 83, f"解析出 {len(docs)} 条，应为 83 条"
    assert errors == [], f"不应有解析失败的文件：{errors}"


def test_load_documents_schema():
    docs, _ = load_documents(DOCS_DIR)
    for d in docs:
        # 这 5 个字段正是 _guard / guard_report / chat 一路会用到的 schema
        assert set(d) == {"doc", "clause", "version", "source", "text"}


def test_load_documents_no_embedding_needed():
    # 关键 proof：这条路径不 import / 不加载任何模型，loader 可独立测试
    docs, errors = load_documents(DOCS_DIR)
    assert errors == []
    assert all(isinstance(d["text"], str) and d["text"] for d in docs)


# ==================================================================
# ③ 行为细节：截断 + 坏文件真的进 errors，而不是悄悄丢
# ==================================================================
def test_max_chars_truncation():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "t.md")
        with open(p, "w", encoding="utf-8") as f:
            f.write("---\ndoc: t.md\nversion: 2026-01-01\nsource: x\n---\n## c\n" + "很长的正文" * 100)
        docs, errors = load_documents(td, max_chars=50)
    assert errors == []
    assert len(docs) == 1
    assert len(docs[0]["text"]) <= 50


def test_bad_frontmatter_goes_to_errors():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "bad.md")
        with open(p, "w", encoding="utf-8") as f:
            f.write("没有元信息，直接 ## 条款\n## c\n正文")
        docs, errors = load_documents(td)
    assert docs == []
    assert errors == [p]            # 坏文件进了 errors，而不是被吞掉


def test_kb_before_inbox_order():
    # 证明排序真的把 kb 排在了 inbox 前面 —— 这是「脏副本不会挤掉干净条款」的前提。
    docs, _ = load_documents(DOCS_DIR)
    # 用【文档名】而不是 source 文本来区分，source 描述文字改了也不会让这条测试假红
    order = [d["doc"] for d in docs]
    INBOX_DOC = "网页抓取残留.md"
    kb_positions = [i for i, s in enumerate(order) if s != INBOX_DOC]
    inbox_positions = [i for i, s in enumerate(order) if s == INBOX_DOC]
    assert kb_positions and inbox_positions, (
        f"kb / inbox 两侧都得有块才谈得上顺序。实际文档顺序：{order[:5]}...")
    assert max(kb_positions) < min(inbox_positions)
