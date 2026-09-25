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

from kb.loader import load_documents, _parse_frontmatter, strip_footer   # noqa: E402

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
    """超长会被截到 max_chars —— 但【必须报出来】，不能静默丢字。

    ★ 2026-09-25 重写过。原版只有一句 `assert len(docs[0]["text"]) <= 50`：
        那是**当前实现的快照**，不是**不变量** —— 它把"丢内容"写成了正确行为，
        于是 loader 悄悄丢掉几百字也没人知道（库里真有一块丢了 95 字）。
        而且这条测试在「改成切成多块」的实现下也会红，等于把实现钉死了。

      现在断言的是真正该守的两件事：
        ① 不超长（保护向量库不被单块撑爆）；
        ② 截断时**必须留痕**（warning 日志里写明丢了多少字）。
      这两条对"切多块"和"截断"两种实现都成立，实现换了不会假红。
    """
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "t.md")
        with open(p, "w", encoding="utf-8") as f:
            f.write("---\ndoc: t.md\nversion: 2026-01-01\nsource: x\n---\n## c\n" + "很长的正文" * 100)
        docs, errors = load_documents(td, max_chars=50)
    assert errors == []
    assert len(docs) == 1
    assert len(docs[0]["text"]) <= 50


def test_truncation_is_not_silent(caplog):
    """★ 真正该守的不变量：截断【不许静默】。

    caplog 是 pytest 内置的夹具（fixture），用来捕获这次用例里产生的日志。
    用法：把它写成参数名 pytest 就会自动传进来；`caplog.text` 是日志全文。
    """
    import logging
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "t.md")
        with open(p, "w", encoding="utf-8") as f:
            f.write("---\ndoc: t.md\nversion: 2026-01-01\nsource: x\n---\n## 长条款\n" + "很长的正文" * 100)
        with caplog.at_level(logging.WARNING):
            load_documents(td, max_chars=50)
    assert "截断" in caplog.text, f"截断超长块却没报出来（又变回静默丢字了）：{caplog.text!r}"
    assert "丢" in caplog.text, f"日志里没写清丢了多少字：{caplog.text!r}"


def test_short_block_is_untouched():
    """不变量：没超长的块必须【一字不改】—— 清洗只能削页脚，不能动正文。"""
    body = "第三十条 本规定自2019年9月1日起施行，由学生工作处、教务处负责解释。"
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "t.md")
        with open(p, "w", encoding="utf-8") as f:
            f.write("---\ndoc: t.md\nversion: 2026-01-01\nsource: x\n---\n## 第三十条\n" + body)
        docs, _ = load_documents(td)
    assert docs[0]["text"] == body, f"正文被改了：{docs[0]['text']!r}"


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


# ==================================================================
# ③ 网页页脚剥离（2026-09-25 补：数据层最大的一处脏）
# ==================================================================
FOOTER_SAMPLE = (
    "第三十条 本规定自2019年9月1日起施行，由学生工作处、教务处负责解释。\n"
    "上一篇： 广东交通职业技术学院学生应征入伍管理办法（试行）\n"
    "下一篇： 广东交通职业技术学院关于规范学生集体外出活动安全管理的规定（试行）\n"
    "欢迎访问二级学院网站\n智慧建造与路桥学院 / 汽车与工程机械学院 / 信息学院\n"
    "联系我们\n党政办电话：020-87024621 招生咨询：020-37236028\n"
    "地址：广州市天河区天源路789号 / 广州市花都区工业大道11号\n"
    "网站管理：学生工作处 版权所有 @ 广东交通职业技术学院 粤ICP备15004264号"
)


def test_strip_footer_keeps_real_content():
    """剥离后只留真条文，且被剥掉的字数要说清楚。"""
    body, n = strip_footer(FOOTER_SAMPLE)
    assert body.startswith("第三十条"), f"真条文没了：{body!r}"
    assert "上一篇" not in body and "粤ICP" not in body, f"页脚没剥干净：{body!r}"
    assert n > 100, f"只剥了 {n} 字，页脚明显不止这么多"


def test_strip_footer_no_false_positive():
    """★ 防误伤：正常条款里没有页脚标记时，必须【一字不动】。

    这条很重要 —— 剥离是"削尾巴"，削错就是真条文被永久删掉，
    和之前 DUP_THRESHOLD 误删条款是同一类不可逆损失。
    """
    for t in ("学生请假一天以内由班主任审批，一周以内由二级学院审批。",
              "第二条 学生请假应填写《请假单》，说明请假事由和请假时间。",
              "对有违反法律法规、本规定以及学校纪律行为的学生，给予相应处分。"):
        body, n = strip_footer(t)
        assert n == 0, f"正常条款被误判成页脚：{t!r}"
        assert body == t, f"正常条款被改了：{body!r}"


def test_strip_footer_too_short_tail_is_kept():
    """尾巴太短不动它 —— 正文里偶然出现"联系我们"不该整段被削。"""
    t = "学生违纪后请联系我们处理。" + "补充说明。" * 2
    body, n = strip_footer(t)
    assert n == 0, f"短尾巴被误切了：{body!r}"


def test_strip_is_not_silent(caplog):
    """★ 剥离【不许静默】—— 这条是变异测试逼出来的，教训值得记：

    我先写了剥离功能 + 一堆断言，跑变异测试时把 `logger.warning` 改成 `logger.info`
    （也就是"悄悄清洗不告诉你"），结果**全套 48 条依然全绿**。
    那就等于我又造了一个 `content[:400]` —— 同样是静默改内容，只是方向相反。
    cc 提醒过这点（"洗完要记一笔账，否则就退化成静默清洗"），我没听进去，
    是变异测试替我抓出来的。

    所以：不只是"剥掉了"，还要"说得出来剥了多少、从哪条剥的"。
    """
    import logging
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "t.md")
        with open(p, "w", encoding="utf-8") as f:
            f.write("---\ndoc: t.md\nversion: 2026-01-01\nsource: x\n---\n"
                    "## 第三十条\n" + FOOTER_SAMPLE)
        with caplog.at_level(logging.WARNING):
            load_documents(td)
    assert "剥离网页页脚" in caplog.text, f"剥离了却没报出来（静默清洗）：{caplog.text!r}"


def test_real_corpus_has_no_footer_left():
    """★ 回归锁：真实语料里不许再残留任何页脚标记。

    这条盯的是【效果】不是【实现】—— 以后换了剥离规则、或抓了新页面，
    只要库里又混进页脚，这里就红。
    """
    docs, _ = load_documents(DOCS_DIR)
    dirty = [d["clause"] for d in docs
             if any(m in d["text"] for m in ("上一篇", "下一篇", "粤ICP备", "版权所有", "联系我们"))]
    assert dirty == [], f"库里还有块带页脚残留：{dirty}"


def test_real_corpus_no_block_is_truncated():
    """★ 剥完页脚后，全库不该再有块被 max_chars 截断（现在最长的也就 300 出头）。

    如果将来抓到一份真长条款把这里顶红了，说明该上「切多块」了，
    而不是继续让它静默丢字 —— 这条就是那个提醒。
    """
    docs, _ = load_documents(DOCS_DIR)
    over = [(d["clause"], len(d["text"])) for d in docs if len(d["text"]) >= 400]
    assert over == [], f"有块被截断了（丢字）：{over}"
