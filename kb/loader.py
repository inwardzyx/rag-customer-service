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

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

REQUIRED_META = ("doc", "version", "source")

# ==================================================================
# 网页「页脚模板」剥离
# ==================================================================
# ★ 为什么要有这段：知识库来自网页抓取，网页的页脚（上一篇/下一篇、学院名单、
#   联系我们、地址、备案号）会和最后一条条款粘在一起被切成同一块。
#   实测两块中招：
#       《学生纪律处分管理规定》第五章-第三十条  原始 495 字 → 真条文仅 38 字（10%）
#       《学生请销假制度》总则-第七条            原始 137 字 → 真条文仅 31 字（23%）
#   这类块长度够、不重复、无手机号身份证、元信息齐全 → **现有 5 道把关一道都拦不住**。
#
# ★ 为什么标记分两组（这个区分是实测逼出来的，不是拍脑袋）：
#   强特征（备案号/版权）固然准，但它们在页脚的**最后几行**；
#   而 `content[:400]` 截断发生在页脚中途 —— 实测那一块被截后的 400 字里
#   **根本没有「粤ICP备」/「版权所有」/「网站管理」**。
#   所以规则必须也认靠前的弱特征（`上一篇`/`下一篇`/`联系我们`），否则
#   在「上游已经出错」的库上会二次失效。只写备案号是不够的。
CHROME_STRONG = ("粤ICP备", "粤公网安备", "版权所有", "网站管理", "网站维护", "扫码关注")
CHROME_WEAK = ("上一篇", "下一篇", "欢迎访问", "联系我们", "党政办电话", "招生咨询")

# 只有当页脚尾巴达到一定长度才切，避免正文里偶然出现"联系我们"就被削掉。
# 30 字是拍的，但很保守：真页脚动辄上百字，而误伤代价是真条文被削。
MIN_FOOTER_CHARS = 30


def strip_footer(text: str) -> tuple[str, int]:
    """剥掉文本尾部粘着的网页页脚，返回 (清洗后文本, 被剥掉的字数)。

    没命中任何标记 → 原样返回，剥掉 0 字（不会误伤正常条款）。

    ★ 返回的是个「二元组」，`body, n = strip_footer(t)` 这样接。
      只写 `body = strip_footer(t)` 的话 body 会是整个元组，后面当字符串用会炸。
    """
    pos = None
    for marker in CHROME_STRONG + CHROME_WEAK:
        p = text.find(marker)          # find 找不到返回 -1，找到返回下标
        if p >= 0 and (pos is None or p < pos):
            pos = p                    # 取【最靠前】的那个标记当切点

    # 没命中，或剩下的尾巴太短（不值得切，也可能是误伤）→ 不动
    if pos is None or len(text) - pos < MIN_FOOTER_CHARS:
        return text, 0

    return text[:pos].strip(), len(text) - pos


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

            # ★ 先剥页脚，再判长度 —— 顺序不能反。
            #   页脚会「垫高」块的长度，先判长度的话这类块一道都拦不住。
            content, stripped = strip_footer(content)
            if stripped:
                # 不静默清洗：剥了多少、哪一条，必须看得见（和下面截断是同一个道理）
                logger.warning("「%s｜%s」剥离网页页脚 %d 字（剩 %d 字正文）",
                               meta["doc"], clause.strip(), stripped, len(content))

            # ★ 超长不再静默截断：丢了多少字要报出来。
            #   为什么现在【不】改成"切成多块"：剥完页脚后全库超长块是 0 个，
            #   为不存在的需求写切分器是过度设计。真需要时再补，见 README「下一步」。
            #   （另：就算这里不截，bge-small-zh 也有 512 token 上限，
            #    会在 embedding 里把尾巴悄悄吃掉 —— 所以截断这道保护不能整个撤掉。）
            if len(content) > max_chars:
                logger.warning("「%s｜%s」长 %d 字，超过 max_chars=%d，已截断（丢 %d 字）",
                               meta["doc"], clause.strip(), len(content),
                               max_chars, len(content) - max_chars)
                content = content[:max_chars]

            docs.append({
                "doc": meta["doc"],
                "clause": clause.strip(),
                "version": meta["version"],
                "source": meta["source"],
                "text": content,
            })
    return docs, errors
