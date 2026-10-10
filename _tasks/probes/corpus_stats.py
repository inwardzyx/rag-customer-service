# -*- coding: utf-8 -*-
"""探针：把两处今晚反复被引用的数字，做成一条可重跑的命令。

为什么要有这个文件：两份评审都判定"必须实跑才能定案"，但**没有一个人跑**
（见 `_out_collab_protocol.md` §2 末尾）。把命令固化下来，是让下一个人
不必"记得怎么算"。

用法：<py> _tasks/probes/corpus_stats.py
"""
from __future__ import annotations

import io
import json
import logging
import sys
from contextlib import redirect_stderr
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

log = io.StringIO()
logging.basicConfig(level=logging.WARNING, stream=log, format="%(message)s")

from kb.loader import load_documents                     # noqa: E402
from evalset.heldout import make_pack                    # noqa: E402

with redirect_stderr(log):
    load_documents(ROOT / "docs")

for line in log.getvalue().splitlines():                 # 加载时清洗了 N 块：截断丢字 …
    if "清洗" in line:
        for part in line.split("：", 1)[-1].split("、"):
            print(part.strip())

qs = json.loads((ROOT / "evalset" / "questions.json").read_text(encoding="utf-8"))
gold = {(q["gold"][0], q["gold"][1]) for q in qs if q.get("gold")}
same = "==" if len(gold) == len(make_pack.ALREADY_COVERED) else "!="
print(f"ALREADY_COVERED {len(make_pack.ALREADY_COVERED)} {same} gold 去重 {len(gold)}")
