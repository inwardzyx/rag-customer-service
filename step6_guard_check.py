# -*- coding: utf-8 -*-
"""
Step 6 的【把关逻辑自检脚本】

为什么不直接起服务验证？
    起服务要占端口、要等模型加载、看完还得手动 Ctrl+C。
    而这个脚本只做一件事：把 step6 里的入库把关跑一遍，把"留下谁 / 拦了谁"打印出来。

用法：
    set LANGSMITH_TRACING=false
    D:/Python-project/.venv/Scripts/python.exe step6_guard_check.py

看什么：
    1. kept 里必须同时有【退款-已发货】和【退款-未发货】两条
       （这是之前修掉的 bug：按文档名分组会互相挤掉）
    2. rejected 里要有：旧版本那条、含手机号身份证那条、话术库重复那条
"""

import os

# 这两行必须放在 import 项目模块之前：
# os.environ 是"进程的环境变量字典"，这里改的是当前这个 Python 进程自己的那份。
os.environ["LANGSMITH_TRACING"] = "false"          # 自检不调大模型，关掉 trace 免得联网卡住
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import step6_fastapi_service as svc                # 导入不会启动服务（因为有 __main__ 保护）

print("正在加载 embedding 模型（第一次慢，之后走缓存）……")
svc.rag.startup()

print("\n" + "=" * 66)
print(f"入库 {len(svc.rag.chunks)} 条，拦下 {len(svc.rag.rejected)} 条，"
      f"耗时 {svc.rag.boot_time:.1f} 秒")
print("=" * 66)

print("\n【留下的（知识库内容）】")
for c in svc.rag.chunks:
    print(f"  ✓ {c['doc']}｜{c['clause']}｜v{c['version']}")
    print(f"      {c['text']}")

print("\n【被拦下的】")
for c, r in svc.rag.rejected:
    print(f"  ✗ {c['doc']}｜{c.get('clause','')}｜v{c['version']}  →  {r}")
    print(f"      {c['text'][:40]}")

# ---- 断言：把"应该发生的事"写成检查，以后改代码改坏了这里会立刻报错 ----
clauses = {(c["doc"], c["clause"]) for c in svc.rag.chunks}
print("\n" + "=" * 66)
checks = [
    ("已发货条款还在", ("退款政策.md", "退款-已发货") in clauses),
    ("未发货条款还在（bug 修复点）", ("退款政策.md", "退款-未发货") in clauses),
    ("发货时效还在", ("发货时效.md", "现货发货") in clauses),
    ("旧版本被拦", any("旧版本" in r for _, r in svc.rag.rejected)),
    ("隐私被拦", any("隐私" in r for _, r in svc.rag.rejected)),
    ("重复话术被拦", any("重复" in r for _, r in svc.rag.rejected)),
]
for name, ok in checks:
    print(f"  {'通过' if ok else '失败'}  {name}")
print("=" * 66)
print("\n全部通过 ✓" if all(ok for _, ok in checks) else "\n有检查没过 ✗")
