# -*- coding: utf-8 -*-
"""scripts/make_gif.py —— 把 record_demo.mjs 截下的逐帧 PNG 合成 README 用的 GIF

三个必须做的处理（少一个，GIF 要么几 MB、要么难看）：
  1. 自动裁掉内容以外的空白 —— 页面 max-width 是 760 + margin 40，而视口是 960x780，
     不裁的话大片白边既占体积又分散注意力
  2. 连续相同的帧去重，并把它们的时间**累加**到保留的那一帧上
     —— "停住让人读"那几秒会产生十几张一模一样的图，直接拼进去纯属浪费
     （实测：64 帧 → 18 帧，体积降一个数量级）
  3. 调色板降到 128 色 —— 页面是白底黑字 + 几个彩色标签，128 色绰绰有余

跑法（先跑 record_demo.mjs 拿到帧）：
    D:\\Python-project\\.venv\\Scripts\\python.exe scripts\\make_gif.py
    D:\\Python-project\\.venv\\Scripts\\python.exe scripts\\make_gif.py --src=D:\\tmp\\frames --dst=D:\\tmp\\x.gif

跑完会看到：
    读入 64 帧
    裁切区域 x=86..892  y=35..467  →  806 x 432
    去重后 18 帧（原 64），总时长 9.6 秒
    已写入 <仓库>\\docs\\images\\demo.gif
    大小 0.15 MB
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageChops

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser(description="逐帧 PNG → README 用 GIF")
ap.add_argument("--src", default=str(REPO / ".demo_frames"), help="帧目录（record_demo.mjs 的输出）")
ap.add_argument("--dst", default=str(REPO / "docs" / "images" / "demo.gif"), help="GIF 输出路径")
ap.add_argument("--frame-ms", type=int, default=150, help="截帧间隔，必须和 record_demo.mjs 的一致")
ap.add_argument("--colors", type=int, default=128, help="调色板颜色数")
ap.add_argument("--pad", type=int, default=14, help="裁切后四周留的白边（像素）")
A = ap.parse_args()

SRC = Path(A.src)
DST = Path(A.dst)

files = sorted(SRC.glob("*.png"))
if not files:
    raise SystemExit(f"{SRC} 里没有帧 —— 先跑 scripts/record_demo.mjs")
print(f"读入 {len(files)} 帧")

raw = [Image.open(f).convert("RGB") for f in files]


def nonwhite_bbox(im: Image.Image):
    """这一帧中"不是白底"的区域。用来求所有帧的并集 → 一次裁切，帧间不错位。"""
    return ImageChops.difference(im, Image.new("RGB", im.size, (255, 255, 255))).getbbox()


boxes = [b for b in (nonwhite_bbox(im) for im in raw) if b]
if not boxes:
    raise SystemExit("所有帧都是纯白的？检查一下截图是不是拍到了空页面")
x0 = max(0, min(b[0] for b in boxes) - A.pad)
y0 = max(0, min(b[1] for b in boxes) - A.pad)
x1 = min(raw[0].width, max(b[2] for b in boxes) + A.pad)
y1 = min(raw[0].height, max(b[3] for b in boxes) + A.pad)
print(f"裁切区域 x={x0}..{x1}  y={y0}..{y1}  →  {x1 - x0} x {y1 - y0}")

cropped = [im.crop((x0, y0, x1, y1)) for im in raw]

# ---------- 去重：和上一帧一模一样 → 不新增帧，只把时长加给上一帧 ----------
frames: list[Image.Image] = []
durations: list[int] = []
for im in cropped:
    if frames and ImageChops.difference(im, frames[-1]).getbbox() is None:
        durations[-1] += A.frame_ms
    else:
        frames.append(im)
        durations.append(A.frame_ms)
print(f"去重后 {len(frames)} 帧（原 {len(cropped)}），总时长 {sum(durations) / 1000:.1f} 秒")

durations[-1] += 1500          # 最后一帧多停一会儿，让人把"我不编"看清

pal = [f.quantize(colors=A.colors, method=Image.Quantize.MEDIANCUT) for f in frames]
DST.parent.mkdir(parents=True, exist_ok=True)
pal[0].save(DST, save_all=True, append_images=pal[1:], duration=durations,
            loop=0, optimize=True, disposal=2)

size = DST.stat().st_size
print(f"已写入 {DST}")
print(f"大小 {size / 1024 / 1024:.2f} MB")
if size > 6 * 1024 * 1024:
    print("★ 超过 6MB，GitHub 上加载会慢 —— 考虑 --colors 降到 64 或把页面调窄")
