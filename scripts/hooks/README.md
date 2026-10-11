# hooks —— 提交前 / 推送前的机械检查

## 装（每个新clone 都要做一次）

```bash
cp scripts/hooks/pre-commit .git/hooks/pre-commit
cp scripts/hooks/pre-push   .git/hooks/pre-push
chmod +x .git/hooks/pre-commit .git/hooks/pre-push
```

⚠️ `.git/hooks/` **不进仓库**，所以别人 clone 下来默认没有这个钩子。
这是 git 的设计，不是 bug。→ 要让别人也有，就得写进 README 的setup 步骤。

## 为什么钩子里不派 cc / DSH

每起一个 `cc` 进程≈**18K token 冷启动**（实测，2026-08-08 那次 13 分钟内换6 种
调用方式，付了 7 次冷启动的钱）。55 个提交就是 100 万 token 上下，
而绝大多数提交（改 README 措辞）根本不需要审稿人。

⇒ **派 agent 留给「改既有代码之前」这个真触发点**（见 `CLAUDE.md` 硬规矩 2）。
钩子只做一件事：**几毫秒能查完、且不留假象**的检查。

## 跑什么

| 钩子 | 调用 | 查什么 |
|---|---|---|
| `pre-commit` | `scripts/check_readme_numbers.py` | README 里写的数字 vs 脚本落盘的真实产物 |
| `pre-push` | `scripts/check_push_paths.py` | **即将推上去的文件**里有没有"本不该进仓库"的 |

## ★ pre-push：为什么会有它（2026-10-11，一起真实事故）

提交 `7252a69`（2026-09-24 **10:53**）把 `CLAUDE.md` + 4 个 `cc_*.md`
**推上了公开仓库**。次日 **16:14** 主干被 `git filter-branch` 重写、
16:37 force-push —— 材料在远端 main 上活了 **约 29.7 小时**。

重写之后这些文件不在当前树里了（`GET /contents/CLAUDE.md` → 404），**但是**：

- 那个提交**仍按 SHA 可达**（不被任何分支指向，却照样能取）；
- `https://github.com/<owner>/<repo>/commit/7252a69.patch`
  一个普通 URL 就返回 **200 / 20788 字节**，5 个文件的**完整正文**都在里面；
- 链接入口就在**公开的 Actions 页面**上（run #1）。

⇒ 教训一句话：**push 是单向门。**
  · `.gitignore` 只挡"以后"，对"历史上进过库的东西"一个字都不说；
  · **删除本地分支/改本地历史对远端毫无帮助** —— 远端对象的存在性
    只由「是否 push 过」决定，此后一切本地操作与之无关。

（在同一件事上，DSH 先误判"从未外泄"、又误以为删本地备份分支有用，
 两次都是"拿局部为真的检查去支撑全局结论"。cc 的复核把时间线从
 `.git/logs/` 的 push 日志里重建出来，才算把这条讲清楚。）

## P1 / P2：两道防线，不能互相替代

- **P1（推前，本钩子）**：拦"即将推上去的可疑路径"。防呆。
  ★ 已用**真实 push** 验过：把 `7252a69` 推到一个本地裸仓库 → 钩子列出 5 个文件并拒绝。
    也就是说**当年若有它，那次事故不会发生**。
- **P2（推后审计，未实现）**：枚举**公开面上印出来的 SHA**（Actions runs 的 `head_sha`）
  → 判断它是否能从 main 到达（不能 ⇒ 孤儿）→ 取该 commit 的 `files[]` 与同一张模式表求交。
  本案可查（run #1 就印着那个 SHA）。
- **做不到的那一半**：**没有 API 能枚举远端的不可达对象**。所以"从未在任何公开面
  出现过的孤儿 SHA"查不到 —— 对人也一样查不到。别为此写一个假检查器。

## 已知边界

- **本 clone 是浅克隆**（`.git/shallow` 定格在初始提交，2026-09-22），
  所以本地任何历史审计只覆盖那之后。
- `--not --remotes` 在"往另一个 remote 推新分支"时可能少算（多 remote 场景），
  详见 `check_push_paths.py` 里的注释。

## ⚠️ 这套东西自己也被审过

`check_readme_numbers.py` 的**第一版跑出来是"✓ 通过"**，
但变异测试（`scripts/mutate_check_numbers.py`）把 README 的 `0.5968`改成 `0.9999` 后，
它**照样报通过** —— 因为第一版只查产物侧、没查 README 侧。

⇒ 这和loader 清洗账塌成 0 是**同一个模式**：探头只照了一面，
却把"没发现问题"读成"没问题"。

现在两段都查，且加了**探头自检**（一条 README 侧断言都没执行到 ⇒ 报"探头失灵"）。

```bash
# 想确认这套检查自己还有效（改了脚本之后必跑）
D:/Python-project/.venv/Scripts/python.exe scripts/mutate_check_numbers.py
```

当前状态：**5 个变异全部检出，恢复后仍报绿。**

## 绕过

`git commit --no-verify` 能跳过（git 原生行为）。

⚠️ 但规矩是：**用了 `--no-verify`，commit 信息里就不能写"检查通过了"。**
绕过检查本身不是漏洞，**假装检查过了**才是。