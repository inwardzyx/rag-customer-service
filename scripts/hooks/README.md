# hooks —— 提交前的机械检查

## 装（每个新clone 都要做一次）

```bash
cp scripts/hooks/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
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

`scripts/check_readme_numbers.py` —— README 里写的数字 vs 脚本落盘的真实产物。

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