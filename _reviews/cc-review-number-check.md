三个文件都已读完。以下是我的结论。

---

## 一、要回答的问题（原句复述）

> `scripts/check_readme_numbers.py` 和 `scripts/mutate_check_numbers.py` 这两个新脚本，
> 它们的自检逻辑有没有"看着在检查、实际检查不到"的洞？

以及卡里加粗的那条：

> 变异测试报"5/5 全检出"，**这个"5/5"本身可信吗**？
> 变异是不是真的注入进去了？（脚本里我写了检查原文是否存在的逻辑，但那段本身也可能有洞）

---

## 二、结论（先回答卡里加粗那条）

**"5/5" 不构成为证据——打印它的那一行数错了东西。**
`mutate_check_numbers.py:139` 打印的是 `len(MUTATIONS)`（清单长度，恒等于 5），不是"实际注入并检出的条数"。计算路径：把 5 条 old 串全改成 README 里不存在的文本 → `not_applied=5`、`undetected=[]` → 行 129-132 只打一行 `⚠ 5 个变异没注入成功`，行 133 的 `if undetected:` 不成立 → 直接落到行 139 打印 `✓ 全部 5 个变异都被检出` → **`return 0`**。
⇒ 想让这条证据成立，必须同时确认输出里**没有 ⚠ 行**；今天那 5 个是否真注入，我无法核实（核实要跑突变脚本，它写 README，被卡的只读约束禁止）。触发条件极廉价：README 那句话哪怕只加一个空格，"vs拒答最高"这种按字面替换（行 40、50）就会全部 `old not in original` 而静默跳过。

**顺带一条覆盖面的结论**：5 条变异全在 README 侧（行 37-63 的 old 串全部只存在于 README）——"5/5"只证明「README→产物」这个方向会响，「产物→README」方向 0 覆盖。而下面的洞 #3、#4 恰好都在没被覆盖的那一半。

---

## 三、漏洞清单（文件:行号 / 为什么是洞 / 依据）

1. **`mutate_check_numbers.py:129-141`** —— 注入失败不参与判定，见上。修法：`not_applied` 非空即 `return 1`，或把行 139 改成 `{len(MUTATIONS)-len(not_applied)}/{len(MUTATIONS)}`。

2. **`mutate_check_numbers.py:96-100` + `102`** —— 注入判据只查 `old in original`（子串存在），不查"这处就是 checker 实际读的那一处"。配合 `replace(..., 1)`（只改第一处）和 checker 侧全部用 `re.search`（只取第一处，`check_readme_numbers.py:93/184/191/209/224/300-301`）：**同一句话在 README 出现两次时，改第二处既不会被 checker 发现、也不会被变异测试发现**（变异也只替换第一处）。修法：注入后断言"checker 读到的值 == 变异目标值"，或对每条变异先记录变异前的读值。

3. **`check_readme_numbers.py:205-217` + `151`** —— README 的 **14/15 没有任何一侧在读它**。行 151 注释写"★ 故意留空：见专项检查"，但 `_check_recall`（行 206）只读 `evalset/report-vector.md`，行 213 拿产物和**硬编码字面量 "14/15"** 比，全程不碰 README；行 256 的 `if a.readme_pattern is None: continue` 又把它从通用 README 循环里跳过。算给我看：把 README 任一处 `14/15` 改成 `12/15` → 三条专项检查（拒答/recall/gold）都读不到它，通用循环跳过它 → errors 为空 → **exit 0**。变异清单也没有这一条。这是最直白的"假通过"，也正好回应卡里"特别欢迎的失效场景"。

4. **`check_readme_numbers.py:220-230`（行 225-226）+ `243-250` + `266-269`** —— 产物侧抽不到时静默降级。条件链：`_check_scale_ranks` 正则不中 → `return []`（行 225-226，对比 `_check_refuse_rate:186` 是报 error）；通用循环里 `expect=None` 的断言只进 `warns`（行 247 `(errors if a.expect else warns)`，gold 排名正好 expect=None）；README 循环里 `actual is None` 直接 `continue`（行 268-269），而 `readme_checked += 1` 在它**之前**（行 266），所以行 287 的"探头失灵"守门看不见这次降级（它数的是"README 文本找到了几条"，不是"真比了几条"）。⇒ 实验记录里"中位排名"那行格式一变（全角箭头、加空格），checker 变成"只确认 README 有这句话"+ 一条 `提示（不算错）` → **exit 0，pre-commit 报绿**。修法：`_check_scale_ranks` 抽不到应 error；守门计数器应在**真发生比较之后**才 +1。

5. **`check_readme_numbers.py:161-170` + `156`** —— "同源比对"。丢字数那条的 `src` 是 `experiments/clean_corpus.py`（**脚本源码**），行 165 和行 167 是**同一条正则** ⇒ 这是「README ↔ 脚本里同一句字面量」互核，不是 README ↔ 产物。推断依据：行 165 要求字面数字，若该脚本是运行时算出再打印（f-string），正则今天根本抽不到、卡里也不会说它报绿；既然报绿，说明 5241 是以字面量形式写在脚本文本里 —— 那么脚本重算与 README 是两份抄写，**一起错就全绿**。行 156 的 src 是 `.md` 实验记录（人写的），同类。（这两个文件我未被允许读，属读码推断。）

6. **`check_readme_numbers.py:93-94`** —— `fetch_readme` 多捕获组时**只取 group(1)**，而行 134 的 readme_pattern 有 2 个组 ⇒ 0.6321 在"阈值余量"这条里被丢弃，只靠行 143 的另一条断言捞回。今天变异 #3 能检出（靠行 143），但两侧规则不一致：产物侧是"拼全部组"（行 112），README 侧是"只取第一组"。将来删掉行 137-144 那条断言，0.6321 会静默失守。

7. **`check_readme_numbers.py:190`** —— `if rd.exists():` 没有 else：README.md 不存在时，拒答准确率的 README 侧检查**无声跳过**（其他断言会因 `fetch_readme` 返回 None 报错，所以总体仍红，但这一条自身无声）。

8. **`check_readme_numbers.py:59-60 / 71-72 / 29-30`（低危，文档与实现不符）** —— `readme_value` 字段从未参与任何比较（全文无引用）；`note` 字段同样从未被 `check()` 读，模块 docstring 行 29-30 承诺的"允许口径不同、用 note 声明"在实现里不存在。将来有人写了 note 以为获得豁免、或以为 readme_value 被核，都会误判。行 72 自己写着"★非空不代表豁免—— 见 check() 的实现"，而 check() 的实现里没有这一段。

9. **`mutate_check_numbers.py:104 / 119`** —— `write_text` 未指定 `newline=""`：读出（universal newlines，`\r\n`→`\n`）再写回（文本模式把 `\n` 翻成 os.linesep）⇒ 若 README 原本是 LF，**跑一次变异测试就把整份文件换成 CRLF**，产生全文件级 diff。（推断；没跑，也不敢跑——它会写仓库文件。）

10. **`mutate_check_numbers.py:107-117`** —— "检出"判据只有"退出码非零"，没核对错误行是否真的提到了被改的值。若 checker 因异常崩溃（traceback、空输出），也会计"✓ 检出"，只打印 `(没打印错误行)`。基线绿（行 84-88）挡住了"一启动就崩"，挡不住"只对变异输入崩"。

11. **`scripts/hooks/pre-commit:20-21`** —— `[ -x "$PY" ] || PY="$(command -v python3 || command -v python)"`：venv 路径失效时**静默回落到裸 python**，与它自己行 18-19 的注释（"裸敲 python 会命中 conda base 3.8，结果不能代表项目环境"）矛盾。当前两脚本只用标准库、3.8 也能跑，风险低；但"静默回落"本身是假的安心，应改成找不到就报错退出。另：本次会话起始的 `git status` 显示 `scripts/hooks/` 整个目录 **untracked** ⇒ 这份 hook 在别的克隆/提交里还不存在（我只看了给定的 git status，没跑 git 命令）。

---

## 四、假装通过的三条具体路径（总结）

- **A（最严重）**：改 README 里的 `14/15` → 永远绿（checker 不读它、变异不测它）。
- **B**：README 那句话改了措辞/加空格 → 5 条变异全部静默跳过 → 仍打印"✓ 全部 5 个变异都被检出"、exit 0。
- **C**：实验记录的"中位排名"行格式一变 → 该检查退化成 warn，其余照绿，exit 0。

---

## 五、只能读码推断、未实证的部分（按卡要求标出）

- 本次**没跑任何脚本**（突变脚本会写 README，被只读约束禁止；checker 可跑但结论不依赖它），也没读 README / 产物文件（不在允许清单里）。以上全部是 code-path 追踪，每条都带了行号，可逐条复核。
- 洞 #5 里"clean_corpus.py 把 5241 写字面量"、洞 #9 的 LF/CRLF、`mutate:69-74` 用 `encoding="utf-8"` 抓子进程输出而子进程 stdout 编码随本机 locale（若子进程是 cp936，`✓/✗` 这类字符可能在子进程侧抛 `UnicodeEncodeError`）——均为推断；作者已见到 `✓ 通过` 退出 0，说明本机路径没问题。
- README 里 14/15、0.5968 当前**是否真的出现多处**，我无法核实（不能读 README）；洞 #2 表述为条件句，成立与否取决于这一点。
