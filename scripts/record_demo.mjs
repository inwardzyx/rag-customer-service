// scripts/record_demo.mjs —— 把 README 那段动图重新录一遍（CDP 驱动浏览器，逐帧截屏）
//
// 为什么要有这个脚本：README 顶部那段 `docs/images/demo.gif` 是**实录**，
// 而"实录"必须能被复现，否则它和示意图没有区别。
//
// 它做三件事：
//   1. 起一个 headless Chrome，开调试端口
//   2. 打开你的服务网页，仿真打字 → 点「提问」→ 等答案出现
//   3. 全程每 150ms 截一帧 PNG，落到输出目录（合成 GIF 见 make_gif.py）
//
// 前置条件（缺一样都跑不起来，脚本会明确报错）：
//   · 服务已经在跑：在仓库根目录执行 `python service.py`，且 `.env` 里配好了
//     DEEPSEEK_API_KEY —— 因为要录一次**真的作答**（拒答那一半不需要 key）
//   · 本机装了 Chrome 或 Edge（自动找；也可用 --chrome= 指定）
//   · Node 18+（本机是 D:\node.exe）
//
// 跑法：
//   D:\node.exe scripts\record_demo.mjs
//   D:\node.exe scripts\record_demo.mjs --out=D:\tmp\frames --url=http://127.0.0.1:8000
//
// 跑完会看到（最后两行是两幕各自拿到的答案，用来确认录的是真东西）：
//   预热截图后，帧数 = 1
//   第一幕答案： 根据资料，处分种类有：警告、严重警告、记过、留校察看、开除学籍。
//   第二幕答案： 知识库里没有能回答这个问题的资料，我不编。
//   帧数： <N>
//
// 然后合成 GIF：
//   D:\Python-project\.venv\Scripts\python.exe scripts\make_gif.py

import { spawn } from "node:child_process";
import { mkdirSync, writeFileSync, rmSync, readdirSync, existsSync } from "node:fs";
import { setTimeout as sleep } from "node:timers/promises";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve } from "node:path";

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const arg = (name, dflt) => {
  const hit = process.argv.find(a => a.startsWith(`--${name}=`));
  return hit ? hit.slice(name.length + 3) : dflt;
};

// ---------- 找浏览器：先看 --chrome= / CHROME_PATH，再按常见路径找 Chrome、Edge ----------
function findBrowser() {
  const explicit = arg("chrome", process.env.CHROME_PATH || "");
  if (explicit) return explicit;
  const cands = [
    "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
    "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
    "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
  ];
  const hit = cands.find(existsSync);
  if (!hit) {
    console.error("找不到 Chrome / Edge。用 --chrome=<路径> 或设 CHROME_PATH 指定。");
    process.exit(2);
  }
  return hit;
}

const CHROME = findBrowser();
const PORT = Number(arg("port", "9223"));               // CDP 调试端口（不是服务端口）
const URL_APP = arg("url", "http://127.0.0.1:8000");
const OUT = resolve(arg("out", join(REPO, ".demo_frames")));
const PROFILE = resolve(arg("profile", join(REPO, ".demo_chrome_profile")));
const W = Number(arg("width", "960")), H = Number(arg("height", "780"));
const FRAME_MS = Number(arg("frame-ms", "150"));        // 截帧间隔 ≈ 6.7 fps

// 两幕：必须一次命中（带出处）+ 一次拒答。
// 只录命中看不出这个项目的差异化；只录拒答则看不出它真能答。
const Q1 = arg("q1", "学校对学生的处分有哪几种？");
const Q2 = arg("q2", "图书馆几点关门？");

console.log(`浏览器   ${CHROME}`);
console.log(`服务     ${URL_APP}`);
console.log(`帧输出   ${OUT}`);

rmSync(OUT, { recursive: true, force: true });
mkdirSync(OUT, { recursive: true });

const chrome = spawn(CHROME, [
  "--headless=new", `--remote-debugging-port=${PORT}`, `--user-data-dir=${PROFILE}`,
  `--window-size=${W},${H}`, "--no-first-run", "--no-default-browser-check",
  "--disable-extensions", "--hide-scrollbars", "about:blank",
], { stdio: "ignore" });

// ★ --user-data-dir 必须显式给：不给的话 Chrome 会去用默认 profile，
//   发现本机已有 Chrome 在跑就把命令行转交过去、自己退出 —— 调试端口永远不会起来。
//   （这个坑我踩过：症状是连接超时，看起来像"CDP 不好使"。）
async function getWs() {
  for (let i = 0; i < 40; i++) {
    try {
      const ts = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
      const p = ts.find(t => t.type === "page");
      if (p?.webSocketDebuggerUrl) return p.webSocketDebuggerUrl;
    } catch { /* 端口还没起来，继续等 */ }
    await sleep(250);
  }
  console.error(`Chrome 调试端口 ${PORT} 没起来。检查：1) 端口是否被占 2) 是否有残留的 headless Chrome`);
  process.exit(3);
}

const ws = new WebSocket(await getWs());
await new Promise(res => ws.addEventListener("open", res, { once: true }));

let seq = 0;
const pending = new Map();
ws.addEventListener("message", ev => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
});
function send(method, params = {}) {
  const id = ++seq;
  return new Promise(res => { pending.set(id, res); ws.send(JSON.stringify({ id, method, params })); });
}
const evaluate = expression =>
  send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });

let frames = 0;
async function shot() {
  try {
    const r = await send("Page.captureScreenshot", { format: "png" });
    if (r.error) { console.log("  [shot] CDP error:", JSON.stringify(r.error)); return; }
    if (!r.result?.data) { console.log("  [shot] 没有 data，返回键:", Object.keys(r)); return; }
    frames++;
    writeFileSync(join(OUT, `f${String(frames).padStart(4, "0")}.png`),
                  Buffer.from(r.result.data, "base64"));
  } catch (e) {
    console.log("  [shot] 异常:", e.message);
  }
}

await send("Page.enable");
await send("Runtime.enable");
await send("Emulation.setDeviceMetricsOverride",
           { width: W, height: H, deviceScaleFactor: 1, mobile: false });

// ★ 先导航、再开截帧循环 —— 反过来写会在 about:blank 上空转，一张有效帧都拿不到。
await send("Page.navigate", { url: URL_APP });
await sleep(2500);
await shot();
console.log("预热截图后，帧数 =", readdirSync(OUT).length);

let capturing = true;
(async () => { while (capturing) { await shot(); await sleep(FRAME_MS); } })();
await sleep(300);

// 仿真打字：逐字写进 textarea（页面读的是 .value，不必派发 input 事件）
const type = async (text, perChar = 90) => {
  for (let i = 1; i <= text.length; i++) {
    await evaluate(`document.getElementById('q').value=${JSON.stringify(text.slice(0, i))}`);
    await sleep(perChar);
  }
};
const click = () => evaluate(`document.querySelector('button').click()`);
const ansText = async () =>
  (await evaluate(`document.getElementById('ans').textContent`)).result?.result?.value ?? "";
const clear = () => evaluate(`document.getElementById('q').value='';` +
                             `document.getElementById('ans').style.display='none';` +
                             `document.getElementById('ans').textContent='';` +
                             `document.getElementById('src').replaceChildren();`);

async function waitAnswer(timeoutMs = 60000) {
  const t0 = Date.now();
  while (Date.now() - t0 < timeoutMs) {
    const t = await ansText();
    if (t && t !== "思考中…") return t;
    await sleep(200);
  }
  throw new Error("等答案超时 —— 服务是不是没起？或者 .env 里没配 DEEPSEEK_API_KEY？");
}

await sleep(900);            // 停在空页面上，让标题先入镜
await type(Q1);
await sleep(500);
await click();
const a1 = await waitAnswer();
await sleep(2500);           // 停住，让人读答案 + 引用条款 + 精排分

await clear();
await sleep(700);
await type(Q2);
await sleep(400);
await click();
const a2 = await waitAnswer();
await sleep(2600);           // 停住，让人读"我不编"

capturing = false;
await sleep(FRAME_MS + 100);
ws.close();
chrome.kill();

console.log("第一幕答案：", a1);
console.log("第二幕答案：", a2);
console.log("帧数：", readdirSync(OUT).length);
console.log(`\n下一步：\n  D:\\Python-project\\.venv\\Scripts\\python.exe scripts\\make_gif.py`);
