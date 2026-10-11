# ==================================================================
# Dockerfile —— 把服务装进一个真能跑起来的容器
# ==================================================================
# 为什么现在才写：顺序是有意的。CI 的慢 job 早就在【同为 Linux + Python 3.14】
#   的 runner 上把 `pip install -r requirements.txt` 跑通了，所以 Dockerfile
#   只剩「打包 + 补容器特有的缺口」这一件事 —— 而不是一边调依赖一边调打包。
#
# ★ Python 版本必须与 CI 完全一致（两个 job 都是 3.14）。版本一旦不一致，
#   "CI 绿了所以容器也会绿"这句话就不成立了，而那正是这个 job 存在的理由。
#
# ★ 为什么 slim 而不是 alpine：alpine 用 musl libc，onnxruntime / numpy 的
#   manylinux 轮子（glibc）在那边装不上，得现场编译 —— 那是另一个量级的事。
#
# ★ 刻意【不】把模型（92MB）烤进镜像，留给首次启动下载。理由：
#   · 镜像小、层可复用；
#   · 缓存在 CACHE_DIR（容器里是 /root/.cache/fastembed），挂卷就能跨容器复用，
#     不必每次重建都重下一遍：docker run -v rag-model:/root/.cache/fastembed ...
#   代价如实说：**冷启动要下 92MB**，这段时间 /health 返回 loading 是正常的。
#   想更快就用上面的挂卷，或干脆自己改成 build 时下载。
#
# ★ 刻意【不】在镜像里写 HF_ENDPOINT：service.py 用的是
#   os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")，
#   所以容器里（国内网络）默认走镜像；CI 里用 -e HF_ENDPOINT=https://huggingface.co
#   覆盖（runner 在海外）。一旦在镜像里写死，"零改代码就能换源"这个口子就没了。

FROM python:3.14-slim

# ★ libgomp1 不是可选项：onnxruntime（fastembed 的推理后端）依赖 OpenMP 运行库，
#   而 python:3.14-slim 里【没有】它，症状是 import 时
#     `libgomp.so.1: cannot open shared object file: No such file or directory`
#   这是"宿主机跑得通、容器里炸"最典型的一个坑，所以显式装上并写明原因 ——
#   免得以后有人看到这行 apt-get 觉得多余就删掉。
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# PYTHONUNBUFFERED：不加的话 python 会把日志攒在缓冲区里，
#   `docker logs` 长时间空白，出问题只能靠猜（CI 里排查全靠它）。
# PYTHONDONTWRITEBYTECODE：容器里不需要 .pyc，少一层无意义的可写层。
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# ★ 依赖层单独放在代码之前：requirements.txt 不变时，改一行代码不会触发重装依赖
#   （onnxruntime + fastembed 这一层是分钟级的，值得为它排一次层序）。
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
 && python -m pip install --no-cache-dir -r requirements.txt

# 再拷代码。
# ★ 之所以把 tests/ 和 evalset/ 也一起拷进来：CI 的容器 job 会【在容器里】
#   跑 tests/test_guard.py（它写死了 263 留 / 23 拦），用同一份真相源验证镜像环境，
#   而不是在 workflow 里再抄一遍这两个数字。代价是镜像大一点，
#   换来的是"容器真的能跑"这件事有真凭据 —— 这是有意的取舍，不是忘了排除。
COPY . .

# ★ 容器里的 loopback 只有容器自己看得见，绑 127.0.0.1 的话 -p 也救不了。
#   所以镜像里默认绑 0.0.0.0（service.py 读 HOST，本机跑时默认仍是 127.0.0.1）。
ENV HOST=0.0.0.0 \
    PORT=8000
EXPOSE 8000

# ★ 用 exec 形式（JSON 数组）而不是 shell 形式：
#   这样 PID 1 就是 python，docker stop 发的 SIGTERM 能直接到 uvicorn，
#   优雅退出生效；shell 形式下 PID 1 是 sh，信号到不了 python，只能等超时被 kill。
#
# ★ 关于密钥：镜像里【没有】.env（见 .dockerignore），这是故意的 ——
#   密钥 COPY 进镜像后 docker history / 镜像导出都能看见。
#   要跑完整问答（rerank + 生成）就在运行时给：
#       docker run --env-file .env -p 8000:8000 rag-cs
#   不给 key 也能跑，但只有【不需要模型生成】的那两条路：
#   /health、/guard-report、以及 /chat 的 use_rerank=false 拒答路径。
#   给默认路径（use_rerank=true）打请求会直接 500 —— 这是【刻意的】：
#   缺 key 属于配置错误，就该炸出来让人看见，不许包装成"调用失败"再降级成 200
#   （理由见 service.py:228 那段注释）。
CMD ["python", "service.py"]
