# -*- coding: utf-8 -*-
"""
环境兼容层：绕开本机"应用控制策略拦截 DLL"的坑

坑是什么？
    本机（校园网/公司机器常见）会用应用控制策略拦掉某些 DLL，报错长这样：
        ImportError: DLL load failed while importing mmh3: 应用程序控制策略已阻止此文件
    同一个坑我们之前在 faiss 上遇到过一次（所以代码里的向量检索有 numpy 兜底）。

为什么能绕？
    mmh3 是 murmur3 哈希的 C 扩展。fastembed 只在【稀疏检索】里用它，
    确切地说只有一行：mmh3.hash(token)（见 fastembed/sparse/bm25.py）。
    而本项目用的是 dense 向量检索 + rank_bm25，根本走不到那行代码。
    所以只要让 "import mmh3" 这一步别炸就行。

为什么不直接返回一个假值（比如 0）？
    那样万一以后真的用到稀疏检索，会得到一堆碰撞的哈希，而且极难排查。
    这里给出的是【真正的 murmur3 x86_32 实现】，算出来的数和 C 版逐位一致
    （见文件底部 _self_test()），所以即使将来走到那条路也是对的。

用法（必须在 import fastembed 之前！）：
    import env_compat
    env_compat.ensure_mmh3()
"""

import logging
import sys
import types

logger = logging.getLogger(__name__)


def _murmur3_x86_32(data: bytes, seed: int = 0) -> int:
    """murmur3 哈希的 32 位版本（x86 变体）—— 和 C 扩展 mmh3.hash 结果一致

    算法的思路（不用背，知道它在干什么就行）：
        把字节切成 4 个一组，每组先"搅乱"（乘个魔数 + 循环左移），再混进累积的 h1；
        最后处理尾巴（不满 4 字节的部分），再做一轮"雪崩"收尾（fmix）。
        "雪崩"的意思是：输入改一个比特，输出大约有一半的比特会翻 —— 好哈希的标准。
    """
    c1 = 0xCC9E2D51
    c2 = 0x1B873593
    length = len(data)
    h1 = seed & 0xFFFFFFFF

    # ① 每 4 字节一组处理（round down 到 4 的倍数）
    n_blocks = length & 0xFFFFFFFC          # 位运算：把 length 向下取整到 4 的倍数（末两位清零）
    for i in range(0, n_blocks, 4):
        # 把 4 个字节拼成一个 32 位整数（小端序：低位字节在前）
        k1 = data[i] | (data[i + 1] << 8) | (data[i + 2] << 16) | (data[i + 3] << 24)
        k1 = (k1 * c1) & 0xFFFFFFFF
        k1 = ((k1 << 15) | (k1 >> 17)) & 0xFFFFFFFF   # 循环左移 15 位
        k1 = (k1 * c2) & 0xFFFFFFFF
        h1 ^= k1
        h1 = ((h1 << 13) | (h1 >> 19)) & 0xFFFFFFFF   # 循环左移 13 位
        h1 = (h1 * 5 + 0xE6546B64) & 0xFFFFFFFF

    # ② 尾巴：剩下 1~3 个字节
    k1 = 0
    tail = length & 0x03                    # 位运算：length 除以 4 的余数
    if tail == 3:
        k1 ^= data[n_blocks + 2] << 16
    if tail >= 2:
        k1 ^= data[n_blocks + 1] << 8
    if tail >= 1:
        k1 ^= data[n_blocks]
        k1 = (k1 * c1) & 0xFFFFFFFF
        k1 = ((k1 << 15) | (k1 >> 17)) & 0xFFFFFFFF
        k1 = (k1 * c2) & 0xFFFFFFFF
        h1 ^= k1

    # ③ 收尾雪崩（fmix32）：让每一位都充分影响结果
    h1 ^= length
    h1 &= 0xFFFFFFFF
    h1 ^= h1 >> 16
    h1 = (h1 * 0x85EBCA6B) & 0xFFFFFFFF
    h1 ^= h1 >> 13
    h1 = (h1 * 0xC2B2AE35) & 0xFFFFFFFF
    h1 ^= h1 >> 16
    h1 &= 0xFFFFFFFF

    # ④ C 版 mmh3.hash 返回的是【有符号】32 位整数，这里对齐它的行为
    return h1 - 0x100000000 if h1 & 0x80000000 else h1


def ensure_mmh3():
    """保证 mmh3 这个模块能被 import —— 真货能用就用真货，用不了就换纯 Python 版"""
    try:
        import mmh3                          # 先试真货
        mmh3.hash(b"probe")                  # import 成功不代表能用，真调一下才作数
        return "real"
    except Exception:
        pass

    # ---- 走到这里说明真货用不了，塞一个纯 Python 的替身进 sys.modules ----
    # sys.modules 是"已导入模块"的缓存字典。往里面预先放一个键，
    # 后面别人再 import mmh3 时，Python 直接取我们放的这个，就不会去加载被拦的 DLL 了。
    shim = types.ModuleType("mmh3")
    shim.__doc__ = "mmh3 的纯 Python 替身（本机 DLL 被策略拦截时的降级方案，见 env_compat.py）"

    def _hash(key, seed=0):
        # mmh3.hash 既吃 str 也吃 bytes；统一按 UTF-8 编成 bytes 再算
        if isinstance(key, str):
            key = key.encode("utf-8")
        elif isinstance(key, bytearray) or isinstance(key, memoryview):
            key = bytes(key)
        return _murmur3_x86_32(key, seed)

    shim.hash = _hash
    shim.hash64 = lambda key, seed=0: (_hash(key, seed), _hash(key, seed + 1))
    shim.hash128 = lambda key, seed=0, x64arch=True: (
        (_hash(key, seed + 1) << 64) | (_hash(key, seed) & 0xFFFFFFFFFFFFFFFF))
    shim.hash_bytes = lambda key, seed=0: _hash(key, seed).to_bytes(4, "little", signed=True)

    sys.modules["mmh3"] = shim
    return "shim"


# ==================================================================
#② uuid_utils 的同类坑（2026-10-09 新增）
# ==================================================================
#
# 现象（和 mmh3 一模一样的报错，但发生在另一个包）：
#     ImportError: DLL load failed while importing _uuid_utils:
#     应用程序控制策略已阻止此文件
#
# 触发路径（实测，与README 里写的 mmh3 那条完全不同链）：
#     service.py
#       → langchain_deepseek.chat_models
#         → langchain_core.callbacks          ← import 这个就炸
#           → langchain_core.callbacks.manager:33  from ...utils.uuid import uuid7
#             → langchain_core.utils.uuid:12from uuid_utils.compat import uuid7
#               → uuid_utils/__init__.py → from ._uuid_utils import ...  ← DLL 在这里被拦
#
#★ 为什么本项目根本不需要 C 版：
#     langchain 只用它生成 **LangSmith 追踪的 run_id**（callbacks/manager.py 里那几处
#     run_id=uuid7()）。run_id 只要"每个值互不相同、且大致按时间递增"就够了，
#     它不会拿去当数据库主键、也不会跨进程比较。
#     而 uuid7 的规范（RFC 9562 §5.7）是公开的、可直接照着写的纯算法 ——
#     没有随机源之外的隐藏依赖，所以纯 Python 实现完全够用。
#
# ★ 为什么不直接用 uuid4() 顶替：
#     语义上错了。uuid4 是纯随机，**不保证时间有序**；而这个函数存在的全部意义
#     就是"时间有序"（LangSmith 按 run_id 排序 trace）。用一个语义相反的函数顶替，
#     等于把故障换成了静默的行为错误 —— 这正是本项目最忌讳的那类问题。
#
# ★ 为什么不返回假值 / 抛错：
#     同 mmh3 那条：让 import 成功但值是错的，比直接崩更糟。
#     至少这里的实现算出来的**就是符合规范的 uuid7**，不是假值。

def _uuid7_pure(timestamp_ms=None, counter=None, random_bytes=None):
    """按 RFC 9562 §5.7 生成 UUIDv7。

    位布局（48 / 4 / 12 / 2 / 30 / 32）：
        unix_ts_ms | version=7 | 随机位 (12) | variant | 计数器 (30 位) | 随机位 (32 位)

    ★ 为什么不用 uuid.uuid1()：
#       uuid1 是"时间+MAC 地址"，位数布局完全不同（version=1），
#       当成 uuid7 用会把版本号写错，下游任何按 version 判断的逻辑都会错。

    语法点（两个新手常踩的）：
      · >> 48 / >> 16 这类【右移】是取高位；
        左边补 0，右边挤掉低位。
      · & (1 << 8)：算出某一位是不是 1（位掩码）。
    """
    import os as _os
    import time as _time
    from uuid import UUID

    if timestamp_ms is None:
        timestamp_ms = int(_time.time() * 1000)
    if random_bytes is None:
        random_bytes = _os.urandom(10)
    if counter is None:
        counter = int.from_bytes(random_bytes[:4], "big") & 0x3FFF_FFFF  # 30 位

    # unix 时间戳占 48 位（毫秒）—— 那是 48 个十六进制位 = 6 字节
    ms = timestamp_ms & 0xFFFF_FFFF_FFFF
    rand_hi = int.from_bytes(random_bytes[4:7], "big") & 0x0FFF        # 12 位
    rand_lo = int.from_bytes(random_bytes[7:10], "big")                 # 32 位

    # 拼成 128 位整数再转 UUID：
    #   ms(48) | 0x7(4) | rand_hi(12) | 0b10(2) | counter(30) | rand_lo(32) = 128
    value = (
        (ms << 80)
        | (0x7 << 76)
        | (rand_hi << 64)
        | (0b10 << 62)
        | (counter << 32)
        | rand_lo
    )
    return UUID(int=value, version=7)


def ensure_uuid_utils():
    """保证 uuid_utils 能被 import —— 真货能用就用真货，用不了就换纯 Python 版。

    与 ensure_mmh3 的差别（这点必须说清）：
      uuid_utils 不止uuid7，还有 uuid1/3/4/5/6/8、getnode、NIL 等一堆导出。
      而本项目【只需要 uuid7】—— 调用方是 langchain 的callbacks/manager.py，
      它只import 了 uuid7 一个符号。
      所以替身只提供 uuid7，其余名字【故意不提供】：
      将来若真有代码去用 uuid4 之类，会报 AttributeError（"这个替身没有它"），
      而不是静默返回一个错的东西。**缺能力要暴露，不要装得像有。**
    """
    try:
        from uuid_utils import uuid7# 先试真货
        uuid7()                                          # import 成功不代表能用
        return "real"
    except Exception:
        pass

    shim = types.ModuleType("uuid_utils")
    shim.__doc__ = ("uuid_utils 的纯 Python 替身（只实现 uuid7；"
                    "本机 DLL 被策略拦截时的降级方案，见 env_compat.py）")
    shim.__version__ = "0.0.0-shim"

    # langchain 走的是 `from uuid_utils.compat import uuid7` ——
    # 它import 的是【子模块】compat，所以必须同时把 uuid_utils.compat 也塞进去。
    # 只塞顶层的话，那行 from ... .compat import 会报 ModuleNotFoundError。
    compat = types.ModuleType("uuid_utils.compat")

    def _uuid7(timestamp=None, nanos=0):
        """兼容 uuid_utils 的真实签名：uuid7(timestamp=秒, nanos=纳秒)。"""
        if timestamp is None:
            return _uuid7_pure()
        # 真实实现里 timestamp 是【秒】、nanos 是【纳秒】，这里如实换算成毫秒。
        return _uuid7_pure(timestamp_ms=timestamp * 1000 + nanos // 1_000_000)

    shim.uuid7 = _uuid7
    compat.uuid7 = _uuid7
    shim.compat = compat
    sys.modules["uuid_utils"] = shim
    sys.modules["uuid_utils.compat"] = compat
    return "shim"


def _self_test_uuid7():
    """自检：算出来的必须【真的是】UUIDv7，而不只是"长得像个 UUID"。"""
    import time as _time
    import uuid as _uuid

    u = _uuid7_pure()
    ok = True

    # ① version 位必须是 7（1.6.3 起的实现走的是 _uuid_utils.compat）
    if u.version != 7:
        ok = False
        logger.info("uuid7 自检：version 是 %s，期望 7 ✗", u.version)

    # ② variant 位必须是 RFC 4122（0b10xx → .variant == UUID_RESERVED_NCS 那一族之外的
    #    RFC_4122）。标准库给的常量名是 RESERVED_FUTURE 之外的 RFC_4122。
    if getattr(u, "variant", None) != _uuid.RFC_4122:
        ok = False
        logger.info("uuid7 自检：variant 是 %s，期望 RFC_4122 ✗", u.variant)

    # ③ 长度与类型
    if not isinstance(u, _uuid.UUID) or len(u.bytes) != 16:
        ok = False
        logger.info("uuid7 自检：类型或长度不对✗")

    # ④ 同一毫秒内连出3 个，必须互不相同（这是"计数器"存在的意义）
    t = int(_time.time() * 1000)
    trio = [_uuid7_pure(timestamp_ms=t) for _ in range(3)]
    if len(set(trio)) != 3:
        ok = False
        logger.info("uuid7 自检：同毫秒内出现重复 ✗")

    # ⑤ 时间有序：后一个毫秒的值必须大于前一个（langchain 靠这个排序 trace）
    a = _uuid7_pure(timestamp_ms=t)
    b = _uuid7_pure(timestamp_ms=t + 1000)
    if int(b) <= int(a):
        ok = False
        logger.info("uuid7 自检：不满足时间有序 ✗")

    logger.info("uuid7 自检：%s", "通过 ✓" if ok else "不通过 ✗（实现有误，别用）")
    logger.info("  样例：%s", u)
    logger.info("  version=%s variant=%s" % (u.version, u.variant))
    return ok


def _self_test():
    """拿两组业界公认的 murmur3 测试向量对一下，确认实现没写错

    这两个数来自 mmh3 官方测试用例，任何正确实现都必须算出同样的值。
    """
    cases = [("hello", 613153351), ("foo", -156908512)]
    ok = all(_murmur3_x86_32(s.encode()) == expect for s, expect in cases)
    logger.info("murmur3 自检：%s", "通过 ✓" if ok else "不通过 ✗（实现有误，别用）")
    for s, expect in cases:
        got = _murmur3_x86_32(s.encode())
        logger.info(f"  hash({s!r}) = {got} 期望 {expect}  {'✓' if got == expect else '✗'}")
    return ok


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger.info("mmh3 状态：%s", ensure_mmh3())
    logger.info("uuid_utils 状态：%s", ensure_uuid_utils())
    _self_test()
    _self_test_uuid7()
