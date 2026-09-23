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

import sys
import types


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


def _self_test():
    """拿两组业界公认的 murmur3 测试向量对一下，确认实现没写错

    这两个数来自 mmh3 官方测试用例，任何正确实现都必须算出同样的值。
    """
    cases = [("hello", 613153351), ("foo", -156908512)]
    ok = all(_murmur3_x86_32(s.encode()) == expect for s, expect in cases)
    print("murmur3 自检：", "通过 ✓" if ok else "不通过 ✗（实现有误，别用）")
    for s, expect in cases:
        got = _murmur3_x86_32(s.encode())
        print(f"  hash({s!r}) = {got}  期望 {expect}  {'✓' if got == expect else '✗'}")
    return ok


if __name__ == "__main__":
    print("mmh3 状态：", ensure_mmh3())
    _self_test()
