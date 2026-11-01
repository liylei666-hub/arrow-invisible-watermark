"""把仓库根目录的源码同步到 ``wxcloudrun/`` 构建上下文。

微信云托管允许把「目标目录」设成 ``wxcloudrun/``，那样构建上下文里必须是
一份完整、自洽的副本。它只是主源码的拷贝，手工复制很容易漏掉一两个文件，
而漏掉的后果是线上镜像和仓库不一致。这个脚本把同步变成一条命令，CI 里的
``test_wxcloudrun_vendor_matches_repo`` 则保证没有人忘记跑。

用法::

    python tools/sync_vendor.py            # 同步（只在有差异时写盘）
    python tools/sync_vendor.py --check    # 只检查，有漂移则退出码 1
"""
from __future__ import annotations

import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR_DIR = "wxcloudrun"

# 相对仓库根目录的路径。Dockerfile 在 wxcloudrun/ 里位于上下文根，
# 因此主仓库根的那一份对应过去。
VENDOR_FILES = (
    "arrow_decode.py",
    "arrow_wm.py",
    "arrow_inner.py",
    "ARROW_clean_pro.png",
    "Dockerfile",
    "services/__init__.py",
    "services/decode_api/__init__.py",
    "services/decode_api/app.py",
    "services/decode_api/requirements.txt",
)

# wxcloudrun/Dockerfile 对应仓库根的 Dockerfile；其余路径一一对应。
_REMAP = {"Dockerfile": "Dockerfile"}


def _dest_rel(rel: str) -> str:
    return _REMAP.get(rel, rel)


def _same(src: str, dst: str) -> bool:
    if not os.path.exists(dst):
        return False
    with open(src, "rb") as a, open(dst, "rb") as b:
        return a.read() == b.read()


def drifted() -> list[str]:
    """返回与主仓库不一致的 wxcloudrun 路径（相对 wxcloudrun/）。"""
    out = []
    for rel in VENDOR_FILES:
        src = os.path.join(ROOT, rel)
        dst = os.path.join(ROOT, VENDOR_DIR, _dest_rel(rel))
        if not os.path.exists(src):
            raise FileNotFoundError(src)
        if not _same(src, dst):
            out.append(_dest_rel(rel))
    return out


def sync() -> list[str]:
    """把有差异的文件复制过去，返回实际写入的路径。"""
    written = []
    for rel in VENDOR_FILES:
        src = os.path.join(ROOT, rel)
        dst = os.path.join(ROOT, VENDOR_DIR, _dest_rel(rel))
        if not os.path.exists(src):
            raise FileNotFoundError(src)
        if _same(src, dst):
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        written.append(_dest_rel(rel))
    return written


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    check_only = "--check" in args
    unknown = [a for a in args if a != "--check"]
    if unknown:
        print(f"未知参数: {' '.join(unknown)}", file=sys.stderr)
        return 2
    if check_only:
        out = drifted()
        if out:
            print("wxcloudrun/ 与主仓库不一致：")
            for rel in out:
                print(f"  - {rel}")
            print("运行 `python tools/sync_vendor.py` 修复。")
            return 1
        print("wxcloudrun/ 与主仓库一致。")
        return 0
    written = sync()
    if written:
        print(f"已同步 {len(written)} 个文件：")
        for rel in written:
            print(f"  - {rel}")
    else:
        print("已经一致，无需改动。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
