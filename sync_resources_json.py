#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扫描 assets/ 与 jniLibs/，按磁盘现状直接重写 resources.json（不做与旧 JSON 的比对）：
  assets：path、url、size、sha256
  jniLibs：文件列表、url、size、sha256

每次运行都根据当前文件重新统计大小与 sha256；增删文件、同名替换均生效。

jniLibs 支持两种目录布局：
  1) 旧结构：<abi>/common/* 与 <abi>/V##/*（HTP）
  2) 新结构：<abi>/*（文件直接放在 ABI 根目录）

若 resources.json 顶层还有其它自定义字段，会从旧文件读入后保留，仅覆盖 assets、jniLibs。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

HTP_DIR_RE = re.compile(r"^V\d+$")

DEFAULT_BASE_URL = (
    "https://raw.githubusercontent.com/XXGUI/xxgui.github.io/refs/heads/main/"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def posix_rel(root: Path, file_path: Path) -> str:
    rel = file_path.relative_to(root).as_posix()
    return rel


def sort_key_path(s: str) -> str:
    """Stable order across OS/locale: case-insensitive path segments."""
    return s.casefold()


def scan_assets(assets_root: Path, base_url: str) -> list[dict]:
    if not assets_root.is_dir():
        return []
    entries: list[dict] = []
    files = [p for p in assets_root.rglob("*") if p.is_file()]
    files.sort(key=lambda p: sort_key_path(posix_rel(assets_root, p)))
    for p in files:
        rel = posix_rel(assets_root, p)
        st = p.stat()
        entries.append(
            {
                "path": rel,
                "url": f"{base_url}assets/{rel}",
                "size": st.st_size,
                "sha256": sha256_file(p),
            }
        )
    return entries


def scan_jni_group_files(group_dir: Path, base_url: str, url_prefix: str) -> list[dict]:
    files: list[dict] = []
    if not group_dir.is_dir():
        return files
    paths = [p for p in group_dir.iterdir() if p.is_file()]
    paths.sort(key=lambda p: sort_key_path(p.name))
    for p in paths:
        name = p.name
        st = p.stat()
        rel_url = f"{url_prefix}/{name}".replace("//", "/")
        files.append(
            {
                "file": name,
                "url": f"{base_url}{rel_url}",
                "size": st.st_size,
                "sha256": sha256_file(p),
            }
        )
    return files


def scan_jni_libs(jni_root: Path, base_url: str) -> dict:
    if not jni_root.is_dir():
        return {}
    out: dict = {}
    for abi_dir in sorted(p for p in jni_root.iterdir() if p.is_dir()):
        abi = abi_dir.name
        block: dict = {}
        # new flat layout: all .so files directly under <abi>/
        flat_files = scan_jni_group_files(abi_dir, base_url, f"jniLibs/{abi}")
        if flat_files:
            block["path"] = f"jniLibs/{abi}"
            block["files"] = flat_files
            out[abi] = block
            continue
        # common
        common_dir = abi_dir / "common"
        if common_dir.is_dir():
            prefix = f"jniLibs/{abi}/common"
            block["common"] = {
                "path": prefix,
                "files": scan_jni_group_files(common_dir, base_url, prefix),
            }
        # htp V##
        htp: dict = {}
        for sub in sorted(p for p in abi_dir.iterdir() if p.is_dir()):
            if sub.name == "common":
                continue
            if not HTP_DIR_RE.match(sub.name):
                continue
            ver = sub.name
            prefix = f"jniLibs/{abi}/{ver}"
            htp[ver] = {
                "path": prefix,
                "htp_arch_ver": ver,
                "files": scan_jni_group_files(sub, base_url, prefix),
            }
        if htp:
            block["htp"] = htp
        if block:
            out[abi] = block
    return out


def load_json(path: Path) -> dict:
    if not path.is_file():
        return {"assets": [], "jniLibs": {}}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="根据 assets/ 与 jniLibs/ 扫描结果直接重写 resources.json（支持旧/新 jniLibs 布局）。"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="仓库根目录（默认：脚本所在目录）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 JSON 路径（默认：<root>/resources.json）",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="资源基础 URL，须以 / 结尾",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印统计，不写入文件",
    )
    args = parser.parse_args()
    root: Path = args.root.resolve()
    out_path = (args.output or (root / "resources.json")).resolve()
    base_url: str = args.base_url
    if not base_url.endswith("/"):
        base_url += "/"

    assets_root = root / "assets"
    jni_root = root / "jniLibs"

    before = load_json(out_path)
    scanned_assets = scan_assets(assets_root, base_url)
    scanned_jni = scan_jni_libs(jni_root, base_url)

    abis = list(scanned_jni.keys())
    print(f"assets：{len(scanned_assets)} 个文件；jniLibs 架构：{abis}")

    if args.dry_run:
        print("预览结束（未写入文件）。")
        return 0

    data = {**before}
    data["assets"] = scanned_assets
    data["jniLibs"] = scanned_jni

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"已写入：{out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
