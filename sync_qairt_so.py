#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按目标目录中已有的 .so 文件名，从 QAIRT 源目录对应子目录查找并复制过来。

默认搜索源目录下的（含子目录，如 hexagon-vXX/unsigned）：
  lib/aarch64-android
  lib/hexagon-v66, v68, v69, v73, v75, v79, v81, v85

双击运行：进入交互菜单，可改源/目标目录并开始同步。
命令行：
  python sync_qairt_so.py
  python sync_qairt_so.py --source "C:\\Qualcomm\\AIStack\\QAIRT\\2.51.40.260930"
  python sync_qairt_so.py --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def _ensure_utf8_console() -> None:
    """双击运行时尽量让 Windows 控制台正确显示中文。"""
    if sys.platform != "win32":
        return
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stdin.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        import ctypes

        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)
    except Exception:
        pass


DEFAULT_SOURCE = Path(r"C:\Qualcomm\AIStack\QAIRT\2.51.40.260930")
DEFAULT_TARGET = Path(r"C:\我的github\xxgui.github.io\jniLibs\arm64-v8a")

HEXAGON_VERSIONS = ("66", "68", "69", "73", "75", "79", "81", "85")

DEFAULT_LIB_SUBDIRS = (
    "lib/aarch64-android",
    *(f"lib/hexagon-v{v}" for v in HEXAGON_VERSIONS),
)


def list_target_so_names(target_dir: Path) -> list[str]:
    if not target_dir.is_dir():
        raise FileNotFoundError(f"目标目录不存在: {target_dir}")
    names = sorted(
        p.name for p in target_dir.iterdir() if p.is_file() and p.suffix.lower() == ".so"
    )
    return names


def resolve_search_dirs(source_root: Path, subdirs: list[str]) -> list[Path]:
    dirs: list[Path] = []
    for rel in subdirs:
        p = source_root / rel
        if p.is_dir():
            dirs.append(p)
        else:
            print(f"  [跳过目录] 不存在: {p}")
    return dirs


def build_source_index(search_dirs: list[Path]) -> dict[str, Path]:
    """文件名 -> 首次命中的源路径（按 search_dirs 顺序优先；目录内递归）。"""
    index: dict[str, Path] = {}
    for d in search_dirs:
        for p in d.rglob("*.so"):
            if not p.is_file():
                continue
            if p.name not in index:
                index[p.name] = p
    return index


def sync_so(
    source_root: Path,
    target_dir: Path,
    subdirs: list[str],
    *,
    dry_run: bool = False,
) -> int:
    print(f"源目录: {source_root}")
    print(f"目标目录: {target_dir}")
    print(f"搜索子目录: {', '.join(subdirs)}")
    if dry_run:
        print("模式: dry-run（只预览，不实际复制）")
    print()

    if not source_root.is_dir():
        print(f"错误: 源目录不存在: {source_root}", file=sys.stderr)
        return 1

    try:
        needed = list_target_so_names(target_dir)
    except FileNotFoundError as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1

    if not needed:
        print("目标目录中没有 .so 文件，无需同步。")
        return 0

    print(f"目标目录共 {len(needed)} 个 .so，将按名称在源目录查找：")
    for name in needed:
        print(f"  - {name}")
    print()

    search_dirs = resolve_search_dirs(source_root, subdirs)
    if not search_dirs:
        print("错误: 没有任何可用的搜索子目录。", file=sys.stderr)
        return 1

    index = build_source_index(search_dirs)
    copied = 0
    skipped = 0

    print("同步结果:")
    for name in needed:
        src = index.get(name)
        if src is None:
            print(f"  [未找到] {name}")
            skipped += 1
            continue

        dst = target_dir / name
        rel_src = src.relative_to(source_root)
        if dry_run:
            print(f"  [将复制] {name}  <-  {rel_src}")
        else:
            shutil.copy2(src, dst)
            print(f"  [已复制] {name}  <-  {rel_src}")
        copied += 1

    print()
    print(f"完成: 复制 {copied}，未找到跳过 {skipped}，合计 {len(needed)}")
    return 0


def prompt_path(label: str, current: Path) -> Path:
    print(f"\n当前{label}: {current}")
    raw = input("请输入新路径（直接回车保持不变）: ").strip().strip('"')
    if not raw:
        return current
    return Path(raw).expanduser()


def pause_if_needed() -> None:
    try:
        input("\n按回车键退出...")
    except EOFError:
        pass


def interactive_menu() -> int:
    source = DEFAULT_SOURCE
    target = DEFAULT_TARGET
    subdirs = list(DEFAULT_LIB_SUBDIRS)
    code = 0

    print("=" * 60)
    print("  QAIRT .so 同步工具")
    print("  按目标目录已有 .so 文件名，从源目录查找并复制")
    print("=" * 60)

    while True:
        print()
        print("-" * 60)
        print(f"  源目录: {source}")
        print(f"  目标目录: {target}")
        print(f"  源目录存在: {'是' if source.is_dir() else '否'}")
        print(f"  目标目录存在: {'是' if target.is_dir() else '否'}")
        print("-" * 60)
        print("  [1] 修改源目录")
        print("  [2] 修改目标目录")
        print("  [3] 预览同步（dry-run，不实际复制）")
        print("  [4] 开始同步")
        print("  [0] 退出")
        print("-" * 60)

        choice = input("请选择操作 [0-4]: ").strip()

        if choice == "1":
            source = prompt_path("源目录", source)
        elif choice == "2":
            target = prompt_path("目标目录", target)
        elif choice == "3":
            print()
            code = sync_so(source.resolve(), target.resolve(), subdirs, dry_run=True)
        elif choice == "4":
            print()
            confirm = input(
                f"确认从\n  {source}\n复制到\n  {target}\n开始同步？[Y/n]: "
            ).strip().lower()
            if confirm in ("", "y", "yes"):
                print()
                code = sync_so(source.resolve(), target.resolve(), subdirs, dry_run=False)
            else:
                print("已取消。")
        elif choice == "0":
            print("已退出。")
            break
        else:
            print("无效选项，请输入 0-4。")

    return code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按目标目录 .so 文件名，从 QAIRT 源目录对应子目录复制同名库。",
    )
    parser.add_argument(
        "--source",
        "-s",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"QAIRT 根目录（默认: {DEFAULT_SOURCE}）",
    )
    parser.add_argument(
        "--target",
        "-t",
        type=Path,
        default=DEFAULT_TARGET,
        help=f"jniLibs 目标目录（默认: {DEFAULT_TARGET}）",
    )
    parser.add_argument(
        "--subdir",
        action="append",
        dest="subdirs",
        metavar="REL",
        help=(
            "相对源目录的搜索子目录，可多次指定；"
            "不指定则使用默认 aarch64-android + hexagon-v66/68/69/73/75/79/81/85"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只预览将要复制的文件，不实际写入",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # 无参数 / 双击运行：进入交互菜单
    if not argv:
        code = interactive_menu()
        pause_if_needed()
        return code

    args = parse_args(argv)
    subdirs = args.subdirs if args.subdirs else list(DEFAULT_LIB_SUBDIRS)
    return sync_so(
        args.source.resolve(),
        args.target.resolve(),
        subdirs,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    _ensure_utf8_console()
    raise SystemExit(main())
