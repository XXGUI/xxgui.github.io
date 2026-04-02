#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通过 adb 读取设备 SoC，识别骁龙（SM 系列），并从同目录 cpu.json 查询 device_id / htp_arch_ver。
非骁龙 SoC 则提示并退出。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


# 用于识别骁龙型号（SMxxxx、SM8750P、SM8750-AC 等）；联发科等为 MT 开头
SM_RE = re.compile(r"(?i)\b(SM\d{4}[A-Z0-9-]*)\b")
MT_RE = re.compile(r"(?i)\b(MT\d{4}[A-Z0-9]*)\b")

# 常见可反映 SoC 的 getprop 键（按优先级大致排序）
SOC_GETPROP_KEYS = (
    "ro.soc.model",
    "ro.hardware.chipname",
    "ro.boot.hardware.chipname",
    "ro.vendor.qti.soc.model",
    "ro.product.board",
    "ro.board.platform",
    "ro.boot.board.platform",
)


def _run_adb(args: list[str], serial: str | None) -> tuple[int, str]:
    cmd = ["adb"]
    if serial:
        cmd.extend(["-s", serial])
    cmd.extend(args)
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
    except FileNotFoundError:
        print("错误: 未找到 adb，请安装 Android SDK Platform-Tools 并加入 PATH。", file=sys.stderr)
        sys.exit(2)
    except subprocess.TimeoutExpired:
        print("错误: adb 命令超时。", file=sys.stderr)
        sys.exit(2)
    out = (p.stdout or "") + (p.stderr or "")
    return p.returncode, out


def adb_shell(serial: str | None, shell_cmd: str) -> str:
    code, out = _run_adb(["shell", shell_cmd], serial)
    if code != 0:
        return ""
    return out


def collect_soc_text(serial: str | None) -> str:
    parts: list[str] = []
    for key in SOC_GETPROP_KEYS:
        line = adb_shell(serial, f"getprop {key}").strip()
        if line:
            parts.append(f"{key}={line}")
    # 全量 getprop 中再搜一遍（部分机型只在非常规键里带型号）
    dump = adb_shell(serial, "getprop")
    if dump:
        parts.append(dump)
    cpuinfo = adb_shell(serial, "cat /proc/cpuinfo 2>/dev/null")
    if cpuinfo:
        parts.append(cpuinfo)
    return "\n".join(parts)


def normalize_sm(raw: str) -> str:
    s = raw.strip()
    m = re.match(r"(?i)^sm(\d{4})(.*)$", s)
    if m:
        rest = m.group(2).upper()
        return "SM" + m.group(1) + rest
    return s.upper()


def extract_sm_codes(text: str) -> list[str]:
    found: list[str] = []
    for m in SM_RE.finditer(text):
        found.append(normalize_sm(m.group(1)))
    # ro.board.platform 常为 sm8650 / sm8750-ac 等形式
    for m in re.finditer(r"(?i)\b(sm\d{4}[a-z0-9-]*)\b", text):
        found.append(normalize_sm(m.group(1)))
    # 去重保序
    seen: set[str] = set()
    out: list[str] = []
    for c in found:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def has_mEDIATEK_hint(text: str) -> bool:
    if MT_RE.search(text):
        return True
    t = text.lower()
    return "mediatek" in t or "mt687" in t or "mt688" in t or "mt689" in t or "mt698" in t or "mt699" in t


def load_cpu_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("chips", [])


def is_snapdragon_chip(entry: dict) -> bool:
    code = str(entry.get("code") or "")
    return code.upper().startswith("SM")


def match_chip(sm_code: str, chips: list[dict]) -> dict | None:
    """在 chips 中匹配最接近的 SM 条目。"""
    sm_code = sm_code.upper().strip()
    sm_list = [c for c in chips if is_snapdragon_chip(c)]
    if not sm_list:
        return None

    for c in sm_list:
        if str(c.get("code", "")).upper() == sm_code:
            return c

    # 设备较短：SM8750 对应库中 SM8750-AC 等
    prefixed = [
        c
        for c in sm_list
        if str(c.get("code", "")).upper().startswith(sm_code + "-")
    ]
    if prefixed:
        prefixed.sort(key=lambda x: len(str(x.get("code", ""))), reverse=True)
        return prefixed[0]

    # 设备较长：SM8750-3-AB 对应库中 SM8750-3-AB
    for c in sorted(sm_list, key=lambda x: -len(str(x.get("code", "")))):
        cu = str(c.get("code", "")).upper()
        if sm_code.startswith(cu) and (len(sm_code) == len(cu) or sm_code[len(cu) :].startswith("-")):
            return c

    return None


def pick_best_sm_code(codes: list[str]) -> str | None:
    if not codes:
        return None
    # 优先最长（更具体）
    return max(codes, key=len)


def main() -> None:
    parser = argparse.ArgumentParser(description="通过 adb 查询骁龙 SoC、device_id、htp_arch_ver")
    parser.add_argument("-s", "--serial", help="adb 设备序列号（多设备时）")
    parser.add_argument(
        "-j",
        "--json",
        type=Path,
        default=None,
        help="cpu.json 路径（默认：与本脚本同目录下的 cpu.json）",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    json_path = args.json or (script_dir / "cpu.json")
    if not json_path.is_file():
        print(f"错误: 找不到数据文件: {json_path}", file=sys.stderr)
        sys.exit(2)

    chips = load_cpu_json(json_path)

    # 确认设备在线
    code, devices_out = _run_adb(["devices"], args.serial)
    if code != 0:
        print("错误: adb devices 执行失败。", file=sys.stderr)
        sys.exit(2)
    lines = [ln.strip() for ln in devices_out.splitlines() if ln.strip() and not ln.startswith("List")]
    if not lines:
        print("错误: 未检测到已连接且授权的设备，请连接手机并允许 USB 调试。", file=sys.stderr)
        sys.exit(2)

    text = collect_soc_text(args.serial)
    if not text.strip():
        print("错误: 无法通过 adb shell 读取设备信息。", file=sys.stderr)
        sys.exit(2)

    if has_mEDIATEK_hint(text) and not extract_sm_codes(text):
        print("当前设备不是骁龙（检测到联发科或其它非 SM 型号线索），已退出。")
        sys.exit(0)

    codes = extract_sm_codes(text)
    sm_code = pick_best_sm_code(codes)

    if not sm_code:
        print("当前设备不是骁龙（未解析到 SM 系列型号），已退出。")
        sys.exit(0)

    entry = match_chip(sm_code, chips)
    marketing = str(entry.get("name", "")) if entry else "(cpu.json 中无匹配条目)"
    device_id = entry.get("device_id") if entry else None
    htp_arch_ver = entry.get("htp_arch_ver") if entry else None

    print("处理器型号（骁龙）:", marketing)
    print("SoC 代码:", sm_code)
    print("device_id:", device_id if device_id is not None else "未知（请补充 cpu.json）")
    print("htp_arch_ver:", htp_arch_ver if htp_arch_ver is not None else "未知（请补充 cpu.json）")


if __name__ == "__main__":
    main()
