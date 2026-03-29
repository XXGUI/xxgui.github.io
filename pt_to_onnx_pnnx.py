#!/usr/bin/env python3
"""
YOLO / PyTorch .pt → ONNX

- YOLOv8 / YOLO11 / YOLO12 等：用 ultralytics 的 YOLO().export(onnx)（pip install ultralytics）。
- YOLOv5 旧权重：Ultralytics 无法加载时，自动改用 torch hub 的 ultralytics/yolov5/export.run。
- 若只想走 v8、禁止回退 v5：加 --ultralytics-only

依赖：pip install ultralytics torch
YOLOv5 分支另需 onnx 等（见 yolov5 export 要求）。

可选：TorchScript (.pt) + 本机 pnnx → .pnnx.onnx

Interactive:
  python pt_to_onnx_pnnx.py

CLI (YOLO, default):
  python pt_to_onnx_pnnx.py weights.pt
  python pt_to_onnx_pnnx.py weights.pt --imgsz 256 --half

CLI (pnnx / TorchScript only):
  python pt_to_onnx_pnnx.py model.ts.pt --backend pnnx
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


# 固定 pnnx 路径（仅 --backend pnnx 时使用）
DEFAULT_PNNX = r"C:/pnnx/pnnx-20251119-windows/pnnx"

DEFAULT_INPUTSHAPE = "[1,3,256,256]"
DEFAULT_IMGSZ = 256


def _yolov5_hub_root() -> Path:
    import torch

    hub_dir = Path(torch.hub.get_dir())
    repos = list(hub_dir.glob("ultralytics_yolov5*"))
    if not repos:
        torch.hub.load(
            "ultralytics/yolov5",
            "yolov5n",
            pretrained=True,
            trust_repo=True,
        )
        repos = list(hub_dir.glob("ultralytics_yolov5*"))
    if not repos:
        raise FileNotFoundError("torch hub 中未找到 ultralytics/yolov5，请检查网络后重试")
    return max(repos, key=lambda p: p.stat().st_mtime)


def export_yolov5_torch_hub(pt: Path, ns: argparse.Namespace) -> Path:
    """Use official YOLOv5 repo export.run (via torch hub cache). Requires onnx, onnx-simplifier, etc."""
    import torch

    root = _yolov5_hub_root()
    root_str = str(root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    from export import run  # type: ignore  # YOLOv5 export.py

    imgsz = int(getattr(ns, "imgsz", DEFAULT_IMGSZ))
    half = bool(getattr(ns, "half", False))
    simplify = getattr(ns, "simplify", True)
    dynamic = bool(getattr(ns, "dynamic", False))
    opset = getattr(ns, "opset", None)
    if opset is None:
        opset = 12

    device = "0" if half and torch.cuda.is_available() else "cpu"
    use_half = bool(half and torch.cuda.is_available())
    if half and not use_half:
        print("提示: YOLOv5 在 CPU 上无法使用 --half FP16 导出，已改用 FP32。", file=sys.stderr)

    w = str(pt.resolve())
    out_files = run(
        weights=w,
        imgsz=(imgsz, imgsz),
        batch_size=1,
        device=device,
        include=("onnx",),
        half=use_half,
        inplace=True,
        dynamic=dynamic,
        simplify=simplify,
        opset=int(opset),
    )
    if not out_files:
        raise RuntimeError("YOLOv5 export 未生成任何文件")
    onnx_path = next((Path(x) for x in out_files if str(x).lower().endswith(".onnx")), None)
    if onnx_path is None or not onnx_path.is_file():
        raise RuntimeError(f"未找到 ONNX 输出: {out_files}")
    return onnx_path


def _should_try_yolov5_fallback(err: BaseException) -> bool:
    s = str(err).lower()
    return (
        "yolov5" in s
        or "forwards compatible" in s
        or "not forwards compatible" in s
    )


def precheck_torchscript(pt: Path, force: bool) -> int:
    """Return 0 if OK to run pnnx, 1 to abort (bad or non-TorchScript .pt)."""
    if force:
        return 0
    try:
        import torch
    except ImportError:
        return 0
    try:
        torch.jit.load(str(pt.resolve()), map_location="cpu")
    except Exception as e:
        print(
            "\n[预检失败] 该 .pt 不是可用的 TorchScript（pnnx 需要 trace/script 保存的 .pt）。",
            file=sys.stderr,
        )
        print(f"  torch.jit.load: {e}", file=sys.stderr)
        print(
            "\n若实际是 YOLO 训练权重，请用默认方式（不要加 --backend pnnx）：\n"
            "  python pt_to_onnx_pnnx.py 你的权重.pt\n",
            file=sys.stderr,
        )
        return 1
    return 0


def resolved_pnnx_executable() -> Path:
    p = Path(DEFAULT_PNNX)
    if p.is_file():
        return p
    p_exe = p.with_suffix(".exe")
    if p_exe.is_file():
        return p_exe
    return p


def build_pnnx_args(ns: argparse.Namespace) -> list[str]:
    pt = ns.pt.resolve()
    cmd: list[str] = [str(ns.pnnx.resolve()), str(pt)]

    cmd.append(f"inputshape={ns.inputshape.strip()}")
    if ns.inputshape2:
        cmd.append(f"inputshape2={ns.inputshape2.strip()}")
    cmd.append(f"device={ns.device}")
    cmd.append(f"fp16={ns.fp16}")
    cmd.append(f"optlevel={ns.optlevel}")

    if ns.pnnxonnx:
        cmd.append(f"pnnxonnx={ns.pnnxonnx}")
    if ns.pnnxparam:
        cmd.append(f"pnnxparam={ns.pnnxparam}")
    if ns.customop:
        cmd.append(f"customop={ns.customop}")
    if ns.moduleop:
        cmd.append(f"moduleop={ns.moduleop}")

    return cmd


def _input_with_default(prompt: str, default: str) -> str:
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw if raw else default


def _input_optional(prompt: str) -> str:
    return input(f"{prompt} (回车跳过): ").strip()


def _prompt_pt(label: str = "1.") -> Path | None:
    while True:
        pt_raw = input(f"{label} 模型 .pt 路径: ").strip().strip('"')
        if not pt_raw:
            print("  未输入路径，退出。")
            return None
        pt = Path(pt_raw)
        if pt.is_file():
            return pt
        print(f"  文件不存在: {pt}")


def interactive_yolo_namespace() -> argparse.Namespace | None:
    print(
        "YOLO 权重 → ONNX\n"
        "  · 选 1：YOLOv8 / YOLO11 等（Ultralytics）；若为旧 YOLOv5 可自动改走 yolov5 导出\n"
        "  · 选 2：仅旧版 YOLOv5 权重（不经 Ultralytics，直接用 torch hub yolov5）\n"
    )
    kind = _input_with_default("1. 权重类型 (1 / 2)", "1").strip()
    yolov5_hub = kind == "2"

    pt = _prompt_pt("2.")
    if pt is None:
        return None

    imgsz_s = _input_with_default("3. 输入边长 imgsz", str(DEFAULT_IMGSZ))
    try:
        imgsz = int(imgsz_s)
    except ValueError:
        print("  无效，使用", DEFAULT_IMGSZ)
        imgsz = DEFAULT_IMGSZ

    half_s = _input_with_default("4. ONNX 半精度 half (y/n)", "n").lower()
    half = half_s in ("y", "yes", "1")

    dyn_s = _input_with_default("5. 动态 batch/尺寸 dynamic (y/n)", "n").lower()
    dynamic = dyn_s in ("y", "yes", "1")

    cwd_hint = Path.cwd()
    onnx_out = _input_optional(
        f"6. ONNX 输出路径（回车则当前目录: {cwd_hint / (pt.stem + '.onnx')}）"
    )

    ultralytics_only = False
    if not yolov5_hub:
        uo = _input_with_default(
            "7. 仅 Ultralytics（YOLOv8 等），失败时不尝试 YOLOv5 回退？(y/n)",
            "n",
        ).lower()
        ultralytics_only = uo in ("y", "yes", "1")

    return argparse.Namespace(
        pt=pt,
        backend="yolo",
        imgsz=imgsz,
        half=half,
        simplify=True,
        dynamic=dynamic,
        opset=None,
        onnx_out=onnx_out,
        yolov5_hub=yolov5_hub,
        ultralytics_only=ultralytics_only,
    )


def interactive_pnnx_namespace() -> argparse.Namespace | None:
    print("TorchScript .pt → pnnx → ONNX\n")

    pt = _prompt_pt()
    if pt is None:
        return None

    if precheck_torchscript(pt, force=False) != 0:
        return None

    inputshape = _input_with_default(
        '2. 输入 shape（例 "[1,3,256,256]"）',
        DEFAULT_INPUTSHAPE,
    )

    inputshape2 = _input_optional("3. 可选 inputshape2")

    dev_in = _input_with_default("4. 设备 (cpu / gpu)", "cpu").lower()
    device = "gpu" if dev_in == "gpu" else "cpu"

    fp16_raw = _input_with_default("5. pnnx fp16 (0/1)", "1")
    fp16 = 1 if fp16_raw != "0" else 0

    opt_raw = _input_with_default("6. optlevel (0/1/2)", "2")
    try:
        optlevel = int(opt_raw)
        if optlevel not in (0, 1, 2):
            raise ValueError
    except ValueError:
        print("  无效，使用 2")
        optlevel = 2

    cwd_hint = Path.cwd()
    pnnxonnx = _input_optional(
        f"7. pnnxonnx（回车则当前目录: {cwd_hint / (pt.stem + '.pnnx.onnx')}）"
    )

    more = _input_with_default("8. moduleop/customop？(y/n)", "n").lower()
    pnnxparam = ""
    customop = ""
    moduleop = ""
    if more in ("y", "yes", "1"):
        pnnxparam = _input_optional("   pnnxparam")
        customop = _input_optional("   customop")
        moduleop = _input_optional("   moduleop")

    return argparse.Namespace(
        pt=pt,
        backend="pnnx",
        inputshape=inputshape,
        inputshape2=inputshape2,
        device=device,
        fp16=fp16,
        optlevel=optlevel,
        pnnxonnx=pnnxonnx,
        pnnxparam=pnnxparam,
        customop=customop,
        moduleop=moduleop,
    )


def interactive_main() -> int:
    mode = _input_with_default(
        "模式: 1=YOLO权重→ONNX  2=TorchScript→pnnx",
        "1",
    ).strip()
    if mode == "2":
        ns = interactive_pnnx_namespace()
        if ns is None:
            return 1
        return run_pnnx(ns)
    ns = interactive_yolo_namespace()
    if ns is None:
        return 1
    return run_yolo_export(ns)


def run_yolo_export(ns: argparse.Namespace) -> int:
    pt = ns.pt
    if not pt.is_file():
        print(f"Error: .pt not found: {pt}", file=sys.stderr)
        return 1

    imgsz = int(getattr(ns, "imgsz", DEFAULT_IMGSZ))
    half = bool(getattr(ns, "half", False))
    simplify = getattr(ns, "simplify", True)
    dynamic = bool(getattr(ns, "dynamic", False))
    opset = getattr(ns, "opset", None)

    dest_str = (getattr(ns, "onnx_out", None) or "").strip()
    if not dest_str:
        dest_str = str((Path.cwd() / f"{pt.stem}.onnx").resolve())
    dest = Path(dest_str)
    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n加载: {pt.resolve()}\n导出 ONNX: imgsz={imgsz}, half={half}, dynamic={dynamic}\n")

    prefer_v5 = bool(getattr(ns, "yolov5_hub", False))
    ultralytics_only = bool(getattr(ns, "ultralytics_only", False))
    exported_path: Path | None = None

    if not prefer_v5:
        try:
            from ultralytics import YOLO
        except ImportError:
            print("Error: 需要 ultralytics。请执行: pip install ultralytics", file=sys.stderr)
            return 1
        try:
            model = YOLO(str(pt.resolve()))
            kwargs: dict = {
                "format": "onnx",
                "imgsz": imgsz,
                "half": half,
                "simplify": simplify,
                "dynamic": dynamic,
            }
            if opset is not None:
                kwargs["opset"] = int(opset)
            exported = model.export(**kwargs)
            exported_path = Path(exported)
        except Exception as e:
            if _should_try_yolov5_fallback(e) and not ultralytics_only:
                print(
                    "\n改用 YOLOv5 官方仓库导出（torch hub: ultralytics/yolov5）…\n",
                    file=sys.stderr,
                )
                try:
                    exported_path = export_yolov5_torch_hub(pt, ns)
                except Exception as e2:
                    print(f"Error: YOLOv5 导出失败: {e2}", file=sys.stderr)
                    print(f"（Ultralytics 报错: {e}）", file=sys.stderr)
                    return 1
            else:
                if ultralytics_only and _should_try_yolov5_fallback(e):
                    print(
                        "Error: Ultralytics 无法加载（可能为 YOLOv5 权重）。"
                        "已启用仅 v8 模式：请去掉 --ultralytics-only 再试，或改用 --yolov5-hub。",
                        file=sys.stderr,
                    )
                    print(f"  原始错误: {e}", file=sys.stderr)
                    return 1
                print(f"Error: 无法加载或导出: {e}", file=sys.stderr)
                return 1
    else:
        try:
            exported_path = export_yolov5_torch_hub(pt, ns)
        except Exception as e:
            print(f"Error: YOLOv5 导出失败: {e}", file=sys.stderr)
            print("若缺少依赖，可安装: pip install onnx onnx-simplifier pandas", file=sys.stderr)
            return 1

    if exported_path is None or not exported_path.is_file():
        print(f"Error: 未找到导出文件: {exported_path}", file=sys.stderr)
        return 1

    try:
        if exported_path.resolve() != dest.resolve():
            if dest.exists():
                dest.unlink()
            shutil.move(str(exported_path), str(dest))
        final = dest
    except OSError as e:
        print(f"Error: 无法移动到目标路径: {e}", file=sys.stderr)
        print(f"导出位置仍在: {exported_path.resolve()}", file=sys.stderr)
        return 1

    print(f"\nONNX: {final.resolve()}")
    return 0


def run_pnnx(ns: argparse.Namespace) -> int:
    pt = ns.pt
    if not pt.is_file():
        print(f"Error: .pt not found: {pt}", file=sys.stderr)
        return 1
    force = bool(getattr(ns, "force_pnnx", False))
    if precheck_torchscript(pt, force=force) != 0:
        return 1
    ns.pnnx = resolved_pnnx_executable()
    exe = ns.pnnx
    if not exe.is_file():
        print(f"Error: pnnx not found: {exe}", file=sys.stderr)
        return 1

    if not (getattr(ns, "pnnxonnx", None) and str(ns.pnnxonnx).strip()):
        ns.pnnxonnx = str((Path.cwd() / f"{pt.stem}.pnnx.onnx").resolve())

    cmd = build_pnnx_args(ns)
    print("\n执行:", " ".join(cmd), "\n")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        return e.returncode
    except OSError as e:
        print(f"Error: failed to run pnnx: {e}", file=sys.stderr)
        return 1

    out = Path(ns.pnnxonnx)
    if out.is_file():
        print(f"\nONNX: {out.resolve()}")
    else:
        print(f"\n若成功应生成: {out.resolve()}", file=sys.stderr)

    return 0


def main() -> int:
    if len(sys.argv) <= 1:
        return interactive_main()

    parser = argparse.ArgumentParser(
        description="YOLOv8/11 (ultralytics) or YOLOv5 (hub) .pt → ONNX; or TorchScript → pnnx."
    )
    parser.add_argument("pt", type=Path, help="Path to .pt weights or TorchScript")
    parser.add_argument(
        "--backend",
        choices=("yolo", "pnnx"),
        default="yolo",
        help="yolo=Ultralytics export (default); pnnx=TorchScript+pnnx",
    )
    # YOLO
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
        help=f"YOLO export image size (default {DEFAULT_IMGSZ})",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="YOLO: FP16 ONNX",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="YOLO: dynamic axes",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=None,
        help="YOLO: ONNX opset (optional)",
    )
    parser.add_argument(
        "--onnx",
        dest="onnx_out",
        default="",
        help="YOLO: output .onnx path (default: cwd/<stem>.onnx)",
    )
    parser.add_argument(
        "--no-simplify",
        dest="simplify",
        action="store_false",
        default=True,
        help="YOLO: disable onnx simplifier",
    )
    parser.add_argument(
        "--yolov5-hub",
        action="store_true",
        help="Skip ultralytics YOLO(); use YOLOv5 repo export only (old v5 .pt)",
    )
    parser.add_argument(
        "--ultralytics-only",
        "--yolov8-only",
        action="store_true",
        dest="ultralytics_only",
        help="YOLOv8/11: use ultralytics only; do not fall back to YOLOv5 hub on error",
    )
    # pnnx
    parser.add_argument(
        "--inputshape",
        default=DEFAULT_INPUTSHAPE,
        help="pnnx: input shape",
    )
    parser.add_argument("--inputshape2", default="", help="pnnx")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu", help="pnnx")
    parser.add_argument("--fp16", type=int, choices=(0, 1), default=1, help="pnnx")
    parser.add_argument("--optlevel", type=int, choices=(0, 1, 2), default=2, help="pnnx")
    parser.add_argument("--pnnxonnx", default="", help="pnnx ONNX output path")
    parser.add_argument("--pnnxparam", default="", help="pnnx")
    parser.add_argument("--customop", default="", help="pnnx")
    parser.add_argument("--moduleop", default="", help="pnnx")
    parser.add_argument(
        "--force-pnnx",
        action="store_true",
        help="pnnx: skip torch.jit.load pre-check",
    )

    ns = parser.parse_args()
    if ns.backend == "yolo":
        return run_yolo_export(ns)
    return run_pnnx(ns)


if __name__ == "__main__":
    sys.exit(main())
