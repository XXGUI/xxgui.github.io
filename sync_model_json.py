#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扫描 model/ 目录下全部 .onnx，与根目录 model.json 逐项比对并写回（无需参数）：

  - precision / size / downloadUrl / modelName：由 ONNX 与文件名自动矫正
  - classNames：若少于 ONNX 推断类别数，自动追加「类别0」「类别1」…
  - 磁盘上存在但 JSON 中没有对应条目的 .onnx：自动追加条目（INT8 变体尽量继承同名 FP 条的
    gameType、iconUrl、classNames）
  - gameType / iconUrl：缺省时补 FPS 与空字符串

仅预览不写文件: python sync_model_json.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse

try:
    import onnx
    from onnx import TensorProto
except ImportError:
    print("错误: 需要安装 onnx。例如: pip install onnx", file=sys.stderr)
    sys.exit(1)

# 与现有 model.json / sync_resources_json 中一致的 Raw 前缀（可被 --base-url 覆盖）
DEFAULT_RAW_BASE = (
    "https://raw.githubusercontent.com/XXGUI/xxgui.github.io/refs/heads/main/"
)

DEFAULT_GAME_TYPE = "FPS"
DEFAULT_ICON_URL = ""
INT8_STEM_SUFFIX = "_int8_io"


def _elem_type_name(elem_type: int) -> str:
    try:
        return TensorProto.DataType.Name(elem_type)
    except Exception:
        return f"UNKNOWN({elem_type})"


def _precision_from_elem_type(elem_type: int) -> str:
    if elem_type == TensorProto.FLOAT:
        return "FP32"
    if elem_type == TensorProto.FLOAT16:
        return "FP16"
    if elem_type == TensorProto.INT8:
        return "INT8"
    if elem_type == TensorProto.UINT8:
        return "UINT8"
    if elem_type == TensorProto.INT32:
        return "INT32"
    return _elem_type_name(elem_type)


def _first_graph_input(model: onnx.ModelProto) -> onnx.ValueInfoProto:
    """跳过 initializer 同名的伪输入，取第一个真实 graph input。"""
    init_names = {x.name for x in model.graph.initializer}
    for inp in model.graph.input:
        if inp.name not in init_names:
            return inp
    raise ValueError("未找到可用的 graph 输入")


def _shape_list(t: onnx.TypeProto.Tensor) -> list[Any]:
    out: list[Any] = []
    for d in t.shape.dim:
        if d.dim_value:
            out.append(int(d.dim_value))
        elif d.dim_param:
            out.append(d.dim_param)
        else:
            out.append("?")
    return out


def infer_yolo_num_classes(out_shape: list[Any]) -> int | None:
    """
    常见 YOLO ONNX 输出:
      [1, 4+nc, anchors] 或 [1, anchors, 4+nc]
    在静态 shape 下用较小维减 4 得到 nc；无法判断时返回 None。
    """
    if len(out_shape) != 3 or out_shape[0] not in (1, "?", "batch"):
        return None
    a, b = out_shape[1], out_shape[2]
    if isinstance(a, int) and isinstance(b, int):
        if a < b and a > 4:
            return a - 4
        if b < a and b > 4:
            return b - 4
    return None


def analyze_onnx(path: Path) -> dict[str, Any]:
    m = onnx.load(str(path))
    inp = _first_graph_input(m)
    tt = inp.type.tensor_type
    shape = _shape_list(tt)
    precision = _precision_from_elem_type(tt.elem_type)

    size: int | None = None
    if len(shape) == 4 and isinstance(shape[2], int):
        size = shape[2]
    elif len(shape) == 4 and isinstance(shape[3], int):
        size = shape[3]

    out0 = m.graph.output[0]
    oshape = _shape_list(out0.type.tensor_type)
    nc = infer_yolo_num_classes(oshape)

    return {
        "path": path,
        "input_name": inp.name,
        "input_shape": shape,
        "precision": precision,
        "size": size,
        "output_shape": oshape,
        "inferred_num_classes": nc,
    }


def raw_url_for_model_file(base: str, filename: str) -> str:
    base = base.rstrip("/") + "/"
    return f"{base}model/{quote(filename, safe='')}"


def _ensure_defaults(e: dict[str, Any]) -> None:
    if not e.get("gameType"):
        e["gameType"] = DEFAULT_GAME_TYPE
    if "iconUrl" not in e or e.get("iconUrl") is None:
        e["iconUrl"] = DEFAULT_ICON_URL


def supplement_class_names(
    e: dict[str, Any],
    meta: dict[str, Any],
    logs: list[str],
    tag: str,
) -> None:
    nc = meta.get("inferred_num_classes")
    if nc is None:
        return
    cn = e.get("classNames")
    if not isinstance(cn, list):
        cn = []
        e["classNames"] = cn
    if len(cn) < nc:
        for i in range(len(cn), nc):
            cn.append(f"类别{i}")
        logs.append(f"{tag} 已补充 classNames 至 {nc} 项（推断 nc={nc}）")
    elif len(cn) > nc:
        logs.append(
            f"{tag} classNames 共 {len(cn)} 项，多于 ONNX 推断类别数 {nc}，请手动核对"
        )


def build_entry_from_onnx(
    path: Path,
    meta: dict[str, Any],
    raw_base: str,
    logs: list[str],
    *,
    class_names_seed: list[str] | None = None,
    game_type: str | None = None,
    icon_url: str | None = None,
) -> dict[str, Any]:
    nc = meta.get("inferred_num_classes")
    if class_names_seed is not None:
        cn = list(class_names_seed)
    elif nc is not None:
        cn = [f"类别{i}" for i in range(nc)]
    else:
        cn = []
    e: dict[str, Any] = {
        "modelName": path.stem,
        "precision": meta["precision"],
        "classNames": cn,
        "size": meta["size"] if meta["size"] is not None else 0,
        "gameType": game_type if game_type is not None else DEFAULT_GAME_TYPE,
        "iconUrl": icon_url if icon_url is not None else DEFAULT_ICON_URL,
        "downloadUrl": raw_url_for_model_file(raw_base, path.name),
    }
    supplement_class_names(e, meta, logs, f"[新增 {path.name}]")
    _ensure_defaults(e)
    return e


def parse_download_basename(download_url: str) -> str | None:
    if not download_url or not download_url.strip():
        return None
    p = urlparse(download_url.strip())
    seg = p.path.rstrip("/").split("/")[-1]
    if not seg.lower().endswith(".onnx"):
        return None
    return unquote(seg)


def discover_onnx_by_stem(model_dir: Path) -> dict[str, Path]:
    """stem（无扩展名）-> 路径。"""
    return {p.stem: p for p in sorted(model_dir.glob("*.onnx"))}


def resolve_onnx_path(
    entry: dict[str, Any],
    model_dir: Path,
    onnx_by_stem: dict[str, Path],
) -> Path | None:
    du = entry.get("downloadUrl") or ""
    base = parse_download_basename(du)
    if base and (model_dir / base).is_file():
        return model_dir / base
    name = (entry.get("modelName") or "").strip()
    if name and name in onnx_by_stem:
        # 与 modelName 完全同名的 stem
        return onnx_by_stem[name]
    return None


def sync_models(
    models: list[dict[str, Any]],
    model_dir: Path,
    raw_base: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    stem_map = discover_onnx_by_stem(model_dir)

    used_files: set[Path] = set()
    logs: list[str] = []
    new_models: list[dict[str, Any]] = []

    for i, entry in enumerate(models):
        e = dict(entry)
        opath = resolve_onnx_path(e, model_dir, stem_map)
        if opath is None:
            logs.append(
                f"[{i}] 无法匹配 ONNX: modelName={e.get('modelName')!r} "
                f"downloadUrl={e.get('downloadUrl')!r}"
            )
            _ensure_defaults(e)
            new_models.append(e)
            continue

        used_files.add(opath.resolve())
        try:
            meta = analyze_onnx(opath)
        except Exception as ex:
            logs.append(f"[{i}] 读取失败 {opath.name}: {ex}")
            _ensure_defaults(e)
            new_models.append(e)
            continue

        changes: list[str] = []
        stem = opath.stem
        if e.get("modelName") != stem:
            changes.append(f"modelName: {e.get('modelName')!r} -> {stem!r}")
            e["modelName"] = stem

        new_url = raw_url_for_model_file(raw_base, opath.name)
        if e.get("downloadUrl") != new_url:
            changes.append("downloadUrl: 已按文件名重建")
            e["downloadUrl"] = new_url

        if meta["size"] is not None and e.get("size") != meta["size"]:
            changes.append(f"size: {e.get('size')} -> {meta['size']}")
            e["size"] = meta["size"]

        if e.get("precision") != meta["precision"]:
            changes.append(f"precision: {e.get('precision')!r} -> {meta['precision']!r}")
            e["precision"] = meta["precision"]

        supplement_class_names(e, meta, logs, f"[{i}] {opath.name}")

        _ensure_defaults(e)

        if changes:
            logs.append(f"[{i}] {opath.name}: " + "; ".join(changes))

        new_models.append(e)

    # 按 modelName 建立索引，供 INT8 变体继承字段
    by_name: dict[str, dict[str, Any]] = {}
    for e in new_models:
        mn = e.get("modelName")
        if isinstance(mn, str) and mn:
            by_name[mn] = e

    # 磁盘上尚未被任一条目使用的 .onnx：自动追加条目
    for p in sorted(model_dir.glob("*.onnx"), key=lambda x: sort_key_path(x.name)):
        if p.resolve() in used_files:
            continue
        try:
            meta = analyze_onnx(p)
        except Exception as ex:
            logs.append(f"[新增跳过] {p.name}: 读取失败 {ex}")
            continue

        st = p.stem
        base_stem: str | None = None
        if st.endswith(INT8_STEM_SUFFIX):
            base_stem = st[: -len(INT8_STEM_SUFFIX)]
        base_e = by_name.get(base_stem) if base_stem else None

        if base_e is not None:
            ne = build_entry_from_onnx(
                p,
                meta,
                raw_base,
                logs,
                class_names_seed=list(base_e.get("classNames") or []),
                game_type=str(base_e.get("gameType") or DEFAULT_GAME_TYPE),
                icon_url=str(base_e.get("iconUrl") or DEFAULT_ICON_URL),
            )
        else:
            ne = build_entry_from_onnx(p, meta, raw_base, logs)

        new_models.append(ne)
        by_name[ne["modelName"]] = ne
        used_files.add(p.resolve())
        logs.append(f"[新增条目] {p.name}")

    return new_models, logs


def sort_key_path(s: str) -> str:
    return s.casefold()


def main() -> int:
    ap = argparse.ArgumentParser(description="根据 model/*.onnx 全量矫正并写回 model.json（默认无需参数）")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印变更，不写 model.json",
    )
    ap.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "model",
        help="ONNX 所在目录（默认: 仓库下 model/）",
    )
    ap.add_argument(
        "--json",
        type=Path,
        default=Path(__file__).resolve().parent / "model.json",
        help="model.json 路径（默认: 仓库根目录 model.json）",
    )
    ap.add_argument(
        "--base-url",
        default="",
        help=f"Raw 根 URL（默认: {DEFAULT_RAW_BASE}）",
    )
    ap.add_argument(
        "--indent",
        type=int,
        default=None,
        help="写入 JSON 时的缩进（默认与读入时一致：有 tab 则用 tab）",
    )
    ns = ap.parse_args()
    model_dir = ns.model_dir
    json_path = ns.json
    raw_base = (ns.base_url or "").strip() or DEFAULT_RAW_BASE

    if not model_dir.is_dir():
        print(f"错误: 目录不存在: {model_dir}", file=sys.stderr)
        return 1
    if not json_path.is_file():
        print(f"错误: 文件不存在: {json_path}", file=sys.stderr)
        return 1

    text = json_path.read_text(encoding="utf-8")
    data = json.loads(text)
    models = data.get("models")
    if not isinstance(models, list):
        print("错误: model.json 根对象需包含 models 数组", file=sys.stderr)
        return 1

    new_models, logs = sync_models(models, model_dir, raw_base)
    data["models"] = new_models

    for line in logs:
        print(line)

    if not ns.dry_run:
        indent = ns.indent
        if indent is None:
            indent = "\t" if "\t" in text[:200] else 2
        json_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=indent) + "\n",
            encoding="utf-8",
        )
        print(f"已写入: {json_path}", file=sys.stderr)
    else:
        print("(dry-run) 未写入 model.json", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
