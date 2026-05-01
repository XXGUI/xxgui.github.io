#!/usr/bin/env python3
"""
ONNX -> TensorFlow SavedModel -> TFLite INT8（全整型推理 I/O）

安装::

    pip install -r requirements-onnx2tflite.txt

说明：

- onnx-tf 会间接加载 tensorflow_probability，需单独安装 **tf-keras**（否则会报
  ``No module named 'tf_keras'``）。
- TensorFlow **2.16+** 随 Keras 3 升级后，onnx-tf 依赖的 **tensorflow_addons**
  常报错（如 ``No module named 'keras.src.engine'``）。本仓库的 requirements
  将 tensorflow 限制在 2.14–2.15；请在独立 venv 中安装。

环境变量：``TF_CPP_MIN_LOG_LEVEL=2`` 减少日志；``TF_ENABLE_ONEDNN_OPTS=0`` 关闭 oneDNN。
"""

from __future__ import annotations

import argparse
import importlib.metadata as ilm
import os
import sys
from typing import Any

# 在 import tensorflow 之前降低 C++ 日志噪声
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def _die(msg: str, hint: str | None = None) -> None:
    print(msg, file=sys.stderr)
    if hint:
        print(hint, file=sys.stderr)
    raise SystemExit(1)


def _parse_major_minor(ver: str) -> tuple[int, int]:
    parts = ver.split(".")
    major = int(parts[0]) if parts and parts[0].isdigit() else 0
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    return major, minor


def _precheck_tensorflow_version() -> None:
    """在导入 tensorflow 之前检查版本，避免不兼容环境输出大量噪音日志。"""
    try:
        tf_ver = ilm.version("tensorflow")
    except Exception:
        # tensorflow 未安装或 metadata 不可读，交给后续 import 逻辑报错
        return
    major, minor = _parse_major_minor(tf_ver)
    if major > 2 or (major == 2 and minor >= 16):
        _die(
            f"检测到 tensorflow=={tf_ver}，与 onnx-tf 常见不兼容（Keras 3 / tensorflow_addons）。",
            "请用独立环境安装兼容依赖后再运行：\n"
            "  python -m venv .venv-tflite\n"
            "  .\\.venv-tflite\\Scripts\\activate\n"
            "  pip install -r requirements-onnx2tflite.txt",
        )


def _import_ml_stack():
    """延迟导入：便于 ``--help`` 秒开，并集中给出依赖错误说明。"""
    _precheck_tensorflow_version()
    try:
        import onnx
    except ModuleNotFoundError:
        _die("未安装 onnx。", "  pip install onnx")
    try:
        import tensorflow as tf
    except ModuleNotFoundError:
        _die("未安装 tensorflow。", "  pip install tensorflow")
    try:
        import tf_keras  # noqa: F401  # tensorflow_probability 经 onnx-tf 间接需要
    except ModuleNotFoundError:
        _die(
            "未安装 tf-keras（onnx-tf → tensorflow_probability 需要）。",
            "  pip install tf-keras\n"
            "  或: pip install tensorflow-probability[tf]",
        )
    try:
        from onnx_tf.backend import prepare
    except ModuleNotFoundError as e:
        _die("未安装 onnx-tf 或不在当前 Python 环境中。", f"  pip install onnx-tf\n  ({e!r})")
    except Exception as e:
        ver = getattr(sys.modules.get("tensorflow"), "__version__", "?")
        _die(
            f"导入 onnx-tf 失败（当前 tensorflow=={ver}）。底层错误: {e!r}",
            "  1) 确认已安装: pip install tf-keras onnx-tf\n"
            "  2) 若使用 TensorFlow 2.16+，请新建 venv 并安装:\n"
            "       pip install -r requirements-onnx2tflite.txt\n"
            "     其中将 tensorflow 限制在 2.14.x–2.15.x，以避免 tensorflow_addons 与 Keras 3 不兼容。",
        )
    try:
        import cv2
    except ModuleNotFoundError:
        _die("未安装 OpenCV。", "  pip install opencv-python-headless")
    import numpy as np

    return np, onnx, tf, cv2, prepare


def input_with_default(prompt: str, default: str) -> str:
    val = input(f"{prompt} [default: {default}]: ")
    return val.strip() or default


def _graph_user_inputs(model: Any) -> list:
    """排除已作为 initializer 挂名的伪输入，避免多算一个权重张量。"""
    init_names = {t.name for t in model.graph.initializer}
    return [inp for inp in model.graph.input if inp.name not in init_names]


def _shape_from_value_info(inp: Any, batch_size: int) -> tuple[list[int], bool]:
    """返回 (shape 列表, 是否含动态维)。动态维用 batch_size 占位。"""
    shape: list[int] = []
    dynamic = False
    for dim in inp.type.tensor_type.shape.dim:
        if dim.dim_value > 0:
            shape.append(int(dim.dim_value))
        elif dim.dim_param:
            dynamic = True
            shape.append(batch_size)
        else:
            dynamic = True
            shape.append(batch_size)
    return shape, dynamic


def _saved_model_input_order(saved_model_dir: str, tf: Any) -> list[str]:
    m = tf.saved_model.load(saved_model_dir)
    sig = m.signatures.get("serving_default")
    if sig is None:
        keys = list(m.signatures.keys())
        if not keys:
            raise RuntimeError("SavedModel 中没有任何 signature")
        sig = m.signatures[keys[0]]
    return list(sig.structured_input_signature[1].keys())


def _align_inputs_to_signature(
    saved_model_dir: str,
    input_info: list[tuple[str, list[int]]],
    tf: Any,
) -> list[tuple[str, list[int]]]:
    """与 TFLiteConverter 期望的代表数据顺序一致（按 serving_default 参数序）。"""
    order = _saved_model_input_order(saved_model_dir, tf)
    by_name = dict(input_info)
    out: list[tuple[str, list[int]]] = []
    for name in order:
        if name in by_name:
            out.append((name, by_name[name]))
    for name, sh in input_info:
        if name not in {n for n, _ in out}:
            out.append((name, sh))
    return out


def representative_dataset_gen(
    image_dir: str,
    input_info: list[tuple[str, list[int]]],
    num_calib: int,
    mean: float,
    std: float,
    cv2: Any,
    np: Any,
):
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    img_files = [
        os.path.join(image_dir, f)
        for f in sorted(os.listdir(image_dir))
        if f.lower().endswith(exts)
    ]
    if not img_files:
        raise RuntimeError(f"代表性数据集目录 {image_dir} 中没有支持的图片。")
    if abs(std) < 1e-12:
        raise ValueError("std 不能为 0（避免除零）。")

    count = 0
    for img_path in img_files:
        if count >= num_calib:
            break
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告: 无法读取图片，跳过: {img_path}", file=sys.stderr)
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        feed: list[Any] = []
        for _name, shape in input_info:
            if len(shape) == 4:
                _b, h, w, c = shape[0], shape[1], shape[2], shape[3]
                resized = cv2.resize(img, (int(w), int(h)))
                resized = resized.astype(np.float32)
                resized = (resized - mean) / std
                batch_data = np.expand_dims(resized, axis=0)
            else:
                batch_data = np.random.default_rng(0).random(
                    shape, dtype=np.float32
                )
            feed.append(batch_data)
        yield feed
        count += 1

    if count == 0:
        raise RuntimeError("没有成功加载任何校准图片（可能全部损坏或为空）。")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ONNX -> TFLite 全 INT8（含 I/O int8）")
    p.add_argument("--onnx", default="model.onnx", help="ONNX 模型路径")
    p.add_argument(
        "--saved-model-dir",
        default="tmp_saved_model",
        help="中间 SavedModel 目录",
    )
    p.add_argument("--output", "-o", default="model_int8.tflite", help="输出 .tflite")
    p.add_argument(
        "--images",
        default="representative_images",
        help="代表性图片目录（用于整型校准）",
    )
    p.add_argument("--batch", type=int, default=1, help="动态 batch 维占位大小")
    p.add_argument("--num-calib", type=int, default=100, help="最多使用多少张图做校准")
    p.add_argument("--mean", type=float, default=0.0, help="输入减均值")
    p.add_argument("--std", type=float, default=1.0, help="输入除标准差")
    p.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="逐项询问路径与参数（覆盖命令行默认值）",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    np, onnx, tf, cv2, prepare = _import_ml_stack()

    if args.interactive:
        args.onnx = input_with_default("ONNX 模型路径", args.onnx)
        args.saved_model_dir = input_with_default(
            "临时 TensorFlow SavedModel 目录", args.saved_model_dir
        )
        args.output = input_with_default("TFLite 输出路径", args.output)
        args.images = input_with_default("代表性数据集目录", args.images)
        args.batch = int(input_with_default("Batch 大小", str(args.batch)))
        args.num_calib = int(
            input_with_default("代表性数据集数量", str(args.num_calib))
        )
        args.mean = float(input_with_default("输入归一化均值 (mean)", str(args.mean)))
        args.std = float(input_with_default("输入归一化标准差 (std)", str(args.std)))

    print("=== ONNX -> TFLite INT8 转换工具 ===")

    if not os.path.isfile(args.onnx):
        _die(f"找不到 ONNX 文件: {args.onnx}")

    print(
        f"\n[1/4] ONNX -> SavedModel: {args.onnx} -> {args.saved_model_dir} ...",
        flush=True,
    )
    onnx_model = onnx.load(args.onnx)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(args.saved_model_dir)
    print(f"SavedModel 已保存到 {args.saved_model_dir}")

    onnx_inputs = _graph_user_inputs(onnx_model)
    input_info: list[tuple[str, list[int]]] = []
    print("\n[2/4] 模型输入（已排除 initializer 伪输入）：")
    for inp in onnx_inputs:
        shape, dyn = _shape_from_value_info(inp, args.batch)
        print(f" - {inp.name}: shape={shape}, dynamic_dim={dyn}")
        input_info.append((inp.name, shape))

    input_info = _align_inputs_to_signature(args.saved_model_dir, input_info, tf)
    print("校准数据张量顺序（与 SavedModel signature 对齐）：")
    for name, shape in input_info:
        print(f" - {name}: {shape}")

    def rep_ds():
        yield from representative_dataset_gen(
            args.images,
            input_info,
            args.num_calib,
            args.mean,
            args.std,
            cv2,
            np,
        )

    print(f"\n[3/4] SavedModel -> TFLite INT8: {args.output} ...", flush=True)
    converter = tf.lite.TFLiteConverter.from_saved_model(args.saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_ds
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    with open(args.output, "wb") as f:
        f.write(tflite_model)

    print(f"\n[4/4] TFLite INT8 已保存: {args.output}")
    print("完成。")


if __name__ == "__main__":
    main()
