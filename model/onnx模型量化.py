#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX 模型精度检测与转换工具
自动识别模型精度并转换为 INT8 或 FP16；可选使用 ONNX Runtime 加载并试跑验证。

部署说明（尤其移动端 QNN/HTP）：
- 常见体感「像没走 QNN」多因：不是完全没走 QNN，而是只走了一部分——Profiling 里 QNN 与 CPU
  节点数并存（例如约三成 QNN、七成 CPU）时，加速只作用在子图上，整体帧率仍可能很低。
- INT8 / MIXED(INT8+FLOAT) 图中 Quant、Conv_quant、Mul 后 QuantizeLinear 等常被划给 CPU；
  输入为 float 时图首端量化类算子也常落 CPU。问题本质是分区占比，不是「EP 完全没挂上」。
- 离线统计 ONNX 图中某类节点个数，与运行时 profiling 里融合/内核名（如 *_kernel_time）
  口径不一致；以设备上 ORT profiling 的 provider 占比为准。
- 本脚本在 PC 上的验证不能代表手机 QNN 分区；「全节点走 QNN」无法由本工具单独保证，
  取决于 QNN EP 算子支持、图结构与量化路径；需结合 Qualcomm 文档与工具链提高 QNN 覆盖占比。
- Android 端使用 onnxruntime-android / onnxruntime-android-qnn 时，建议 -t int8_io 或 int8_qnn，
  并加 --qnn-int8 使用 QInt8 激活与权重量化（与包内 QNN EP 整型路径一致）；折叠图边界类型见 --qnn-int8 说明。
"""

import sys
import os
import argparse
import struct
import inspect
from typing import Dict, List, Tuple, Optional

try:
    import onnx
    import numpy as np
    from onnx import numpy_helper
except ImportError:
    print("错误：请先安装依赖库")
    print("pip install onnx numpy")
    sys.exit(1)

# ONNX Runtime：推理会话（InferenceSession）与量化（quantize_dynamic）共用同一包
try:
    import onnxruntime as ort
    ONNXRUNTIME_INFERENCE_AVAILABLE = True
except ImportError:
    ort = None  # type: ignore
    ONNXRUNTIME_INFERENCE_AVAILABLE = False

try:
    from onnxruntime.quantization import (
        quantize_dynamic,
        quantize,
        QuantType,
        quantize_static,
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
    )
    ONNXRUNTIME_QUANT_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_QUANT_AVAILABLE = False
    quantize = None  # type: ignore
    print("警告：未安装 onnxruntime 或未包含 quantization 模块，无法执行 INT8 量化")

try:
    from onnxruntime.quantization.execution_providers.qnn.quant_config import get_qnn_qdq_config
    QNN_QDQ_CONFIG_AVAILABLE = True
except ImportError:
    get_qnn_qdq_config = None  # type: ignore
    QNN_QDQ_CONFIG_AVAILABLE = False

try:
    from onnxconverter_common import float16
    ONNXCONVERTER_AVAILABLE = True
except ImportError:
    ONNXCONVERTER_AVAILABLE = False
    print("警告：未安装 onnxconverter-common，无法执行 FP16 转换")


class ONNXPrecisionChecker:
    """检测 ONNX 模型精度信息"""

    DTYPE_MAP = {
        1: ("FLOAT32", "FP32", 4),
        2: ("UINT8", "UINT8", 1),
        3: ("INT8", "INT8", 1),
        4: ("UINT16", "UINT16", 2),
        5: ("INT16", "INT16", 2),
        6: ("INT32", "INT32", 4),
        7: ("INT64", "INT64", 8),
        10: ("FLOAT16", "FP16", 2),
        11: ("DOUBLE", "FP64", 8),
        12: ("UINT32", "UINT32", 4),
        13: ("UINT64", "UINT64", 8),
    }

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = onnx.load(model_path)
        self.initializers_info = []
        self.inputs_info = []
        self.outputs_info = []
        self.weight_stats = {}

    def get_dtype_info(self, tensor_type: int) -> Tuple[str, str, int]:
        """获取数据类型信息"""
        return self.DTYPE_MAP.get(tensor_type, (f"UNKNOWN({tensor_type})", "UNKNOWN", 0))

    def analyze_initializers(self) -> Dict:
        """分析权重（initializers）的精度分布"""
        dtype_counts = {}
        total_params = 0
        total_bytes = 0

        for init in self.model.graph.initializer:
            tensor = numpy_helper.to_array(init)
            dtype = tensor.dtype
            dtype_name = str(dtype)
            num_elements = tensor.size
            bytes_per_elem = tensor.itemsize

            if dtype_name not in dtype_counts:
                dtype_counts[dtype_name] = {
                    "count": 0,
                    "elements": 0,
                    "bytes": 0,
                    "shapes": []
                }

            dtype_counts[dtype_name]["count"] += 1
            dtype_counts[dtype_name]["elements"] += num_elements
            dtype_counts[dtype_name]["bytes"] += num_elements * bytes_per_elem
            dtype_counts[dtype_name]["shapes"].append(tensor.shape)

            total_params += num_elements
            total_bytes += num_elements * bytes_per_elem

            self.initializers_info.append({
                "name": init.name,
                "dtype": dtype_name,
                "shape": tensor.shape,
                "elements": num_elements,
                "bytes": num_elements * bytes_per_elem
            })

        self.weight_stats = {
            "total_initializers": len(self.model.graph.initializer),
            "total_parameters": total_params,
            "total_bytes": total_bytes,
            "dtype_distribution": dtype_counts
        }

        return self.weight_stats

    def analyze_io(self):
        """分析输入输出精度"""
        for inp in self.model.graph.input:
            tensor_type = inp.type.tensor_type
            dtype_info = self.get_dtype_info(tensor_type.elem_type)
            shape = [dim.dim_value if dim.dim_value else dim.dim_param 
                     for dim in tensor_type.shape.dim]
            self.inputs_info.append({
                "name": inp.name,
                "dtype": dtype_info[0],
                "shape": shape
            })

        for out in self.model.graph.output:
            tensor_type = out.type.tensor_type
            dtype_info = self.get_dtype_info(tensor_type.elem_type)
            shape = [dim.dim_value if dim.dim_value else dim.dim_param 
                     for dim in tensor_type.shape.dim]
            self.outputs_info.append({
                "name": out.name,
                "dtype": dtype_info[0],
                "shape": shape
            })

    def determine_model_precision(self) -> str:
        """判断模型整体精度（支持混合精度）"""
        summary = self.get_precision_summary()
        return summary["precision_label"]

    def get_precision_summary(self) -> Dict:
        """返回更准确的精度摘要（含字节占比与 IO 类型）"""
        if not self.weight_stats:
            self.analyze_initializers()
        if not self.inputs_info or not self.outputs_info:
            self.analyze_io()

        dtype_dist = self.weight_stats.get("dtype_distribution", {})
        total_bytes = float(self.weight_stats.get("total_bytes", 0))

        fp16_bytes = 0
        fp32_bytes = 0
        int8_bytes = 0

        for dtype_name, info in dtype_dist.items():
            bytes_count = info.get("bytes", 0)
            dtype_lower = dtype_name.lower()
            if "float16" in dtype_lower:
                fp16_bytes += bytes_count
            elif "float32" in dtype_lower:
                fp32_bytes += bytes_count
            elif dtype_lower in ("int8", "uint8"):
                int8_bytes += bytes_count

        fp16_ratio = (fp16_bytes / total_bytes * 100.0) if total_bytes > 0 else 0.0
        fp32_ratio = (fp32_bytes / total_bytes * 100.0) if total_bytes > 0 else 0.0
        int8_ratio = (int8_bytes / total_bytes * 100.0) if total_bytes > 0 else 0.0

        has_fp16 = fp16_bytes > 0
        has_fp32 = fp32_bytes > 0
        has_int8 = int8_bytes > 0

        if has_fp16 and not has_fp32 and not has_int8:
            precision_label = "FP16"
        elif has_fp32 and not has_fp16 and not has_int8:
            precision_label = "FP32"
        elif has_int8 and not has_fp16 and not has_fp32:
            precision_label = "INT8"
        elif has_fp16 and has_fp32 and not has_int8:
            precision_label = "MIXED(FP16+FP32)"
        elif has_int8 and (has_fp16 or has_fp32):
            precision_label = "MIXED(INT8+FLOAT)"
        else:
            precision_label = "UNKNOWN"

        input_dtypes = [inp["dtype"] for inp in self.inputs_info]
        output_dtypes = [out["dtype"] for out in self.outputs_info]

        return {
            "precision_label": precision_label,
            "total_bytes": total_bytes,
            "fp16_bytes": fp16_bytes,
            "fp32_bytes": fp32_bytes,
            "int8_bytes": int8_bytes,
            "fp16_ratio": fp16_ratio,
            "fp32_ratio": fp32_ratio,
            "int8_ratio": int8_ratio,
            "input_dtypes": input_dtypes,
            "output_dtypes": output_dtypes
        }

    def print_report(self):
        """打印详细报告"""
        print("=" * 60)
        print(f"ONNX 模型分析报告: {os.path.basename(self.model_path)}")
        print("=" * 60)

        # 模型基本信息
        print(f"\n模型版本: {self.model.ir_version}")
        print(f"Opset 版本: {self.model.opset_import[0].version if self.model.opset_import else 'N/A'}")
        print(f"生产者: {self.model.producer_name} {self.model.producer_version}")

        # 精度分析
        if not self.weight_stats:
            self.analyze_initializers()
        self.analyze_io()

        summary = self.get_precision_summary()
        current_precision = summary["precision_label"]
        print(f"\n🔍 当前模型精度: {current_precision}")
        print(f"   - FP16 字节占比: {summary['fp16_ratio']:.2f}%")
        print(f"   - FP32 字节占比: {summary['fp32_ratio']:.2f}%")
        print(f"   - INT8 字节占比: {summary['int8_ratio']:.2f}%")
        print(f"   - 输入类型: {', '.join(summary['input_dtypes']) if summary['input_dtypes'] else 'N/A'}")
        print(f"   - 输出类型: {', '.join(summary['output_dtypes']) if summary['output_dtypes'] else 'N/A'}")

        # 权重分布
        print(f"\n📊 权重分布:")
        print(f"  总权重数量: {self.weight_stats['total_initializers']}")
        print(f"  总参数量: {self.weight_stats['total_parameters']:,}")
        print(f"  总大小: {self.weight_stats['total_bytes'] / 1024 / 1024:.2f} MB")

        print(f"\n  数据类型分布:")
        for dtype, info in self.weight_stats['dtype_distribution'].items():
            percentage = (info['bytes'] / self.weight_stats['total_bytes']) * 100
            print(f"    - {dtype:12s}: {info['count']:3d} 个张量, "
                  f"{info['elements']:12,} 参数, "
                  f"{info['bytes']/1024/1024:6.2f} MB ({percentage:5.1f}%)")

        # 输入信息
        print(f"\n📥 模型输入:")
        for inp in self.inputs_info:
            print(f"  - {inp['name']}: {inp['dtype']} {inp['shape']}")

        # 输出信息
        print(f"\n📤 模型输出:")
        for out in self.outputs_info:
            print(f"  - {out['name']}: {out['dtype']} {out['shape']}")

        print("=" * 60)

        return current_precision


class ONNXConverter:
    """ONNX 模型转换器"""

    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path
        self.checker = ONNXPrecisionChecker(input_path)

    def to_fp16(self, keep_io_types: bool = True) -> bool:
        """转换为 FP16"""
        if not ONNXCONVERTER_AVAILABLE:
            print("错误：未安装 onnxconverter-common")
            print("pip install onnxconverter-common")
            return False

        print(f"\n🔄 正在转换为 FP16...")
        print(f"   保持输入输出类型: {keep_io_types}")

        try:
            model = onnx.load(self.input_path)
            model_fp16 = float16.convert_float_to_float16(
                model, 
                keep_io_types=keep_io_types,
                disable_shape_infer=False
            )
            onnx.save(model_fp16, self.output_path)

            # 验证
            new_checker = ONNXPrecisionChecker(self.output_path)
            new_checker.analyze_initializers()
            new_checker.analyze_io()
            summary = new_checker.get_precision_summary()
            new_precision = summary["precision_label"]

            orig_size = os.path.getsize(self.input_path) / 1024 / 1024
            new_size = os.path.getsize(self.output_path) / 1024 / 1024

            print(f"✅ 转换完成: {self.output_path}")
            print(f"   原始大小: {orig_size:.2f} MB")
            print(f"   转换后大小: {new_size:.2f} MB")
            print(f"   压缩率: {new_size/orig_size*100:.1f}%")
            print(f"   确认精度: {new_precision}")
            print(f"   FP16 字节占比: {summary['fp16_ratio']:.2f}%")
            print(f"   FP32 字节占比: {summary['fp32_ratio']:.2f}%")
            print(f"   输入类型: {', '.join(summary['input_dtypes']) if summary['input_dtypes'] else 'N/A'}")
            print(f"   输出类型: {', '.join(summary['output_dtypes']) if summary['output_dtypes'] else 'N/A'}")

            return True

        except Exception as e:
            print(f"❌ 转换失败: {e}")
            return False

    def to_int8_dynamic(self) -> bool:
        """动态量化转换为 INT8"""
        if not ONNXRUNTIME_QUANT_AVAILABLE:
            print("错误：未安装 onnxruntime")
            print("pip install onnxruntime")
            return False

        print(f"\n🔄 正在进行动态 INT8 量化...")
        print(
            "⚠️ 若你需要「图输入/输出在 ONNX 里声明为整型」以便走整型推理路径：不要用本模式。"
            "请改用 -t int8_io 或 -t int8_qnn 并加 --uint8-io（静态 QDQ + 折叠边界）。"
        )

        try:
            # 传入路径时 ORT 会在模型同目录生成「原名-inferred.onnx」做 shape inference，
            # 在 Windows + 非 ASCII 路径下 onnx.shape_inference.infer_shapes_path 常失败。
            # 先 onnx.load 再传入 ModelProto，会走临时目录流程，避免该问题。
            model_proto = onnx.load(self.input_path)

            qd_signature = inspect.signature(quantize_dynamic)
            kwargs = {
                "model_input": model_proto,
                "model_output": self.output_path,
                "weight_type": QuantType.QInt8,
            }
            if "optimize_model" in qd_signature.parameters:
                kwargs["optimize_model"] = True

            try:
                quantize_dynamic(**kwargs)
            except TypeError as te:
                if "optimize_model" in kwargs:
                    kwargs.pop("optimize_model", None)
                    quantize_dynamic(**kwargs)
                else:
                    raise te

            # 验证
            new_checker = ONNXPrecisionChecker(self.output_path)
            new_checker.analyze_initializers()
            new_checker.analyze_io()
            summary = new_checker.get_precision_summary()
            new_precision = summary["precision_label"]

            orig_size = os.path.getsize(self.input_path) / 1024 / 1024
            new_size = os.path.getsize(self.output_path) / 1024 / 1024

            print(f"✅ 量化完成: {self.output_path}")
            print(f"   原始大小: {orig_size:.2f} MB")
            print(f"   量化后大小: {new_size:.2f} MB")
            print(f"   压缩率: {new_size/orig_size*100:.1f}%")
            print(f"   确认精度: {new_precision}")
            print(f"   INT8 字节占比: {summary['int8_ratio']:.2f}%")
            print(f"   FP32 字节占比: {summary['fp32_ratio']:.2f}%")
            print(f"   输入类型: {', '.join(summary['input_dtypes']) if summary['input_dtypes'] else 'N/A'}")
            print(f"   输出类型: {', '.join(summary['output_dtypes']) if summary['output_dtypes'] else 'N/A'}")
            print(
                "   说明：图接口为 FLOAT32 不等于「整条推理都在 FP32 上跑」——权重已大量 INT8，"
                "图内仍有整型计算；只是 ORT 动态量化默认把 graph.input/output 声明成 float，便于兼容现有代码。"
            )
            print(
                "   若业务要求 ONNX 边界上必须是 tensor(uint8/int8)：动态量化通常做不到；"
                "请使用 -t int8_qnn（静态 QDQ）并加 --uint8-io 尝试折叠首端 QuantizeLinear，"
                "或使用 Qualcomm QNN 官方工具链导出。"
            )

            _print_int8_deployment_notes(self.output_path, summary)

            return True

        except Exception as e:
            print(f"❌ 量化失败: {e}")
            return False

    def to_int8_qnn_static(
        self,
        calibration_data_path: Optional[str] = None,
        random_batches: int = 16,
        qnn_int8: bool = False,
    ) -> bool:
        """静态 QDQ（get_qnn_qdq_config），面向 QNN EP / onnxruntime-android-qnn；可选 QInt8 整型路径。"""
        if not ONNXRUNTIME_QUANT_AVAILABLE or quantize is None:
            print("错误：未安装 onnxruntime")
            return False
        if not QNN_QDQ_CONFIG_AVAILABLE or get_qnn_qdq_config is None:
            print("错误：当前 onnxruntime 不包含 execution_providers.qnn.quant_config.get_qnn_qdq_config")
            print("请升级 onnxruntime 或使用 -t int8 动态量化")
            return False

        print("\n🔄 正在进行静态 INT8 量化（QNN QDQ 配置，quantize_static）...")
        if qnn_int8:
            print("   --qnn-int8：激活与权重均使用 QInt8（对称整型，便于 Android onnxruntime-android-qnn）")

        try:
            model_proto = onnx.load(self.input_path)
            if calibration_data_path and os.path.isfile(calibration_data_path):
                reader: CalibrationDataReader = _NpyCalibrationReader(self.input_path, calibration_data_path)
                print(f"   校准数据: {calibration_data_path}")
            else:
                reader = _RandomCalibrationReader(self.input_path, random_batches)
                print(f"   未提供有效 .npy 校准文件，使用 {random_batches} 批随机数据（精度可能明显变差）")

            act_t = QuantType.QInt8 if qnn_int8 else QuantType.QUInt8
            w_t = QuantType.QInt8 if qnn_int8 else QuantType.QUInt8
            cfg = get_qnn_qdq_config(
                model_proto,
                reader,
                calibrate_method=CalibrationMethod.MinMax,
                activation_type=act_t,
                weight_type=w_t,
            )
            quantize(model_proto, self.output_path, cfg)

            new_checker = ONNXPrecisionChecker(self.output_path)
            new_checker.analyze_initializers()
            new_checker.analyze_io()
            summary = new_checker.get_precision_summary()

            orig_size = os.path.getsize(self.input_path) / 1024 / 1024
            new_size = os.path.getsize(self.output_path) / 1024 / 1024
            print(f"✅ QNN QDQ 量化完成: {self.output_path}")
            print(f"   原始大小: {orig_size:.2f} MB")
            print(f"   量化后大小: {new_size:.2f} MB")
            print(f"   确认精度: {summary['precision_label']}")
            print(f"   输入类型: {', '.join(summary['input_dtypes']) if summary['input_dtypes'] else 'N/A'}")
            print(f"   输出类型: {', '.join(summary['output_dtypes']) if summary['output_dtypes'] else 'N/A'}")
            print("   提示：静态 QDQ 量化后接口仍常为 float；-t int8_io 或 --uint8-io 会折叠图边界；加 --qnn-int8 时边界折叠为 int8。")

            _print_int8_deployment_notes(self.output_path, summary)
            return True

        except Exception as e:
            print(f"❌ QNN QDQ 量化失败: {e}")
            return False

    def to_int8_static(self, calibration_data_path: Optional[str] = None) -> bool:
        """静态量化转换为 INT8（需要校准数据）"""
        if not ONNXRUNTIME_QUANT_AVAILABLE:
            print("错误：未安装 onnxruntime")
            return False

        print(f"\n🔄 正在进行静态 INT8 量化...")
        print(f"   注意：静态量化需要校准数据以获得最佳精度")

        if calibration_data_path is None:
            print("   未提供校准数据，将使用随机数据（精度可能不佳）")
            # 这里可以实现随机数据生成逻辑

        try:
            # 静态量化实现较复杂，这里先提示用户
            print("   静态量化需要实现 CalibrationDataReader")
            print("   建议使用动态量化，或参考 ONNX Runtime 文档实现静态量化")
            return False

        except Exception as e:
            print(f"❌ 量化失败: {e}")
            return False


def _shape_list_for_calib(shape) -> List[int]:
    out: List[int] = []
    for d in shape:
        if d is None or isinstance(d, str):
            out.append(1)
        elif isinstance(d, int) and d < 0:
            out.append(1)
        else:
            out.append(int(d))
    return out


class _RandomCalibrationReader(CalibrationDataReader):
    """随机校准数据（仅用于无 .npy 时的占位，精度无保证）。"""

    def __init__(self, model_path: str, num_batches: int = 16):
        if not ONNXRUNTIME_INFERENCE_AVAILABLE or ort is None:
            raise RuntimeError("需要 onnxruntime 以构建校准用 InferenceSession")
        self._num_batches = num_batches
        self._idx = 0
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self._session = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
        self._feeds: List[Dict[str, np.ndarray]] = []
        for _ in range(num_batches):
            feed: Dict[str, np.ndarray] = {}
            for inp in self._session.get_inputs():
                shape = _shape_list_for_calib(inp.shape)
                feed[inp.name] = np.random.randn(*shape).astype(np.float32)
            self._feeds.append(feed)

    def get_next(self) -> Optional[dict]:
        if self._idx >= len(self._feeds):
            return None
        r = self._feeds[self._idx]
        self._idx += 1
        return r

    def __len__(self) -> int:
        return self._num_batches


class _NpyCalibrationReader(CalibrationDataReader):
    """从单个 .npy 读取 NCHW 批次 (N,H,W,C) 或 (N,C,H,W) —— 按首维 N 逐批喂入第一个图输入名。"""

    def __init__(self, model_path: str, npy_path: str, input_name: Optional[str] = None):
        arr = np.load(npy_path)
        if arr.ndim < 2:
            raise ValueError("校准 .npy 至少二维 [N, ...]")
        self._arr = np.asarray(arr, dtype=np.float32)
        self._n = self._arr.shape[0]
        self._idx = 0
        m = onnx.load(model_path)
        self._input_name = input_name or m.graph.input[0].name

    def get_next(self) -> Optional[dict]:
        if self._idx >= self._n:
            return None
        batch = self._arr[self._idx : self._idx + 1]
        self._idx += 1
        return {self._input_name: batch}

    def __len__(self) -> int:
        return self._n


def _tensor_rename_global(g: onnx.GraphProto, old: str, new: str) -> None:
    """将图中张量名 old 全部替换为 new（节点、initializer、value_info、input/output）。"""
    if old == new:
        return
    for node in g.node:
        for i, name in enumerate(node.input):
            if name == old:
                node.input[i] = new
        for i, name in enumerate(node.output):
            if name == old:
                node.output[i] = new
    for vi in g.value_info:
        if vi.name == old:
            vi.name = new
    for vi in g.input:
        if vi.name == old:
            vi.name = new
    for vi in g.output:
        if vi.name == old:
            vi.name = new
    for init in g.initializer:
        if init.name == old:
            init.name = new


def apply_input_fold_uint8(
    model: onnx.ModelProto,
    io_elem_type: int,
) -> Tuple[bool, str]:
    """去掉「图输入 → QuantizeLinear」，将 graph.input 的 elem_type 设为 io_elem_type（UINT8 或 INT8）。"""
    g = model.graph
    input_names = {i.name for i in g.input}
    repl: Dict[str, str] = {}
    to_remove: List[onnx.NodeProto] = []

    for node in g.node:
        if node.op_type != "QuantizeLinear":
            continue
        if not node.input or node.input[0] not in input_names:
            continue
        if not node.output:
            continue
        repl[node.output[0]] = node.input[0]
        to_remove.append(node)

    if not to_remove:
        return (
            False,
            "[输入] 未找到「图输入→QuantizeLinear」（动态量化常无此显式节点）。请用 -t int8_qnn 或 -t int8_io 生成 QDQ。",
        )

    remove_ids = {id(n) for n in to_remove}
    for node in g.node:
        if id(node) in remove_ids:
            continue
        for i, name in enumerate(node.input):
            if name in repl:
                node.input[i] = repl[name]

    kept = [n for n in g.node if id(n) not in remove_ids]
    g.ClearField("node")
    g.node.extend(kept)

    repl_out_names = set(repl.keys())
    kept_vis = [v for v in g.value_info if v.name not in repl_out_names]
    g.ClearField("value_info")
    g.value_info.extend(kept_vis)

    for vi in g.input:
        if vi.name in set(repl.values()) and vi.type.HasField("tensor_type"):
            vi.type.tensor_type.elem_type = io_elem_type

    from onnx import TensorProto

    tname = "int8" if io_elem_type == TensorProto.INT8 else "uint8"
    return True, f"[输入] 已改为 tensor({tname})"


def apply_output_fold_uint8(
    model: onnx.ModelProto,
    io_elem_type: int,
) -> Tuple[bool, str]:
    """去掉末级「DequantizeLinear → graph.output」，把输出边界的 elem_type 设为 io_elem_type。"""
    g = model.graph
    output_names = {o.name for o in g.output}
    to_remove: List[Tuple[onnx.NodeProto, str, str]] = []

    for node in g.node:
        if node.op_type != "DequantizeLinear":
            continue
        if not node.output or node.output[0] not in output_names:
            continue
        if not node.input:
            continue
        to_remove.append((node, node.output[0], node.input[0]))

    if not to_remove:
        return (
            False,
            "[输出] 未找到「DequantizeLinear→图输出」。可能末级未量化为 QDQ，或需手动改导出。",
        )

    remove_ids = {id(n) for n, _, _ in to_remove}
    kept = [n for n in g.node if id(n) not in remove_ids]
    g.ClearField("node")
    g.node.extend(kept)

    out_names_done: List[str] = []
    for _, out_name, u8_in in to_remove:
        if u8_in != out_name:
            _tensor_rename_global(g, u8_in, out_name)
        out_names_done.append(out_name)

    for vi in g.output:
        if vi.name in out_names_done and vi.type.HasField("tensor_type"):
            vi.type.tensor_type.elem_type = io_elem_type

    from onnx import TensorProto

    tname = "int8" if io_elem_type == TensorProto.INT8 else "uint8"
    return True, f"[输出] 已改为 tensor({tname})"


def fold_qdq_graph_io_uint8(
    model_path: str,
    output_path: str,
    *,
    io_elem_type: Optional[int] = None,
) -> Tuple[bool, str]:
    """折叠 Q/DQ 边界。io_elem_type：TensorProto.UINT8（默认）或 INT8（与 --qnn-int8 的 QInt8 路径一致）。"""
    from onnx import TensorProto

    if io_elem_type is None:
        io_elem_type = TensorProto.UINT8

    try:
        model = onnx.load(model_path)
    except Exception as e:
        return False, f"无法加载模型: {e}"

    ok_in, msg_in = apply_input_fold_uint8(model, io_elem_type)
    ok_out, msg_out = apply_output_fold_uint8(model, io_elem_type)

    onnx.save(model, output_path)
    try:
        onnx.checker.check_model(model)
        chk = ""
    except Exception as ce:
        chk = f"；onnx.checker: {ce}"

    if not ok_in and not ok_out:
        return False, f"{msg_in}\n{msg_out}"

    parts = [p for p in (msg_in, msg_out) if p.startswith("[") and "已" in p]
    return True, "✅ " + "；".join(parts) + f" → {output_path}{chk}"


def fold_qdq_graph_input_uint8(model_path: str, output_path: str) -> Tuple[bool, str]:
    """兼容旧接口：仅折叠输入（默认 UINT8 边界）。"""
    from onnx import TensorProto

    try:
        model = onnx.load(model_path)
    except Exception as e:
        return False, f"无法加载模型: {e}"
    ok, msg = apply_input_fold_uint8(model, TensorProto.UINT8)
    if not ok:
        return False, msg
    onnx.save(model, output_path)
    try:
        onnx.checker.check_model(model)
    except Exception as ce:
        return True, f"✅ 已保存 {output_path}；onnx.checker 提示: {ce}"
    return True, f"✅ {msg} {output_path}"


def _count_quant_related_nodes(model_path: str) -> Tuple[int, int, int]:
    """统计 QuantizeLinear / DequantizeLinear / QLinear* 等节点数量（用于提示 CPU 回退风险）。"""
    try:
        m = onnx.load(model_path)
    except Exception:
        return -1, -1, -1
    q = sum(1 for n in m.graph.node if n.op_type == "QuantizeLinear")
    dq = sum(1 for n in m.graph.node if n.op_type == "DequantizeLinear")
    qlinear = sum(1 for n in m.graph.node if n.op_type.startswith("QLinear"))
    return q, dq, qlinear


def _parse_ort_providers(s: Optional[str]) -> Optional[List[str]]:
    if not s or not str(s).strip():
        return None
    return [p.strip() for p in str(s).split(",") if p.strip()]


def _print_int8_deployment_notes(output_path: str, summary: Dict) -> None:
    """INT8 动态量化完成后：说明 QNN 部分覆盖、Profiling 口径、与「全图走 QNN」的可行边界。"""
    q, dq, ql = _count_quant_related_nodes(output_path)
    prec = summary.get("precision_label", "")

    print("\n" + "=" * 60)
    print("📌 INT8 部署与性能（QNN / HTP / CPU 分区）")
    print("=" * 60)
    if "MIXED" in prec or (summary.get("fp32_ratio") or 0) > 0.01:
        print("• 当前为 MIXED(INT8+FLOAT) 或仍含少量 FP32：混精与 Quant/Dequant 更易导致大块子图留在 CPU。")
    if q >= 0:
        print(f"• 离线粗计 ONNX 图：QuantizeLinear={q}，DequantizeLinear={dq}，QLinear*={ql}。")
        print("  若与手机上 profiling 名称不一致（如 Conv_quant、…QuantizeLinear_kernel_time）不矛盾：")
        print("  离线是图节点类型统计，运行时是融合/内核口径——以设备上 provider 占比为准。")
    print("")
    print("=" * 60)


def _ort_type_to_numpy(ort_type: str):
    mapping = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
        "tensor(int32)": np.int32,
        "tensor(int64)": np.int64,
        "tensor(uint8)": np.uint8,
        "tensor(int8)": np.int8,
        "tensor(uint16)": np.uint16,
        "tensor(bool)": np.bool_,
    }
    return mapping.get(ort_type, np.float32)


def _concrete_shape(shape: List) -> List[int]:
    out: List[int] = []
    for d in shape:
        if d is None or isinstance(d, str):
            out.append(1)
        elif isinstance(d, int) and d < 0:
            out.append(1)
        else:
            out.append(int(d))
    return out


def _dummy_feed_for_input(inp) -> np.ndarray:
    """为 InferenceSession 构造单条随机输入。"""
    shape = _concrete_shape(list(inp.shape))
    dtype = _ort_type_to_numpy(inp.type)
    if np.issubdtype(dtype, np.integer):
        if dtype == np.int8:
            return np.random.randint(-128, 128, size=shape, dtype=dtype)
        return np.random.randint(0, 256, size=shape, dtype=dtype)
    if dtype == np.bool_:
        return np.random.randint(0, 2, size=shape).astype(np.bool_)
    return np.random.randn(*shape).astype(dtype)


def verify_onnxruntime_session(
    model_path: str,
    run_inference: bool = True,
    *,
    intra_op_num_threads: int = 0,
    providers: Optional[List[str]] = None,
    enable_profiling: bool = False,
) -> bool:
    """使用 ONNX Runtime 加载模型，可选试跑一次推理，用于部署前自检。

    intra_op_num_threads: >0 时设置 SessionOptions.intra_op_num_threads（仅影响落在 CPU 的算子段）。
    providers: 非空时按该顺序使用执行提供程序（如移动端 QNN）；None 时使用本机默认可用列表。
    enable_profiling: 为 True 时在试跑后调用 end_profiling()，便于查看 node-summary 与各 EP 节点占比。
    """
    if not ONNXRUNTIME_INFERENCE_AVAILABLE or ort is None:
        print("错误：未安装 onnxruntime，无法进行 ONNX Runtime 验证")
        print("pip install onnxruntime")
        return False

    print(f"\n🔧 ONNX Runtime 验证: {model_path}")
    profiling_effective = bool(enable_profiling and run_inference)
    if enable_profiling and not run_inference:
        print("   ⚠️ --ort-profiling 需配合试跑；当前已跳过 profiling（请去掉 --no-run-ort）")

    try:
        so = ort.SessionOptions()
        so.log_severity_level = 3
        if intra_op_num_threads and intra_op_num_threads > 0:
            so.intra_op_num_threads = intra_op_num_threads
            print(f"   SessionOptions.intra_op_num_threads = {intra_op_num_threads}")
        if profiling_effective:
            so.enable_profiling = True
            print("   SessionOptions.enable_profiling = True（试跑后将生成 profiling 数据）")
        use_providers = providers if providers else ort.get_available_providers()
        sess = ort.InferenceSession(model_path, so, providers=use_providers)
    except Exception as e:
        print(f"❌ InferenceSession 加载失败: {e}")
        return False

    print(f"   onnxruntime 版本: {ort.__version__}")
    print(f"   执行提供程序: {sess.get_providers()}")
    for inp in sess.get_inputs():
        print(f"   输入: {inp.name}  {inp.type}  shape={inp.shape}")
    for out in sess.get_outputs():
        print(f"   输出: {out.name}  {out.type}  shape={out.shape}")

    if not run_inference:
        print("   （已跳过试跑推理，见 --no-run-ort；Profiling 需至少一次 run 才有数据）")
        print("✅ ONNX Runtime 可正常加载该模型")
        return True

    try:
        feed = {inp.name: _dummy_feed_for_input(inp) for inp in sess.get_inputs()}
        outputs = sess.run(None, feed)
    except Exception as e:
        print(f"⚠️ 试跑推理失败（可能与动态维度或类型有关）: {e}")
        print("   可尝试加 --no-run-ort 仅检查加载，或在业务代码中传入真实输入。")
        return False

    if profiling_effective:
        try:
            prof_path = sess.end_profiling()
            print(f"   ORT profiling 输出: {prof_path}")
            print("   可用该文件结合文档分析 node-summary 与各 ExecutionProvider 的节点占比（如 QNN vs CPU）。")
        except Exception as pe:
            print(f"   ⚠️ end_profiling 失败: {pe}")

    print(f"   试跑成功: 共 {len(outputs)} 个输出")
    for i, o in enumerate(outputs):
        print(f"     output[{i}] shape={o.shape} dtype={o.dtype}")
    print("✅ ONNX Runtime 加载与试跑通过")
    return True


def interactive_configure_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> argparse.Namespace:
    """无模型路径时在终端交互补全参数；默认量化目标 int8_io。若命令行已带 -t/-o 等则保留，只问缺的项。"""
    choices = {"fp16", "int8", "int8_qnn", "int8_io", "int8_static"}
    print("\n" + "=" * 52)
    print("交互模式（直接回车使用方括号内默认值）")
    print("=" * 52 + "\n")

    while True:
        raw = input("请输入 ONNX 模型路径: ").strip()
        path = raw.strip('"').strip("'")
        if not path:
            print("路径不能为空。")
            continue
        if not os.path.isfile(path):
            print(f"找不到文件: {path}")
            continue
        break
    args.model = path

    if args.target is None:
        print("\n可选: fp16 | int8 | int8_qnn | int8_io | int8_static")
        print("      输入 detect 表示只检测精度、不转换")
        t_in = input("量化目标 [int8_io]: ").strip().lower()
        if not t_in:
            t_in = "int8_io"
        if t_in in ("detect", "检测", "none", "0"):
            args.target = None
        elif t_in in choices:
            args.target = t_in
        else:
            print(f"未识别「{t_in}」，使用默认 int8_io。")
            args.target = "int8_io"
    else:
        print(f"\n量化目标（来自命令行）: {args.target}")

    if args.target in ("int8_qnn", "int8_io"):
        if not args.calibration_data:
            cal = input("校准数据 .npy（可空则随机校准，精度较差）: ").strip().strip('"').strip("'")
            args.calibration_data = cal if cal else None
        rb = input("随机校准批次数 [16]: ").strip()
        if rb.isdigit():
            args.calibrate_random_batches = max(1, int(rb))
        if not args.qnn_int8:
            qn = input("Android QNN 使用 --qnn-int8 [Y/n]: ").strip().lower()
            if not qn:
                args.qnn_int8 = True
            else:
                args.qnn_int8 = qn not in ("n", "no", "否")

    if args.target == "fp16" and not args.keep_io_fp32:
        kio = input("FP16 是否保持输入输出为 FP32 (--keep-io-fp32) [Y/n]: ").strip().lower()
        args.keep_io_fp32 = kio not in ("n", "no", "否")

    if not args.output:
        out_in = input("输出 .onnx 路径（回车则按目标自动生成）: ").strip().strip('"').strip("'")
        args.output = out_in if out_in else None

    if not args.skip_verify_ort:
        sk = input("转换后跳过 ORT 验证 (--skip-verify-ort) [y/N]: ").strip().lower()
        args.skip_verify_ort = sk in ("y", "yes", "是", "1")

    if args.target:
        if args.output:
            preview_out = args.output
        else:
            base, ext = os.path.splitext(args.model)
            preview_out = f"{base}_{args.target}{ext}"
    else:
        preview_out = "—"

    print("\n" + "-" * 52)
    print("请确认:")
    print(f"  模型:     {args.model}")
    print(f"  量化目标: {args.target or 'detect（仅检测）'}")
    print(f"  将保存为: {preview_out}")
    if args.target in ("int8_qnn", "int8_io"):
        print(f"  QNN Int8: {args.qnn_int8}")
    if args.target == "fp16":
        print(f"  FP16 保持 IO FP32: {args.keep_io_fp32}")
    print(f"  跳过验证: {args.skip_verify_ort}")
    print("-" * 52)
    ok = input("确认执行? [Y/n]: ").strip().lower()
    if ok in ("n", "no", "否"):
        print("已取消。")
        sys.exit(0)

    return args


def main():
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="ONNX 模型精度检测与转换工具（支持 ONNX Runtime 量化与 InferenceSession 验证）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 仅检测模型精度
  python onnx模型量化.py best.onnx

  # 用 ONNX Runtime 验证模型可加载（可选试跑）
  python onnx模型量化.py best.onnx --verify-ort

  # 检测并转换为 FP16（成功后自动用 ORT 验证输出；可加 --skip-verify-ort 跳过）
  python onnx模型量化.py best.onnx -t fp16 -o best_fp16.onnx

  # Android onnxruntime-android-qnn：QInt8 量化 + 图边界 int8（推荐加真实 calib.npy）
  python onnx模型量化.py best.onnx -t int8_io -o out.onnx --calibration-data calib.npy --qnn-int8

  # 无参数：进入交互模式（手动输入模型路径，默认量化目标 int8_io）
  python onnx模型量化.py

  # 验证时开启 ORT profiling、并提高 CPU 段线程数（移动端请换成本机 QNN 等 providers）
  python onnx模型量化.py best.onnx --verify-ort --ort-profiling --ort-intra-op-threads 4
        """
    )

    parser.add_argument(
        "model",
        nargs="?",
        help="输入 ONNX 模型路径（可省略：在终端交互输入；默认走交互并默认量化 int8_io）",
    )
    parser.add_argument(
        "-t",
        "--target",
        choices=["fp16", "int8", "int8_qnn", "int8_io", "int8_static"],
        help="int8=动态；int8_qnn=静态 QDQ(QNN)；int8_io=静态 QDQ+自动折叠图边界；加 --qnn-int8 用 QInt8 并折叠为 int8 边界（Android QNN）",
    )
    parser.add_argument("-o", "--output", 
                       help="输出模型路径（默认: 自动生成）")
    parser.add_argument("--keep-io-fp32", action="store_true",
                       help="FP16 转换时保持输入输出为 FP32（推荐）")
    parser.add_argument(
        "--calibration-data",
        help="int8_qnn 用：校准数据 .npy（首维为 batch，形状需与模型输入一致）；不提供则用随机数据",
    )
    parser.add_argument(
        "--calibrate-random-batches",
        type=int,
        default=16,
        metavar="N",
        help="未提供 --calibration-data 时，int8_qnn 使用的随机校准批次数（默认 16）",
    )
    parser.add_argument(
        "--uint8-io",
        action="store_true",
        help="量化成功后折叠 Q/DQ 图边界（默认 uint8；加 --qnn-int8 时为 int8）",
    )
    parser.add_argument(
        "--qnn-int8",
        action="store_true",
        help="静态 QNN 量化使用 QInt8 激活与权重；折叠图输入/输出为 tensor(int8)。适用于 Android onnxruntime-android-qnn",
    )
    parser.add_argument(
        "--verify-ort",
        action="store_true",
        help="未使用 -t 时：对输入模型做 ORT 验证；若转换失败：对原始模型验证（转换成功时默认已自动验证输出，无需再加）",
    )
    parser.add_argument(
        "--skip-verify-ort",
        action="store_true",
        help="转换成功后跳过 ONNX Runtime 自动验证（默认会验证输出模型）",
    )
    parser.add_argument(
        "--no-run-ort",
        action="store_true",
        help="ORT 验证时只加载会话，不试跑推理（适用于自动验证与 --verify-ort）",
    )
    parser.add_argument(
        "--ort-intra-op-threads",
        type=int,
        default=0,
        metavar="N",
        help="ORT 验证时设置 SessionOptions.intra_op_num_threads=N；0 表示不修改（默认）。仅缓解落在 CPU 的算子段",
    )
    parser.add_argument(
        "--ort-providers",
        default=None,
        metavar="LIST",
        help="逗号分隔的执行提供程序顺序，如 CPUExecutionProvider 或 QNNExecutionProvider,CPUExecutionProvider；"
        "省略则使用本机 ORT 默认可用列表（移动端与 PC 可用列表不同）",
    )
    parser.add_argument(
        "--ort-profiling",
        action="store_true",
        help="ORT 验证试跑时 enable_profiling，结束后输出 profiling 文件路径，便于分析各 EP 节点占比",
    )

    args = parser.parse_args()

    if args.model is None:
        if sys.stdin.isatty():
            args = interactive_configure_args(parser, args)
        else:
            script = os.path.basename(sys.argv[0] if sys.argv else "onnx模型量化.py")
            print("未指定 ONNX 模型路径，且当前非交互终端。")
            print(f"请传入: python {script} best.onnx")
            print("或在终端直接运行脚本进入交互模式（默认量化 int8_io）。\n")
            parser.print_help()
            sys.exit(1)

    if not os.path.exists(args.model):
        print(f"错误：找不到模型文件 {args.model}")
        sys.exit(1)

    # 第一步：检测当前精度
    checker = ONNXPrecisionChecker(args.model)
    current_precision = checker.print_report()

    convert_success: Optional[bool] = None

    if args.target:
        # 生成默认输出路径
        if not args.output:
            base, ext = os.path.splitext(args.model)
            args.output = f"{base}_{args.target}{ext}"

        converter = ONNXConverter(args.model, args.output)

        if args.target == "fp16":
            convert_success = converter.to_fp16(keep_io_types=args.keep_io_fp32)
        elif args.target == "int8":
            convert_success = converter.to_int8_dynamic()
        elif args.target == "int8_qnn":
            convert_success = converter.to_int8_qnn_static(
                calibration_data_path=args.calibration_data,
                random_batches=max(1, args.calibrate_random_batches),
                qnn_int8=args.qnn_int8,
            )
        elif args.target == "int8_io":
            convert_success = converter.to_int8_qnn_static(
                calibration_data_path=args.calibration_data,
                random_batches=max(1, args.calibrate_random_batches),
                qnn_int8=args.qnn_int8,
            )
        elif args.target == "int8_static":
            convert_success = converter.to_int8_static(args.calibration_data)
        else:
            print(f"错误：不支持的转换类型 {args.target}")
            sys.exit(1)
    else:
        print(
            "\n💡 需要「图边界为整型」请用 -t int8_io（推荐）或 -t int8_qnn 加 --uint8-io；"
            "不要用 -t int8 动态量化跑 NPU 整型路径。转换成功后会自动 ORT 验证（可 --skip-verify-ort）"
        )

    need_io_fold = convert_success and args.output and os.path.isfile(args.output) and (
        args.uint8_io or args.target == "int8_io"
    )
    if args.target and need_io_fold:
        from onnx import TensorProto

        io_elem = TensorProto.INT8 if args.qnn_int8 else TensorProto.UINT8
        ok_fold, fold_msg = fold_qdq_graph_io_uint8(
            args.output, args.output, io_elem_type=io_elem
        )
        print(fold_msg)

    verify_ok: Optional[bool] = None
    should_verify = False
    verify_path: Optional[str] = None

    # 转换成功：默认对输出文件做 ORT 验证（可用 --skip-verify-ort 关闭）
    if args.target and convert_success is True and not args.skip_verify_ort:
        should_verify = True
        verify_path = args.output
    # 未转换时显式要求验证，或转换失败时仍用 --verify-ort 检查原始模型
    elif args.verify_ort:
        should_verify = True
        if convert_success is False:
            print("\n💡 转换未成功，改为对原始模型做 ONNX Runtime 验证。")
        if convert_success is True and args.output and os.path.isfile(args.output):
            verify_path = args.output
        else:
            verify_path = args.model

    if should_verify and verify_path:
        verify_ok = verify_onnxruntime_session(
            verify_path,
            run_inference=not args.no_run_ort,
            intra_op_num_threads=max(0, args.ort_intra_op_threads),
            providers=_parse_ort_providers(args.ort_providers),
            enable_profiling=args.ort_profiling,
        )

    if args.target and convert_success is False:
        sys.exit(1)
    if should_verify and verify_ok is False:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()