from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from src.config_utils import get_boundary_tensors
from src.graph_utils import extract_prefix_graph_def, import_graph, load_graph_def
from src.io_utils import stack_named_outputs, tensor_name_to_key

tf1 = tf.compat.v1


def compare_named_outputs(reference_outputs: dict[str, np.ndarray], candidate_outputs: dict[str, np.ndarray]) -> dict[str, Any]:
    diffs: dict[str, Any] = {}
    max_abs_diff = 0.0
    mean_abs_diff = 0.0

    for key, reference_value in reference_outputs.items():
        candidate_value = candidate_outputs[key]
        diff = np.abs(reference_value - candidate_value)
        diffs[key] = {
            "shape": list(reference_value.shape),
            "max_abs_diff": float(diff.max()) if diff.size else 0.0,
            "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
        }
        max_abs_diff = max(max_abs_diff, diffs[key]["max_abs_diff"])
        mean_abs_diff += diffs[key]["mean_abs_diff"]

    if diffs:
        mean_abs_diff /= len(diffs)

    return {
        "per_output": diffs,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
    }


def summarize_timing(timings: list[dict[str, float]]) -> dict[str, Any]:
    if not timings:
        return {"per_frame": [], "mean": {}}

    keys = list(timings[0].keys())
    return {
        "per_frame": timings,
        "mean": {
            key: float(np.mean([item[key] for item in timings]))
            for key in keys
        },
    }


class HybridRunner:
    def __init__(self, config: dict[str, Any], boundary_tensors: list[str] | None = None) -> None:
        self.config = config
        self.boundary_tensors = get_boundary_tensors(config, override=boundary_tensors)

    def run_full_cpu(self, samples: list[dict[str, Any]]) -> tuple[dict[str, np.ndarray], list[float]]:
        graph = import_graph(load_graph_def(self.config["model_path"]))
        input_tensor = graph.get_tensor_by_name(self.config["input_tensor"])
        output_tensor_names = list(self.config["output_tensors"])
        output_tensors = [graph.get_tensor_by_name(name) for name in output_tensor_names]

        per_frame_outputs = []
        timings_ms = []
        with tf1.Session(graph=graph) as session:
            for sample in samples:
                start_time = time.perf_counter()
                values = session.run(output_tensors, feed_dict={input_tensor: sample["input"]})
                timings_ms.append((time.perf_counter() - start_time) * 1000.0)
                per_frame_outputs.append(
                    {
                        tensor_name_to_key(name): value
                        for name, value in zip(output_tensor_names, values)
                    }
                )

        return stack_named_outputs(per_frame_outputs), timings_ms

    def run_float_prefix(self, samples: list[dict[str, Any]]) -> tuple[dict[str, np.ndarray], list[float]]:
        prefix_graph = import_graph(
            extract_prefix_graph_def(
                graph_path=self.config["model_path"],
                boundary_tensors=self.boundary_tensors,
                input_tensor_name=self.config["input_tensor"],
                fixed_input_shape=self.config.get("fixed_input_shape"),
            )
        )
        input_tensor = prefix_graph.get_tensor_by_name(self.config["input_tensor"])
        boundary_output_tensors = [prefix_graph.get_tensor_by_name(name) for name in self.boundary_tensors]

        per_frame_boundaries = []
        timings_ms = []
        with tf1.Session(graph=prefix_graph) as session:
            for sample in samples:
                start_time = time.perf_counter()
                values = session.run(boundary_output_tensors, feed_dict={input_tensor: sample["input"]})
                timings_ms.append((time.perf_counter() - start_time) * 1000.0)
                per_frame_boundaries.append(
                    {
                        tensor_name_to_key(name): value
                        for name, value in zip(self.boundary_tensors, values)
                    }
                )

        return stack_named_outputs(per_frame_boundaries), timings_ms

    def run_partitioned_cpu(
        self,
        samples: list[dict[str, Any]],
    ) -> tuple[dict[str, np.ndarray], dict[str, list[float]], dict[str, np.ndarray]]:
        boundary_outputs, prefix_timings_ms = self.run_float_prefix(samples)
        suffix_outputs, suffix_timings_ms = self.run_suffix_from_boundaries(boundary_outputs)
        return (
            suffix_outputs,
            {"prefix_ms": prefix_timings_ms, "suffix_ms": suffix_timings_ms},
            boundary_outputs,
        )

    def run_suffix_from_boundaries(self, boundary_outputs: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], list[float]]:
        metadata = self._load_split_metadata()
        suffix_graph = import_graph(load_graph_def(metadata["suffix_graph_path"]))
        placeholder_map = metadata["suffix_placeholder_map"]
        output_tensor_names = list(metadata["output_tensors"])

        placeholder_tensors = [
            suffix_graph.get_tensor_by_name(placeholder_map[boundary_tensor])
            for boundary_tensor in self.boundary_tensors
        ]
        output_tensors = [suffix_graph.get_tensor_by_name(name) for name in output_tensor_names]
        sample_count = self._infer_sample_count(boundary_outputs)

        per_frame_outputs = []
        timings_ms = []
        with tf1.Session(graph=suffix_graph) as session:
            for sample_index in range(sample_count):
                feed_dict = {
                    placeholder: self._with_batch_dimension(
                        boundary_outputs[tensor_name_to_key(boundary_tensor)][sample_index]
                    )
                    for placeholder, boundary_tensor in zip(placeholder_tensors, self.boundary_tensors)
                }
                start_time = time.perf_counter()
                values = session.run(output_tensors, feed_dict=feed_dict)
                timings_ms.append((time.perf_counter() - start_time) * 1000.0)
                per_frame_outputs.append(
                    {
                        tensor_name_to_key(name): value
                        for name, value in zip(output_tensor_names, values)
                    }
                )

        return stack_named_outputs(per_frame_outputs), timings_ms

    def run_hybrid_tpu(
        self,
        samples: list[dict[str, Any]],
        compiled_tflite_path: str,
    ) -> dict[str, Any]:
        compiled_path = Path(compiled_tflite_path)
        if not compiled_path.exists():
            raise FileNotFoundError(
                f"Compiled Edge TPU model not found: {compiled_path}. "
                "Run edgetpu_compiler on the prefix TFLite model first."
            )

        interpreter, input_details, output_details = self._load_tpu_interpreter(str(compiled_path))
        if len(output_details) != len(self.boundary_tensors):
            raise ValueError(
                "TFLite output count does not match boundary tensor count: "
                f"{len(output_details)} outputs for {len(self.boundary_tensors)} boundaries. "
                f"Boundaries: {self.boundary_tensors}"
            )

        output_metadata = self._build_tflite_output_metadata(output_details)
        input_detail = input_details[0]
        input_scale, input_zero_point = input_detail.get("quantization", (0.0, 0))

        per_frame_boundaries = []
        timings = []
        for sample in samples:
            total_start = time.perf_counter()
            self._set_tflite_input(interpreter, input_detail, sample["input"], input_scale, input_zero_point)

            tpu_start = time.perf_counter()
            interpreter.invoke()
            t_tpu = (time.perf_counter() - tpu_start) * 1000.0

            transfer_start = time.perf_counter()
            boundary_values = {}
            for boundary_tensor, output_detail in zip(self.boundary_tensors, output_details):
                raw_value = interpreter.get_tensor(output_detail["index"])
                boundary_values[tensor_name_to_key(boundary_tensor)] = self._dequantize_output(raw_value, output_detail)
            t_transfer = (time.perf_counter() - transfer_start) * 1000.0
            per_frame_boundaries.append(boundary_values)
            timings.append(
                {
                    "t_tpu": t_tpu,
                    "t_transfer": t_transfer,
                    "t_cpu": 0.0,
                    "t_total": (time.perf_counter() - total_start) * 1000.0,
                }
            )

        boundary_outputs = stack_named_outputs(per_frame_boundaries)
        suffix_outputs, suffix_timings_ms = self.run_suffix_from_boundaries(boundary_outputs)
        for timing, suffix_ms in zip(timings, suffix_timings_ms):
            timing["t_cpu"] = suffix_ms
            timing["t_total"] = timing["t_tpu"] + timing["t_transfer"] + timing["t_cpu"]

        return {
            "outputs": suffix_outputs,
            "boundary_outputs": boundary_outputs,
            "timings_ms": timings,
            "tflite_output_metadata": output_metadata,
        }

    def _load_split_metadata(self) -> dict[str, Any]:
        metadata_path = Path(self.config["split_metadata_path"])
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Split metadata not found: {metadata_path}. "
                "Run split.py --config <config> --force before hybrid execution."
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        suffix_graph_path = Path(metadata["suffix_graph_path"])
        if not suffix_graph_path.exists():
            raise FileNotFoundError(
                f"Suffix graph not found: {suffix_graph_path}. "
                "Run split.py --config <config> --force before hybrid execution."
            )
        for boundary_tensor in self.boundary_tensors:
            if boundary_tensor not in metadata["suffix_placeholder_map"]:
                raise ValueError(f"Boundary tensor missing from split metadata: {boundary_tensor}")
        return metadata

    @staticmethod
    def _infer_sample_count(outputs: dict[str, np.ndarray]) -> int:
        if not outputs:
            return 0
        first_value = next(iter(outputs.values()))
        return int(first_value.shape[0])

    @staticmethod
    def _with_batch_dimension(value: np.ndarray) -> np.ndarray:
        return np.expand_dims(value, axis=0)

    @staticmethod
    def _load_tpu_interpreter(compiled_tflite_path: str):
        try:
            from pycoral.utils.edgetpu import make_interpreter
        except ImportError as exc:
            raise RuntimeError(
                "PyCoral is required for TPU execution. Install the Coral Python runtime "
                "on the target Linux/Coral machine, then rerun this command."
            ) from exc

        interpreter = make_interpreter(compiled_tflite_path)
        interpreter.allocate_tensors()
        return interpreter, interpreter.get_input_details(), interpreter.get_output_details()

    def _build_tflite_output_metadata(self, output_details: list[dict[str, Any]]) -> list[dict[str, Any]]:
        metadata = []
        for index, (boundary_tensor, detail) in enumerate(zip(self.boundary_tensors, output_details)):
            scale, zero_point = detail.get("quantization", (0.0, 0))
            metadata.append(
                {
                    "index": index,
                    "tflite_index": int(detail["index"]),
                    "name": detail.get("name"),
                    "boundary_tensor": boundary_tensor,
                    "boundary_key": tensor_name_to_key(boundary_tensor),
                    "dtype": np.dtype(detail["dtype"]).name,
                    "shape": [int(dim) for dim in detail.get("shape", [])],
                    "quantization": {
                        "scale": float(scale),
                        "zero_point": int(zero_point),
                        "dequantized": bool(scale and np.issubdtype(np.dtype(detail["dtype"]), np.integer)),
                    },
                }
            )
        return metadata

    @staticmethod
    def _set_tflite_input(interpreter, input_detail: dict[str, Any], value: np.ndarray, scale: float, zero_point: int) -> None:
        target_dtype = np.dtype(input_detail["dtype"])
        tensor_value = value
        if np.issubdtype(target_dtype, np.integer):
            if not scale:
                raise ValueError("Quantized TFLite input is missing a non-zero quantization scale.")
            tensor_value = np.round(value / scale + zero_point)
            tensor_value = np.clip(tensor_value, np.iinfo(target_dtype).min, np.iinfo(target_dtype).max)
        interpreter.set_tensor(input_detail["index"], tensor_value.astype(target_dtype))

    @staticmethod
    def _dequantize_output(raw_value: np.ndarray, output_detail: dict[str, Any]) -> np.ndarray:
        dtype = np.dtype(output_detail["dtype"])
        scale, zero_point = output_detail.get("quantization", (0.0, 0))
        if np.issubdtype(dtype, np.integer) and scale:
            return (raw_value.astype(np.float32) - float(zero_point)) * float(scale)
        return raw_value.astype(np.float32, copy=False)
