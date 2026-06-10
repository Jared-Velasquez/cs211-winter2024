from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


REQUIRED_METADATA_FIELDS = [
    "task_name",
    "task_type",
    "model",
    "partition_id",
    "model_path",
    "input_tensor",
    "output_tensors",
    "data_loader",
    "input_shape",
    "resize",
    "input_dtype",
    "input_normalization",
    "tpu_output_tensors",
    "cpu_input_tensors",
    "boundary_tensor_shapes",
    "cpu_graph_path",
    "quant_mode",
    "num_tpu_ops",
    "num_cpu_ops",
    "boundary_bandwidth_bytes",
    "has_skip_crossing",
]

OPTIONAL_PATH_FIELDS = {"meta_path", "tpu_tflite_path"}
DATA_LOADER_PATH_FIELDS = {
    "video_frames": ["video_path"],
    "coco_images": ["images_dir", "annotations_path"],
    "ap10k_pose": ["images_dir", "annotations_path"],
    "voc_segmentation": ["images_dir", "annotations_dir"],
}
SUMMARY_FIELDS = [
    "task_name",
    "task_type",
    "model",
    "partition_id",
    "quant_mode",
    "num_tpu_ops",
    "num_cpu_ops",
    "boundary_tensor_shapes",
    "boundary_bandwidth_bytes",
    "has_skip_crossing",
    "tpu_ops_mapped_edgetpu",
    "edgetpu_compiled",
    "edgetpu_rejection_reason",
    "input_shape",
    "resize",
    "input_dtype",
    "input_normalization",
]


def _fully_qualified_tensor_name(name: str) -> str:
    return name if ":" in name else f"{name}:0"


def _tensor_name_to_signature_key(name: str) -> str:
    op_name = name.split(":", 1)[0]
    return op_name.replace("/", "_")


def _tensor_name_to_key(name: str) -> str:
    return name.replace(":", "_").replace("/", "_")


def _candidate_error(message: str) -> ValueError:
    return ValueError(f"{message}. Regenerate or backfill this candidate's metadata.json.")


def compare_named_outputs(reference_outputs: dict[str, np.ndarray], candidate_outputs: dict[str, np.ndarray]) -> dict[str, Any]:
    import numpy as np

    diffs: dict[str, Any] = {}
    max_abs_diff = 0.0
    mean_abs_diff = 0.0

    for key, reference_value in reference_outputs.items():
        if key not in candidate_outputs:
            raise KeyError(f"Candidate outputs are missing expected key: {key}")
        candidate_value = candidate_outputs[key]
        if reference_value.shape != candidate_value.shape:
            raise ValueError(
                f"Output shape mismatch for {key}: "
                f"reference {reference_value.shape}, candidate {candidate_value.shape}"
            )
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
    import numpy as np

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
    def __init__(self, candidate_dir: str, require_tpu: bool = True) -> None:
        self.repo_root = Path.cwd().resolve()
        self.candidate_dir = Path(candidate_dir).expanduser().resolve()
        self.require_tpu = require_tpu
        self.metadata_path = self.candidate_dir / "metadata.json"
        self.candidate_metadata = self._load_and_validate_metadata()
        self.config = self._build_loader_config()
        self.boundary_tensors = [
            _fully_qualified_tensor_name(name)
            for name in self.candidate_metadata["tpu_output_tensors"]
        ]
        self.cpu_input_tensors = [
            _fully_qualified_tensor_name(name)
            for name in self.candidate_metadata["cpu_input_tensors"]
        ]

    def run_full_cpu(self, samples: list[dict[str, Any]]) -> tuple[dict[str, np.ndarray], list[float]]:
        import numpy as np
        import tensorflow as tf

        from src.graph_utils import import_graph, load_graph_def
        from src.io_utils import stack_named_outputs

        tf1 = tf.compat.v1
        self._require_samples(samples)
        graph = import_graph(load_graph_def(self.config["model_path"]))
        input_tensor = graph.get_tensor_by_name(self.config["input_tensor"])
        output_tensor_names = list(self.config["output_tensors"])
        output_tensors = [graph.get_tensor_by_name(name) for name in output_tensor_names]

        # Some graphs (e.g. SSD frozen_inference_graph.pb) have additional placeholder
        # tensors (e.g. image_tensor) that the postprocessor references for image shape,
        # even when we bypass the preprocessor by feeding an intermediate tensor directly.
        # Feed dummy zero-filled arrays for those placeholders so TF1 doesn't reject the run.
        primary_op = self.config["input_tensor"].split(":")[0]
        input_shape = [int(d) for d in self.config["input_shape"]]
        extra_feeds: dict = {}
        for op in graph.get_operations():
            if op.type != "Placeholder" or op.name == primary_op:
                continue
            tensor = op.outputs[0]
            raw_shape = tensor.shape.as_list()
            resolved = [
                (input_shape[i] if i < len(input_shape) else 1) if dim is None else dim
                for i, dim in enumerate(raw_shape)
            ]
            extra_feeds[tensor] = np.zeros(resolved, dtype=tensor.dtype.as_numpy_dtype)

        per_frame_outputs = []
        timings_ms = []
        with tf1.Session(graph=graph) as session:
            for sample in samples:
                feed = {input_tensor: sample["input"], **extra_feeds}
                start_time = time.perf_counter()
                values = session.run(output_tensors, feed_dict=feed)
                timings_ms.append((time.perf_counter() - start_time) * 1000.0)
                per_frame_outputs.append(
                    {
                        _tensor_name_to_key(name): value
                        for name, value in zip(output_tensor_names, values)
                    }
                )

        return stack_named_outputs(per_frame_outputs), timings_ms

    def run_float_prefix(self, samples: list[dict[str, Any]]) -> tuple[dict[str, np.ndarray], list[float]]:
        import tensorflow as tf

        from src.graph_utils import extract_prefix_graph_def, import_graph
        from src.io_utils import stack_named_outputs

        tf1 = tf.compat.v1
        self._require_samples(samples)
        prefix_graph = import_graph(
            extract_prefix_graph_def(
                graph_path=self.config["model_path"],
                boundary_tensors=self.boundary_tensors,
                input_tensor_name=self.config["input_tensor"],
                fixed_input_shape=self.config["input_shape"],
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
                        _tensor_name_to_key(name): value
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
        import tensorflow as tf

        from src.io_utils import stack_named_outputs

        cpu_graph_path = Path(self.config["cpu_graph_path"])

        # Try TF2 eager SavedModel first. Some models (e.g. SSD) export a cpu_savedmodel
        # whose signature depends on a placeholder (image_tensor) that was not declared as
        # an input — TF2's "lifting" step rejects this with UnliftableError. In that case
        # fall back to a TF1 session on the original frozen graph with the boundary tensor
        # fed directly as an override.
        try:
            loaded = tf.saved_model.load(str(cpu_graph_path))
        except Exception as exc:
            if "UnliftableError" in type(exc).__name__ or "Unable to lift tensor" in str(exc):
                return self._run_suffix_via_full_graph_tf1(boundary_outputs)
            raise

        signature = loaded.signatures.get("serving_default")
        if signature is None:
            raise ValueError(
                f"Candidate CPU SavedModel has no serving_default signature: {cpu_graph_path}. "
                "Regenerate this candidate's cpu_savedmodel artifact."
            )

        input_keys = [_tensor_name_to_signature_key(name) for name in self.cpu_input_tensors]
        missing_inputs = [key for key in input_keys if key not in signature.structured_input_signature[1]]
        if missing_inputs:
            available = sorted(signature.structured_input_signature[1].keys())
            raise _candidate_error(
                f"Candidate CPU SavedModel missing expected inputs {missing_inputs}; available inputs are {available}"
            )

        sample_count = self._infer_sample_count(boundary_outputs)
        per_frame_outputs = []
        timings_ms = []
        for sample_index in range(sample_count):
            feed = {}
            for tensor_name, input_key in zip(self.cpu_input_tensors, input_keys):
                boundary_key = _tensor_name_to_key(tensor_name)
                if boundary_key not in boundary_outputs:
                    raise KeyError(f"Boundary outputs missing expected key: {boundary_key}")
                feed[input_key] = tf.convert_to_tensor(
                    self._with_batch_dimension(boundary_outputs[boundary_key][sample_index])
                )

            start_time = time.perf_counter()
            values = signature(**feed)
            timings_ms.append((time.perf_counter() - start_time) * 1000.0)
            per_frame_outputs.append(
                {
                    key: value.numpy()
                    for key, value in values.items()
                }
            )

        return stack_named_outputs(per_frame_outputs), timings_ms

    def _run_suffix_via_full_graph_tf1(self, boundary_outputs: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], list[float]]:
        """Fallback suffix runner using TF1 session on the original frozen graph.

        Used when the cpu_savedmodel has dangling placeholders (e.g. SSD's image_tensor)
        that prevent TF2 eager loading. The boundary tensors are fed as overrides directly
        into the full graph. Any other placeholders receive zero-filled dummy tensors of
        the correct shape — for SSD this is image_tensor [1,H,W,3] uint8, whose shape
        determines H and W for the Postprocessor's box normalization; only shape matters.
        """
        import numpy as np
        import tensorflow as tf

        from src.graph_utils import import_graph, load_graph_def
        from src.io_utils import stack_named_outputs

        tf1 = tf.compat.v1
        graph = import_graph(load_graph_def(self.config["model_path"]))
        output_tensor_names = list(self.config["output_tensors"])
        output_tensors = [graph.get_tensor_by_name(name) for name in output_tensor_names]

        boundary_op_names = {name.split(":")[0] for name in self.boundary_tensors}
        input_shape = [int(d) for d in self.config["input_shape"]]
        extra_feeds: dict = {}
        for op in graph.get_operations():
            # Skip non-placeholders and any tensor we are feeding as a boundary override.
            # Do NOT skip the model's primary input placeholder (e.g. ImageTensor for DeepLab,
            # image_tensor for SSD) — the suffix run does not feed the primary input, so we
            # must supply a dummy zero tensor. The decoder/postprocessor uses it only for shape
            # lookups (output resize H×W, box normalization), not for pixel computation.
            if op.type != "Placeholder" or op.name in boundary_op_names:
                continue
            tensor = op.outputs[0]
            raw_shape = tensor.shape.as_list()
            resolved = [
                (input_shape[i] if i < len(input_shape) else 1) if dim is None else dim
                for i, dim in enumerate(raw_shape)
            ]
            extra_feeds[tensor] = np.zeros(resolved, dtype=tensor.dtype.as_numpy_dtype)

        sample_count = self._infer_sample_count(boundary_outputs)
        per_frame_outputs = []
        timings_ms = []
        with tf1.Session(graph=graph) as session:
            for sample_index in range(sample_count):
                feed = dict(extra_feeds)
                for tensor_name in self.boundary_tensors:
                    boundary_key = _tensor_name_to_key(tensor_name)
                    if boundary_key not in boundary_outputs:
                        raise KeyError(f"Boundary outputs missing expected key: {boundary_key}")
                    tensor = graph.get_tensor_by_name(tensor_name)
                    feed[tensor] = self._with_batch_dimension(
                        boundary_outputs[boundary_key][sample_index]
                    )
                start_time = time.perf_counter()
                values = session.run(output_tensors, feed_dict=feed)
                timings_ms.append((time.perf_counter() - start_time) * 1000.0)
                per_frame_outputs.append(
                    {
                        _tensor_name_to_key(name): value
                        for name, value in zip(output_tensor_names, values)
                    }
                )

        return stack_named_outputs(per_frame_outputs), timings_ms

    def run_hybrid_tpu(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        from src.io_utils import stack_named_outputs

        self._require_samples(samples)
        compiled_path = Path(self.config["tpu_edgetpu_path"])
        interpreter, input_details, output_details = self._load_tpu_interpreter(str(compiled_path))
        self._validate_tflite_contract(input_details, output_details)

        output_metadata = self._build_tflite_output_metadata(output_details)
        input_detail = input_details[0]
        input_scale, input_zero_point = input_detail.get("quantization", (0.0, 0))

        per_frame_boundaries = []
        timings = []
        for sample in samples:
            total_start = time.perf_counter()
            transfer_ms = 0.0
            transfer_start = time.perf_counter()
            self._set_tflite_input(interpreter, input_detail, sample["input"], input_scale, input_zero_point)
            transfer_ms += (time.perf_counter() - transfer_start) * 1000.0

            tpu_start = time.perf_counter()
            interpreter.invoke()
            t_tpu = (time.perf_counter() - tpu_start) * 1000.0

            transfer_start = time.perf_counter()
            boundary_values = {}
            for boundary_tensor, output_detail in zip(self.boundary_tensors, output_details):
                raw_value = interpreter.get_tensor(output_detail["index"])
                boundary_values[_tensor_name_to_key(boundary_tensor)] = self._dequantize_output(raw_value, output_detail)
            transfer_ms += (time.perf_counter() - transfer_start) * 1000.0
            per_frame_boundaries.append(boundary_values)
            timings.append(
                {
                    "t_tpu": t_tpu,
                    "t_transfer": transfer_ms,
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

    def validate_samples(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        import numpy as np

        self._require_samples(samples)
        expected_input_shape = tuple(int(dim) for dim in self.config["input_shape"])
        normalization = self.config["input_normalization"]
        input_dtype = np.dtype(self.config["input_dtype"])
        validated = []

        for sample in samples:
            value = np.asarray(sample["input"])
            if normalization == "ssd":
                value = (value / 128.0) - 1.0
            elif normalization != "none":
                raise _candidate_error(f"Unsupported input_normalization value: {normalization}")

            value = value.astype(input_dtype, copy=False)
            if tuple(value.shape) != expected_input_shape:
                raise _candidate_error(
                    f"Sample {sample.get('sample_id')} has input shape {list(value.shape)}, expected {list(expected_input_shape)}"
                )

            sample_copy = dict(sample)
            sample_copy["input"] = value
            sample_copy["input_shape"] = list(value.shape[1:])
            validated.append(sample_copy)

        return validated

    def candidate_summary_fields(self) -> dict[str, Any]:
        return {
            key: self.candidate_metadata.get(key)
            for key in SUMMARY_FIELDS
            if key in self.candidate_metadata
        }

    def results_path(self, filename: str) -> Path:
        return self.candidate_dir / "results" / filename

    def _load_and_validate_metadata(self) -> dict[str, Any]:
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Candidate metadata not found: {self.metadata_path}. "
                "Regenerate or backfill this candidate's metadata.json."
            )
        try:
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise _candidate_error(f"Candidate metadata is not valid JSON: {exc}") from exc
        if not isinstance(metadata, dict):
            raise _candidate_error("Candidate metadata must be a JSON object")

        loader = metadata.get("data_loader")
        required = list(REQUIRED_METADATA_FIELDS)
        required.extend(DATA_LOADER_PATH_FIELDS.get(loader, []))
        if self.require_tpu:
            required.extend(["tpu_edgetpu_path", "edgetpu_compiled", "tpu_ops_mapped_edgetpu"])

        missing = [key for key in required if key not in metadata]
        if missing:
            raise _candidate_error(f"Candidate metadata missing required field `{missing[0]}`")

        self._validate_metadata_types(metadata)
        self._validate_metadata_shapes(metadata)
        self._validate_metadata_paths(metadata)
        if self.require_tpu and metadata["edgetpu_compiled"] is not True:
            raise _candidate_error("Candidate metadata field `edgetpu_compiled` must be true for TPU mode")

        return metadata

    def _validate_metadata_types(self, metadata: dict[str, Any]) -> None:
        list_fields = [
            "output_tensors",
            "input_shape",
            "resize",
            "tpu_output_tensors",
            "cpu_input_tensors",
            "boundary_tensor_shapes",
        ]
        string_fields = [
            "task_name",
            "task_type",
            "model",
            "partition_id",
            "model_path",
            "input_tensor",
            "data_loader",
            "input_dtype",
            "input_normalization",
            "cpu_graph_path",
            "quant_mode",
        ]
        int_fields = ["num_tpu_ops", "num_cpu_ops"]

        for field in list_fields:
            if not isinstance(metadata[field], list):
                raise _candidate_error(f"Candidate metadata field `{field}` must be a list")
        for field in string_fields:
            if not isinstance(metadata[field], str) or not metadata[field]:
                raise _candidate_error(f"Candidate metadata field `{field}` must be a non-empty string")
        for field in int_fields:
            if not isinstance(metadata[field], int):
                raise _candidate_error(f"Candidate metadata field `{field}` must be an integer")
        for field in ["output_tensors", "tpu_output_tensors", "cpu_input_tensors"]:
            if not all(isinstance(item, str) and item for item in metadata[field]):
                raise _candidate_error(f"Candidate metadata field `{field}` must contain non-empty strings")
        if metadata["data_loader"] not in DATA_LOADER_PATH_FIELDS:
            raise _candidate_error(f"Unsupported candidate data_loader `{metadata['data_loader']}`")
        if metadata["input_dtype"] not in {"float32", "uint8"}:
            raise _candidate_error("Candidate metadata field `input_dtype` must be `float32` or `uint8`")
        if metadata["input_normalization"] not in {"none", "ssd"}:
            raise _candidate_error("Candidate metadata field `input_normalization` must be `none` or `ssd`")
        if self.require_tpu and not isinstance(metadata["edgetpu_compiled"], bool):
            raise _candidate_error("Candidate metadata field `edgetpu_compiled` must be a boolean")

    def _validate_metadata_shapes(self, metadata: dict[str, Any]) -> None:
        input_shape = metadata["input_shape"]
        resize = metadata["resize"]
        if len(input_shape) != 4 or not all(isinstance(dim, int) and dim > 0 for dim in input_shape):
            raise _candidate_error("Candidate metadata field `input_shape` must be four positive integers")
        if len(resize) != 2 or not all(isinstance(dim, int) and dim > 0 for dim in resize):
            raise _candidate_error("Candidate metadata field `resize` must be [width, height]")
        if input_shape[0] != 1:
            raise _candidate_error("Candidate metadata field `input_shape` must use batch size 1")
        if input_shape[3] != 3:
            raise _candidate_error("Candidate metadata field `input_shape` must use 3 input channels")
        if [input_shape[2], input_shape[1]] != resize:
            raise _candidate_error(
                f"Candidate resize {resize} does not match input_shape height/width {input_shape}"
            )

        boundary_count = len(metadata["tpu_output_tensors"])
        if boundary_count == 0:
            raise _candidate_error("Candidate metadata field `tpu_output_tensors` must not be empty")
        if len(metadata["cpu_input_tensors"]) != boundary_count:
            raise _candidate_error("Candidate metadata `cpu_input_tensors` count must match `tpu_output_tensors` count")
        if len(metadata["boundary_tensor_shapes"]) != boundary_count:
            raise _candidate_error("Candidate metadata `boundary_tensor_shapes` count must match `tpu_output_tensors` count")
        for shape in metadata["boundary_tensor_shapes"]:
            if not isinstance(shape, list) or not all(isinstance(dim, int) and dim > 0 for dim in shape):
                raise _candidate_error("Candidate metadata `boundary_tensor_shapes` entries must be positive integer lists")

    def _validate_metadata_paths(self, metadata: dict[str, Any]) -> None:
        path_fields = ["model_path", "cpu_graph_path"]
        path_fields.extend(DATA_LOADER_PATH_FIELDS[metadata["data_loader"]])
        if self.require_tpu:
            path_fields.append("tpu_edgetpu_path")
        for field in OPTIONAL_PATH_FIELDS:
            if field in metadata and metadata[field]:
                path_fields.append(field)

        for field in path_fields:
            resolved = self._resolve_metadata_path(metadata[field])
            if not resolved.exists():
                raise FileNotFoundError(
                    f"Candidate metadata path field `{field}` does not exist: {resolved}. "
                    "Regenerate or backfill this candidate's metadata.json."
                )
            metadata[field] = str(resolved)

    def _build_loader_config(self) -> dict[str, Any]:
        config_keys = {
            "task_name",
            "task_type",
            "model_path",
            "input_tensor",
            "output_tensors",
            "data_loader",
            "resize",
            "input_shape",
            "input_dtype",
            "input_normalization",
            "cpu_graph_path",
            "tpu_edgetpu_path",
            "video_path",
            "images_dir",
            "annotations_path",
            "annotations_dir",
            "meta_path",
            "require_single_instance",
            "annotation_strategy",
            "skip_missing_images",
        }
        return {
            key: value
            for key, value in self.candidate_metadata.items()
            if key in config_keys
        }

    def _resolve_metadata_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            return path.resolve()

        candidates = [
            (self.repo_root / path).resolve(),
            (self.candidate_dir / path).resolve(),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _validate_tflite_contract(self, input_details: list[dict[str, Any]], output_details: list[dict[str, Any]]) -> None:
        if len(input_details) != 1:
            raise _candidate_error(f"Compiled TFLite model must have one input, found {len(input_details)}")
        if len(output_details) != len(self.boundary_tensors):
            raise _candidate_error(
                "Compiled TFLite output count does not match `tpu_output_tensors`: "
                f"{len(output_details)} outputs for {len(self.boundary_tensors)} boundaries"
            )

        actual_shape = [int(dim) for dim in input_details[0].get("shape", [])]
        expected_shape = [int(dim) for dim in self.config["input_shape"]]
        if actual_shape != expected_shape:
            raise _candidate_error(
                f"Compiled TFLite input shape {actual_shape} does not match metadata input_shape {expected_shape}"
            )

    @staticmethod
    def _require_samples(samples: list[dict[str, Any]]) -> None:
        if not samples:
            raise RuntimeError("No input samples were loaded.")

    @staticmethod
    def _infer_sample_count(outputs: dict[str, np.ndarray]) -> int:
        if not outputs:
            return 0
        first_value = next(iter(outputs.values()))
        return int(first_value.shape[0])

    @staticmethod
    def _with_batch_dimension(value: np.ndarray) -> np.ndarray:
        import numpy as np

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
        import numpy as np

        metadata = []
        for index, (boundary_tensor, detail) in enumerate(zip(self.boundary_tensors, output_details)):
            scale, zero_point = detail.get("quantization", (0.0, 0))
            metadata.append(
                {
                    "index": index,
                    "tflite_index": int(detail["index"]),
                    "name": detail.get("name"),
                    "boundary_tensor": boundary_tensor,
                    "boundary_key": _tensor_name_to_key(boundary_tensor),
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
        import numpy as np

        target_dtype = np.dtype(input_detail["dtype"])
        tensor_value = value
        if np.issubdtype(target_dtype, np.integer):
            if scale:
                tensor_value = np.round(value / scale + zero_point)
            else:
                # scale=0 means no quantization params embedded — model expects raw integer
                # pixel data (e.g. DeepLab ImageTensor uint8). Input is float32 in [0,255];
                # round and clip to the target integer range.
                tensor_value = np.round(value)
            tensor_value = np.clip(tensor_value, np.iinfo(target_dtype).min, np.iinfo(target_dtype).max)
        interpreter.set_tensor(input_detail["index"], tensor_value.astype(target_dtype))

    @staticmethod
    def _dequantize_output(raw_value: np.ndarray, output_detail: dict[str, Any]) -> np.ndarray:
        import numpy as np

        dtype = np.dtype(output_detail["dtype"])
        scale, zero_point = output_detail.get("quantization", (0.0, 0))
        if np.issubdtype(dtype, np.integer) and scale:
            return (raw_value.astype(np.float32) - float(zero_point)) * float(scale)
        return raw_value.astype(np.float32, copy=False)
