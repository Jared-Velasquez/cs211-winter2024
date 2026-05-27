from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from src.config_utils import load_config
from src.data_loaders import load_samples
from src.io_utils import save_json, save_npz, stack_named_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a full model on the Edge TPU and record timing and outputs.")
    parser.add_argument("--config", required=True, help="Path to the task config JSON.")
    parser.add_argument("--tflite-model", default=None, help="Path to compiled Edge TPU .tflite. Defaults to config tflite_model_path.")
    parser.add_argument("--frame-limit", type=int, default=None, help="Maximum number of samples to evaluate.")
    return parser.parse_args()


def _set_input(interpreter, input_detail: dict, value: np.ndarray) -> None:
    target_dtype = np.dtype(input_detail["dtype"])
    if np.issubdtype(target_dtype, np.integer):
        scale, zero_point = input_detail.get("quantization", (0.0, 0))
        if not scale:
            raise ValueError("Quantized TFLite input has no quantization scale.")
        tensor_value = np.round(value / scale + zero_point)
        tensor_value = np.clip(tensor_value, np.iinfo(target_dtype).min, np.iinfo(target_dtype).max)
        interpreter.set_tensor(input_detail["index"], tensor_value.astype(target_dtype))
    else:
        interpreter.set_tensor(input_detail["index"], value.astype(target_dtype))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    tflite_path = args.tflite_model or config.get("tflite_model_path")
    if not tflite_path:
        raise ValueError("No tflite_model_path in config and --tflite-model not provided.")
    if not Path(tflite_path).exists():
        raise FileNotFoundError(f"TFLite model not found: {tflite_path}")

    from pycoral.utils.edgetpu import make_interpreter
    interpreter = make_interpreter(tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_detail = input_details[0]

    samples = load_samples(config, frame_limit=args.frame_limit)
    if not samples:
        raise RuntimeError("No samples loaded.")

    per_frame_outputs = []
    latency_per_frame = []

    print(f"Running {len(samples)} frames on Edge TPU: {Path(tflite_path).name}")
    for sample in samples:
        _set_input(interpreter, input_detail, sample["input"])
        t0 = time.perf_counter()
        interpreter.invoke()
        latency_per_frame.append((time.perf_counter() - t0) * 1000.0)
        per_frame_outputs.append({
            detail["name"]: interpreter.get_tensor(detail["index"]).copy()
            for detail in output_details
        })

    outputs = stack_named_outputs(per_frame_outputs)
    artifacts_dir = Path(config["artifacts_dir"])
    output_path = str(artifacts_dir / "tpu_full_outputs.npz")
    save_npz(output_path, outputs)

    latency_ms = {
        "mean": float(np.mean(latency_per_frame)),
        "min": float(np.min(latency_per_frame)),
        "max": float(np.max(latency_per_frame)),
        "per_frame": latency_per_frame,
    }

    tflite_output_metadata = [
        {
            "name": d["name"],
            "shape": [int(x) for x in d["shape"]],
            "dtype": np.dtype(d["dtype"]).name,
            "quantization": list(d.get("quantization", (0.0, 0))),
        }
        for d in output_details
    ]

    summary = {
        "task_name": config["task_name"],
        "task_type": config["task_type"],
        "mode": "tpu_full",
        "tflite_model_path": tflite_path,
        "num_frames": len(samples),
        "sample_ids": [s["sample_id"] for s in samples],
        "latency_ms": latency_ms,
        "output_shapes": {k: list(v.shape) for k, v in outputs.items()},
        "tflite_output_metadata": tflite_output_metadata,
        "output_path": output_path,
    }

    summary_path = str(artifacts_dir / "tpu_full_summary.json")
    save_json(summary_path, summary)

    print(f"  mean latency: {latency_ms['mean']:.1f}ms  min: {latency_ms['min']:.1f}ms  max: {latency_ms['max']:.1f}ms")
    print(f"  outputs: {', '.join(f'{k} {list(v.shape)}' for k, v in outputs.items())}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved outputs: {output_path}")


if __name__ == "__main__":
    main()
