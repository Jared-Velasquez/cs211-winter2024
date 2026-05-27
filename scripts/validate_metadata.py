"""Validate candidate metadata.json files for the strict HybridRunner format."""
import argparse
import json
import os
import sys
from pathlib import Path


REQUIRED_FIELDS = [
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
    "tpu_tflite_path",
    "cpu_graph_path",
    "num_tpu_ops",
    "num_cpu_ops",
    "boundary_tensor_shapes",
    "boundary_bandwidth_bytes",
    "has_skip_crossing",
    "quant_mode",
    "edgetpu_compiled",
    "tpu_edgetpu_path",
    "tpu_ops_mapped_edgetpu",
    "edgetpu_rejection_reason",
]

DATA_LOADER_PATH_FIELDS = {
    "video_frames": ["video_path"],
    "coco_images": ["images_dir", "annotations_path"],
    "ap10k_pose": ["images_dir", "annotations_path"],
    "voc_segmentation": ["images_dir", "annotations_dir"],
}

PATH_FIELDS = {
    "model_path",
    "tpu_tflite_path",
    "cpu_graph_path",
    "video_path",
    "images_dir",
    "annotations_path",
    "annotations_dir",
    "meta_path",
}


def find_metadata_files(root):
    out = []
    for dirpath, _, filenames in os.walk(root):
        if "metadata.json" in filenames:
            out.append(os.path.join(dirpath, "metadata.json"))
    return sorted(out)


def _resolve(path_value, metadata_path):
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    repo_root = Path.cwd()
    candidate_dir = Path(metadata_path).parent
    repo_path = repo_root / path
    if repo_path.exists():
        return repo_path
    candidate_path = candidate_dir / path
    if candidate_path.exists():
        return candidate_path
    return repo_path


def _is_nonempty_string(value):
    return isinstance(value, str) and bool(value)


def _is_string_list(value):
    return isinstance(value, list) and all(_is_nonempty_string(item) for item in value)


def _is_int_list(value, length=None):
    if not isinstance(value, list):
        return False
    if length is not None and len(value) != length:
        return False
    return all(isinstance(item, int) and item > 0 for item in value)


def validate(path):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:  # noqa: BLE001
        return f"parse error: {exc}", "?", "?", "?", "?"

    if not isinstance(data, dict):
        return "not a JSON object", "?", "?", "?", "?"

    pid = data.get("partition_id", "?")
    model = data.get("model", "?")
    n_tpu = data.get("num_tpu_ops", "?")
    n_cpu = data.get("num_cpu_ops", "?")

    missing = [key for key in REQUIRED_FIELDS if key not in data]
    if missing:
        return f"missing field {missing[0]}", pid, model, n_tpu, n_cpu

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
    ]
    for field in string_fields:
        if not _is_nonempty_string(data[field]):
            return f"{field} not a non-empty string", pid, model, n_tpu, n_cpu

    for field in ["output_tensors", "tpu_output_tensors", "cpu_input_tensors"]:
        if not _is_string_list(data[field]):
            return f"{field} not a non-empty string list", pid, model, n_tpu, n_cpu

    if not _is_int_list(data["input_shape"], length=4):
        return "input_shape must be four positive integers", pid, model, n_tpu, n_cpu
    if not _is_int_list(data["resize"], length=2):
        return "resize must be [width, height]", pid, model, n_tpu, n_cpu
    if data["resize"] != [data["input_shape"][2], data["input_shape"][1]]:
        return "resize does not match input_shape width/height", pid, model, n_tpu, n_cpu

    boundary_count = len(data["tpu_output_tensors"])
    if len(data["cpu_input_tensors"]) != boundary_count:
        return "cpu_input_tensors count mismatch", pid, model, n_tpu, n_cpu
    if len(data["boundary_tensor_shapes"]) != boundary_count:
        return "boundary_tensor_shapes count mismatch", pid, model, n_tpu, n_cpu
    for shape in data["boundary_tensor_shapes"]:
        if not _is_int_list(shape):
            return "boundary_tensor_shapes contains invalid shape", pid, model, n_tpu, n_cpu

    if not isinstance(data["num_tpu_ops"], int):
        return "num_tpu_ops not an int", pid, model, n_tpu, n_cpu
    if not isinstance(data["num_cpu_ops"], int):
        return "num_cpu_ops not an int", pid, model, n_tpu, n_cpu
    if not isinstance(data["has_skip_crossing"], bool):
        return "has_skip_crossing not a bool", pid, model, n_tpu, n_cpu
    if data["boundary_bandwidth_bytes"] is not None and not isinstance(data["boundary_bandwidth_bytes"], int):
        return "boundary_bandwidth_bytes not an int or null", pid, model, n_tpu, n_cpu
    if data["edgetpu_compiled"] is not None and not isinstance(data["edgetpu_compiled"], bool):
        return "edgetpu_compiled not a bool or null", pid, model, n_tpu, n_cpu

    loader = data["data_loader"]
    if loader not in DATA_LOADER_PATH_FIELDS:
        return f"unsupported data_loader {loader}", pid, model, n_tpu, n_cpu
    for field in DATA_LOADER_PATH_FIELDS[loader]:
        if field not in data:
            return f"missing loader path field {field}", pid, model, n_tpu, n_cpu

    if data["input_dtype"] not in {"float32", "uint8"}:
        return "input_dtype must be float32 or uint8", pid, model, n_tpu, n_cpu
    if data["input_normalization"] not in {"none", "ssd"}:
        return "input_normalization must be none or ssd", pid, model, n_tpu, n_cpu

    path_fields = {"model_path", "tpu_tflite_path", "cpu_graph_path"}
    path_fields.update(DATA_LOADER_PATH_FIELDS[loader])
    if data.get("meta_path"):
        path_fields.add("meta_path")
    for field in sorted(path_fields):
        value = data.get(field)
        if not _is_nonempty_string(value):
            return f"{field} not a non-empty string", pid, model, n_tpu, n_cpu
        if not _resolve(value, path).exists():
            return f"path does not exist: {field}={value}", pid, model, n_tpu, n_cpu

    if data["edgetpu_compiled"] is True:
        if not _is_nonempty_string(data["tpu_edgetpu_path"]):
            return "compiled candidate missing tpu_edgetpu_path", pid, model, n_tpu, n_cpu
        if not _resolve(data["tpu_edgetpu_path"], path).exists():
            return "compiled candidate tpu_edgetpu_path does not exist", pid, model, n_tpu, n_cpu

    return "OK", pid, model, n_tpu, n_cpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="artifacts",
                        help="Directory to walk for metadata.json files")
    args = parser.parse_args()

    files = find_metadata_files(args.root)
    if not files:
        print(f"No metadata.json files found under {args.root}")
        return 0

    rows = []
    any_bad = False
    for metadata_file in files:
        status, pid, model, n_tpu, n_cpu = validate(metadata_file)
        rows.append((pid, model, n_tpu, n_cpu, status, metadata_file))
        if status != "OK":
            any_bad = True

    headers = ("partition_id", "model", "num_tpu_ops", "num_cpu_ops", "status")
    widths = [max(len(header), max(len(str(row[i])) for row in rows)) for i, header in enumerate(headers)]
    fmt = "  ".join(f"{{:{width}}}" for width in widths)
    print(fmt.format(*headers))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print(fmt.format(*(str(col) for col in row[:5])))
        if row[4] != "OK":
            print(f"    in {row[5]}")

    return 1 if any_bad else 0


if __name__ == "__main__":
    sys.exit(main())
