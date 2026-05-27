"""Walk artifacts/ and validate each metadata.json against the
Partition Candidate schema documented in
docs/student_project_plan.md section 4 (with the additions from
docs/student_b_weeks_3_6.md).

Exits non-zero if any metadata.json is invalid or missing required
fields. Designed to be plain Python — no jsonschema dependency.
"""
import argparse
import json
import os
import sys


REQUIRED_FIELDS = [
    "model",
    "partition_id",
    "tpu_output_tensors",
    "cpu_input_tensors",
    "tpu_tflite_path",
    "cpu_graph_path",
    "num_tpu_ops",
    "num_cpu_ops",
    "boundary_tensor_shapes",
]

OPTIONAL_FIELDS = [
    "boundary_bandwidth_bytes",
    "has_skip_crossing",
    "quant_mode",
]

ALL_FIELDS = set(REQUIRED_FIELDS) | set(OPTIONAL_FIELDS)


def find_metadata_files(root):
    out = []
    for dirpath, _, filenames in os.walk(root):
        if "metadata.json" in filenames:
            out.append(os.path.join(dirpath, "metadata.json"))
    return sorted(out)


def validate(path):
    """Return (status_str, partition_id, model, num_tpu_ops,
    num_cpu_ops). status_str is 'OK' or 'missing field X' / 'parse
    error'."""
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as e:
        return f"parse error: {e}", "?", "?", "?", "?"

    if not isinstance(data, dict):
        return "not a JSON object", "?", "?", "?", "?"

    missing = [k for k in REQUIRED_FIELDS if k not in data]
    pid = data.get("partition_id", "?")
    model = data.get("model", "?")
    n_tpu = data.get("num_tpu_ops", "?")
    n_cpu = data.get("num_cpu_ops", "?")

    if missing:
        return f"missing field {missing[0]}", pid, model, n_tpu, n_cpu

    # Type checks for required fields
    type_errors = []
    if not isinstance(data["tpu_output_tensors"], list):
        type_errors.append("tpu_output_tensors not a list")
    if not isinstance(data["cpu_input_tensors"], list):
        type_errors.append("cpu_input_tensors not a list")
    if not isinstance(data["num_tpu_ops"], int):
        type_errors.append("num_tpu_ops not an int")
    if not isinstance(data["num_cpu_ops"], int):
        type_errors.append("num_cpu_ops not an int")
    if not isinstance(data["boundary_tensor_shapes"], list):
        type_errors.append("boundary_tensor_shapes not a list")
    if type_errors:
        return type_errors[0], pid, model, n_tpu, n_cpu

    return "OK", pid, model, n_tpu, n_cpu


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="artifacts",
                   help="Directory to walk for metadata.json files")
    args = p.parse_args()

    files = find_metadata_files(args.root)
    if not files:
        print(f"No metadata.json files found under {args.root}")
        # Not an error — nothing to validate yet.
        return 0

    rows = []
    any_bad = False
    for f in files:
        status, pid, model, n_tpu, n_cpu = validate(f)
        rows.append((pid, model, n_tpu, n_cpu, status, f))
        if status != "OK":
            any_bad = True

    # Print table
    headers = ("partition_id", "model", "num_tpu_ops", "num_cpu_ops",
               "status")
    widths = [max(len(h), max(len(str(r[i])) for r in rows))
              for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:{w}}}" for w in widths)
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt.format(*(str(c) for c in r[:5])))
        if r[4] != "OK":
            print(f"    in {r[5]}")

    return 1 if any_bad else 0


if __name__ == "__main__":
    sys.exit(main())
