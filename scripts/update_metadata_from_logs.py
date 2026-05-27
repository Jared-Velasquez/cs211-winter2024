#!/usr/bin/env python3
"""Parse edgetpu_compiler logs and update metadata.json for all 20 candidates.

For 16 compiled candidates (log present):
  - Parses operator table → mapped_count, total_count, ops_breakdown
  - Sets edgetpu_compiled: true
  - Sets tpu_edgetpu_path: "artifacts/<id>/tpu_int8_pure_edgetpu.tflite" (relative)
  - Sets tpu_ops_mapped_edgetpu: "<mapped>/<total>"
  - Adds edgetpu_ops_breakdown: [{op, count, mapped}, ...]
  - Sets edgetpu_rejection_reason: null

For 4 DeepLab failed candidates (no log):
  - Sets edgetpu_compiled: false
  - Sets tpu_edgetpu_path: null
  - Sets tpu_ops_mapped_edgetpu: null
  - Sets edgetpu_ops_breakdown: null
  - Sets edgetpu_rejection_reason: "Compilation failed due to large activation tensors in model"
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPILED_DIR = os.path.join(ROOT, "compiled_artifacts")
ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")

FAILED_DEEPLAB = {
    "deeplab_split_after_expanded_conv_16",
    "deeplab_split_after_aspp",
    "deeplab_split_after_logits",
    "deeplab_split_after_resize",
}

FAILURE_REASON = "Compilation failed due to large activation tensors in model"

# Map status strings from log to classification
def classify_status(status_str: str) -> bool:
    """Return True if mapped to Edge TPU, False otherwise."""
    return "Mapped to Edge TPU" in status_str


def parse_log(log_path: str) -> tuple[int, int, list[dict]]:
    """Parse a compiler log file.

    Returns:
        (mapped_count, total_count, ops_breakdown)
        ops_breakdown = [{op: str, count: int, mapped: bool}, ...]
    """
    with open(log_path) as f:
        content = f.read()

    # Find the operator table section
    # Format: "Operator                       Count      Status"
    # followed by rows like "CONV_2D                        11         Mapped to Edge TPU"
    lines = content.splitlines()

    # Find header line
    header_idx = None
    for i, line in enumerate(lines):
        if "Operator" in line and "Count" in line and "Status" in line:
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(f"No operator table found in {log_path}")

    ops_breakdown = []
    mapped_count = 0
    total_count = 0

    for line in lines[header_idx + 1:]:
        line = line.strip()
        if not line:
            continue

        # Parse: OPERATOR_NAME   <count>   <status description>
        # The operator name may have underscores but no spaces; count is integer; rest is status
        m = re.match(r'^(\S+)\s+(\d+)\s+(.+)$', line)
        if not m:
            continue

        op_name = m.group(1)
        count = int(m.group(2))
        status_str = m.group(3).strip()
        mapped = classify_status(status_str)

        ops_breakdown.append({
            "op": op_name,
            "count": count,
            "mapped": mapped,
        })

        total_count += count
        if mapped:
            mapped_count += count

    return mapped_count, total_count, ops_breakdown


def update_compiled(candidate_id: str, mapped: int, total: int, breakdown: list[dict]) -> None:
    meta_path = os.path.join(ARTIFACTS_DIR, candidate_id, "metadata.json")
    with open(meta_path) as f:
        data = json.load(f)

    data["edgetpu_compiled"] = True
    data["tpu_edgetpu_path"] = f"artifacts/{candidate_id}/tpu_int8_pure_edgetpu.tflite"
    data["tpu_ops_mapped_edgetpu"] = f"{mapped}/{total}"
    data["edgetpu_ops_breakdown"] = breakdown
    data["edgetpu_rejection_reason"] = None

    with open(meta_path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def update_failed(candidate_id: str) -> None:
    meta_path = os.path.join(ARTIFACTS_DIR, candidate_id, "metadata.json")
    with open(meta_path) as f:
        data = json.load(f)

    data["edgetpu_compiled"] = False
    data["tpu_edgetpu_path"] = None
    data["tpu_ops_mapped_edgetpu"] = None
    data["edgetpu_ops_breakdown"] = None
    data["edgetpu_rejection_reason"] = FAILURE_REASON

    with open(meta_path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def main():
    compiled_updated = []
    failed_updated = []
    errors = []

    for candidate_id in sorted(os.listdir(COMPILED_DIR)):
        src_dir = os.path.join(COMPILED_DIR, candidate_id)
        if not os.path.isdir(src_dir):
            continue

        # Check if this is a known failure
        if candidate_id in FAILED_DEEPLAB:
            update_failed(candidate_id)
            failed_updated.append(candidate_id)
            print(f"  ✗ {candidate_id}: marked as failed (no edgetpu)")
            continue

        # Find log file
        log_files = [f for f in os.listdir(src_dir) if f.endswith(".log")]
        if not log_files:
            errors.append(f"{candidate_id}: no .log file found")
            continue

        log_path = os.path.join(src_dir, log_files[0])
        try:
            mapped, total, breakdown = parse_log(log_path)
        except Exception as e:
            errors.append(f"{candidate_id}: log parse error — {e}")
            continue

        update_compiled(candidate_id, mapped, total, breakdown)
        compiled_updated.append((candidate_id, mapped, total))
        print(f"  ✅ {candidate_id}: {mapped}/{total} ops mapped, {len(breakdown)} op types")

    print(f"\n{'='*60}")
    print(f"Updated {len(compiled_updated)} compiled candidates.")
    print(f"Updated {len(failed_updated)} failed candidates.")
    if errors:
        print(f"\nERRORS:")
        for e in errors:
            print(f"  ❌ {e}")
        sys.exit(1)
    print("\nDone — run python3 scripts/validate_metadata.py to verify.")


if __name__ == "__main__":
    main()
