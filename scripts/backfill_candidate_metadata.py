"""Backfill task-level fields into per-candidate metadata.json files from Student B.

Student B's metadata.json files contain partition-level fields only.
HybridRunner requires additional task-level fields (task_name, task_type,
model_path, input_tensor, output_tensors, data_loader, input_shape, resize,
input_dtype, input_normalization, and data paths).

Run once after merging Student B's artifacts. Re-running is safe (idempotent).

Usage:
    python3 scripts/backfill_candidate_metadata.py
    python3 scripts/backfill_candidate_metadata.py --dry-run
    python3 scripts/backfill_candidate_metadata.py --model dlc
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Task-level fields to merge into each model family's candidates.
# Only adds fields that are MISSING — never overwrites existing values.
TASK_FIELDS: dict[str, dict] = {
    "dlc_resnet50": {
        "task_name": "task_a_dlc",
        "task_type": "pose_estimation",
        "model_path": "snapshot-1000.pb",
        "input_tensor": "Placeholder:0",
        "output_tensors": ["concat_1:0"],
        "data_loader": "ap10k_pose",
        "input_shape": [1, 320, 320, 3],
        "resize": [320, 320],
        "input_dtype": "float32",
        "input_normalization": "none",
        "images_dir": "data/task_a/data/ap-10k/data",
        "annotations_path": "data/task_a/data/ap-10k/annotations/ap10k-val-split1.json",
        "require_single_instance": True,
        "annotation_strategy": "largest_instance",
    },
    # SSD bypasses the Preprocessor (TensorArrayV3 ops are not TFLite-convertible).
    # The input tensor for both run_full_cpu and the TPU prefix is the post-preprocessor
    # entry point. float32 in [-1, 1]; input_normalization="ssd" applies pixel/128-1.
    "ssd_mobilenet_v2": {
        "task_name": "task_b_detection",
        "task_type": "object_detection",
        "model_path": "data/task_b/models/frozen_inference_graph.pb",
        "input_tensor": "FeatureExtractor/MobilenetV2/MobilenetV2/input:0",
        "output_tensors": [
            "detection_boxes:0",
            "detection_scores:0",
            "detection_classes:0",
            "num_detections:0",
        ],
        "data_loader": "coco_images",
        "input_shape": [1, 300, 300, 3],
        "resize": [300, 300],
        "input_dtype": "float32",
        "input_normalization": "ssd",
        "images_dir": "data/task_b/data/val2017",
        "annotations_path": "data/task_b/data/annotations/instances_val2017.json",
    },
    "deeplab_v3_mobilenetv2": {
        "task_name": "task_c_segmentation",
        "task_type": "semantic_segmentation",
        "model_path": "data/task_c/models/frozen_inference_graph.pb",
        "input_tensor": "ImageTensor:0",
        "output_tensors": ["SemanticPredictions:0"],
        "data_loader": "voc_segmentation",
        "input_shape": [1, 513, 513, 3],
        "resize": [513, 513],
        "input_dtype": "uint8",
        "input_normalization": "none",
        "images_dir": "data/task_c/data/pascal-voc-2012-DatasetNinja/val/img",
        "annotations_dir": "data/task_c/data/pascal-voc-2012-DatasetNinja/val/ann",
        "meta_path": "data/task_c/data/pascal-voc-2012-DatasetNinja/meta.json",
    },
}

MODEL_FILTER_PREFIXES: dict[str, str] = {
    "dlc": "dlc_",
    "ssd": "ssd_",
    "deeplab": "deeplab_",
}


def backfill_metadata(candidate_dir: Path, dry_run: bool = False) -> bool:
    """Add missing task-level fields to a candidate's metadata.json.

    Returns True if the file was (or would be) updated.
    """
    metadata_path = candidate_dir / "metadata.json"
    if not metadata_path.exists():
        print(f"  SKIP  (no metadata.json)  {candidate_dir.name}")
        return False

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_key = metadata.get("model")
    if model_key not in TASK_FIELDS:
        print(f"  SKIP  (unknown model '{model_key}')  {candidate_dir.name}")
        return False

    task_fields = TASK_FIELDS[model_key]
    added: list[str] = []
    for key, value in task_fields.items():
        if key not in metadata:
            metadata[key] = value
            added.append(key)

    if not added:
        print(f"  OK    (already complete)  {candidate_dir.name}")
        return False

    label = "DRY-RUN" if dry_run else "UPDATED"
    print(f"  {label}  {candidate_dir.name}: +{', '.join(added)}")
    if not dry_run:
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing.")
    parser.add_argument(
        "--model",
        choices=["dlc", "ssd", "deeplab"],
        default=None,
        help="Only backfill candidates for one model family (default: all).",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=str(REPO_ROOT / "artifacts"),
        help="Root artifacts directory (default: artifacts/).",
    )
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    prefix_filter = MODEL_FILTER_PREFIXES.get(args.model) if args.model else None

    candidate_dirs = sorted(
        p
        for p in artifacts_dir.iterdir()
        if p.is_dir()
        and (p / "metadata.json").exists()
        and (prefix_filter is None or p.name.startswith(prefix_filter))
    )

    if not candidate_dirs:
        print(f"No candidate directories found under {artifacts_dir}")
        return

    print(f"Scanning {len(candidate_dirs)} candidate(s) under {artifacts_dir}\n")
    updated = sum(backfill_metadata(d, dry_run=args.dry_run) for d in candidate_dirs)
    verb = "Would update" if args.dry_run else "Updated"
    print(f"\n{verb} {updated}/{len(candidate_dirs)} candidates.")


if __name__ == "__main__":
    main()
