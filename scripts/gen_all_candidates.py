"""Batch-generate per-candidate TPU+CPU SavedModels and metadata.json.

Runs gen_tflite.py (TPU side) and extract_cpu_subgraph.py (CPU side) for every
partition candidate across DLC, SSD MobileNet V2, and DeepLab V3, then writes
a metadata.json that validates against the Partition Candidate schema
(docs/student_project_plan.md §4 + docs/student_b_weeks_3_6.md).

Usage (from project root):
    python3 scripts/gen_all_candidates.py
    python3 scripts/gen_all_candidates.py --dry-run        # print commands only
    python3 scripts/gen_all_candidates.py --model dlc      # one model only
    python3 scripts/gen_all_candidates.py --skip-existing  # skip already-extracted dirs
"""
import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Candidate registry
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent.resolve()
ARTIFACTS = ROOT / "artifacts"

# Each candidate dict keys:
#   model_name   : slug for metadata "model" field
#   model_file   : path to frozen .pb (relative to ROOT)
#   input_tensor : name of input placeholder in the frozen graph
#   input_shape  : list[int] — fixed shape for the TPU subgraph input
#   output_tensors : list[str] — boundary tensor names (without :0)
#   final_outputs  : list[str] — final output tensor(s) of the full graph
#   partition_id : str — kebab-case slug
#   num_tpu_ops  : int — from partition_points_*.md
#   num_cpu_ops  : int — from partition_points_*.md
#   boundary_tensor_shapes : list[list[int]] — one per boundary tensor
#   has_skip_crossing : bool
#   bandwidth_kib : float  (informational, used to compute bandwidth_bytes)

DLC_MODEL = "snapshot-1000.pb"
DLC_INPUT = "Placeholder"
DLC_SHAPE = [1, 320, 320, 3]
DLC_FINALS = ["concat_1"]

SSD_MODEL = "data/task_b/models/frozen_inference_graph.pb"
# Use the post-Preprocessor float32 input to MobileNetV2 instead of image_tensor.
# The SSD Preprocessor cluster contains TensorArrayV3 ops that TFLite cannot
# convert; bypassing it lets us quantize the backbone subgraphs normally.
# Calibration images must be normalized to [-1, 1] (pixel/128 - 1) to match
# what the real Preprocessor produces.  See convert.py --calib-normalize ssd.
SSD_INPUT = "FeatureExtractor/MobilenetV2/MobilenetV2/input"
SSD_SHAPE = [1, 300, 300, 3]   # float32, range [-1, 1]
SSD_FINALS = ["detection_boxes", "detection_scores", "detection_classes", "num_detections"]

DEEPLAB_MODEL = "data/task_c/models/frozen_inference_graph.pb"
DEEPLAB_INPUT = "ImageTensor"
DEEPLAB_SHAPE = [1, 513, 513, 3]
DEEPLAB_FINALS = ["SemanticPredictions"]

CANDIDATES = [
    # ------------------------------------------------------------------ DLC
    dict(
        model_name="dlc_resnet50",
        model_file=DLC_MODEL,
        input_tensor=DLC_INPUT,
        input_shape=DLC_SHAPE,
        output_tensors=["resnet_v1_50/block1/unit_3/bottleneck_v1/Relu"],
        final_outputs=DLC_FINALS,
        partition_id="split_after_block1",
        num_tpu_ops=154,
        num_cpu_ops=933,
        boundary_tensor_shapes=[[1, 40, 40, 256]],
        has_skip_crossing=False,
        bandwidth_kib=400.0,
    ),
    dict(
        model_name="dlc_resnet50",
        model_file=DLC_MODEL,
        input_tensor=DLC_INPUT,
        input_shape=DLC_SHAPE,
        output_tensors=["resnet_v1_50/block2/unit_4/bottleneck_v1/Relu"],
        final_outputs=DLC_FINALS,
        partition_id="split_after_block2",
        num_tpu_ops=329,
        num_cpu_ops=758,
        boundary_tensor_shapes=[[1, 20, 20, 512]],
        has_skip_crossing=False,
        bandwidth_kib=200.0,
    ),
    dict(
        model_name="dlc_resnet50",
        model_file=DLC_MODEL,
        input_tensor=DLC_INPUT,
        input_shape=DLC_SHAPE,
        output_tensors=["resnet_v1_50/block3/unit_6/bottleneck_v1/Relu"],
        final_outputs=DLC_FINALS,
        partition_id="split_after_block3",
        num_tpu_ops=581,
        num_cpu_ops=506,
        boundary_tensor_shapes=[[1, 20, 20, 1024]],
        has_skip_crossing=False,
        bandwidth_kib=400.0,
    ),
    dict(
        model_name="dlc_resnet50",
        model_file=DLC_MODEL,
        input_tensor=DLC_INPUT,
        input_shape=DLC_SHAPE,
        output_tensors=["resnet_v1_50/block4/unit_1/bottleneck_v1/Relu"],
        final_outputs=DLC_FINALS,
        partition_id="split_in_block4_unit1",
        num_tpu_ops=705,
        num_cpu_ops=382,
        boundary_tensor_shapes=[[1, 20, 20, 2048]],
        has_skip_crossing=False,
        bandwidth_kib=800.0,
    ),
    dict(
        model_name="dlc_resnet50",
        model_file=DLC_MODEL,
        input_tensor=DLC_INPUT,
        input_shape=DLC_SHAPE,
        output_tensors=["resnet_v1_50/block4/unit_3/bottleneck_v1/Relu"],
        final_outputs=DLC_FINALS,
        partition_id="split_after_block4",
        num_tpu_ops=929,
        num_cpu_ops=158,
        boundary_tensor_shapes=[[1, 20, 20, 2048]],
        has_skip_crossing=False,
        bandwidth_kib=800.0,
    ),
    dict(
        model_name="dlc_resnet50",
        model_file=DLC_MODEL,
        input_tensor=DLC_INPUT,
        input_shape=DLC_SHAPE,
        output_tensors=[
            "pose/part_pred/block4/conv2d_transpose",
            "pose/locref_pred/block4/conv2d_transpose",
        ],
        final_outputs=DLC_FINALS,
        partition_id="split_at_conv2d_transpose",
        num_tpu_ops=973,
        num_cpu_ops=115,
        boundary_tensor_shapes=[[1, 40, 40, 39], [1, 40, 40, 78]],
        has_skip_crossing=False,
        bandwidth_kib=182.8,
    ),
    dict(
        model_name="dlc_resnet50",
        model_file=DLC_MODEL,
        input_tensor=DLC_INPUT,
        input_shape=DLC_SHAPE,
        output_tensors=[
            "pose/part_pred/block4/BiasAdd",
            "pose/locref_pred/block4/BiasAdd",
        ],
        final_outputs=DLC_FINALS,
        partition_id="split_at_bias_add",
        num_tpu_ops=979,
        num_cpu_ops=109,
        boundary_tensor_shapes=[[1, 40, 40, 39], [1, 40, 40, 78]],
        has_skip_crossing=False,
        bandwidth_kib=182.8,
    ),
    # ------------------------------------------------------------------ SSD
    dict(
        model_name="ssd_mobilenet_v2",
        model_file=SSD_MODEL,
        input_tensor=SSD_INPUT,
        input_shape=SSD_SHAPE,
        output_tensors=["FeatureExtractor/MobilenetV2/expanded_conv_5/output"],
        final_outputs=SSD_FINALS,
        partition_id="split_after_expanded_conv_5",
        num_tpu_ops=441,
        num_cpu_ops=7553,
        boundary_tensor_shapes=[[1, 38, 38, 32]],
        has_skip_crossing=True,
        bandwidth_kib=45.1,
    ),
    dict(
        model_name="ssd_mobilenet_v2",
        model_file=SSD_MODEL,
        input_tensor=SSD_INPUT,
        input_shape=SSD_SHAPE,
        output_tensors=["FeatureExtractor/MobilenetV2/expanded_conv_10/output"],
        final_outputs=SSD_FINALS,
        partition_id="split_after_expanded_conv_10",
        num_tpu_ops=764,
        num_cpu_ops=7230,
        boundary_tensor_shapes=[[1, 19, 19, 96]],
        has_skip_crossing=True,
        bandwidth_kib=34.0,
    ),
    dict(
        model_name="ssd_mobilenet_v2",
        model_file=SSD_MODEL,
        input_tensor=SSD_INPUT,
        input_shape=SSD_SHAPE,
        output_tensors=["FeatureExtractor/MobilenetV2/expanded_conv_13/output"],
        final_outputs=SSD_FINALS,
        partition_id="split_after_expanded_conv_13",
        num_tpu_ops=958,
        num_cpu_ops=7934,
        boundary_tensor_shapes=[[1, 10, 10, 160]],
        has_skip_crossing=True,
        bandwidth_kib=16.0,
    ),
    dict(
        model_name="ssd_mobilenet_v2",
        model_file=SSD_MODEL,
        input_tensor=SSD_INPUT,
        input_shape=SSD_SHAPE,
        output_tensors=["FeatureExtractor/MobilenetV2/Conv_1/Relu6"],
        final_outputs=SSD_FINALS,
        partition_id="split_after_conv_1",
        num_tpu_ops=1172,
        num_cpu_ops=7720,
        boundary_tensor_shapes=[[1, 10, 10, 1280]],
        has_skip_crossing=True,
        bandwidth_kib=125.0,
    ),
    dict(
        model_name="ssd_mobilenet_v2",
        model_file=SSD_MODEL,
        input_tensor=SSD_INPUT,
        input_shape=SSD_SHAPE,
        output_tensors=[
            "BoxPredictor_0/ClassPredictor/BiasAdd",
            "BoxPredictor_0/BoxEncodingPredictor/BiasAdd",
            "BoxPredictor_1/ClassPredictor/BiasAdd",
            "BoxPredictor_1/BoxEncodingPredictor/BiasAdd",
            "BoxPredictor_2/ClassPredictor/BiasAdd",
            "BoxPredictor_2/BoxEncodingPredictor/BiasAdd",
            "BoxPredictor_3/ClassPredictor/BiasAdd",
            "BoxPredictor_3/BoxEncodingPredictor/BiasAdd",
            "BoxPredictor_4/ClassPredictor/BiasAdd",
            "BoxPredictor_4/BoxEncodingPredictor/BiasAdd",
            "BoxPredictor_5/ClassPredictor/BiasAdd",
            "BoxPredictor_5/BoxEncodingPredictor/BiasAdd",
        ],
        final_outputs=SSD_FINALS,
        partition_id="split_at_box_predictor_biasadds",
        num_tpu_ops=1484,
        num_cpu_ops=7915,
        # shapes per level: class [1,N,N,{3,6}*91] box [1,N,N,{3,6}*4]
        # summarised here as approximate values from stride analysis
        boundary_tensor_shapes=[
            [1, 19, 19, 273], [1, 19, 19, 12],
            [1, 10, 10, 546], [1, 10, 10, 24],
            [1, 5,  5,  546], [1, 5,  5,  24],
            [1, 3,  3,  546], [1, 3,  3,  24],
            [1, 2,  2,  546], [1, 2,  2,  24],
            [1, 1,  1,  546], [1, 1,  1,  24],
        ],
        has_skip_crossing=True,
        bandwidth_kib=178.0,
    ),
    dict(
        model_name="ssd_mobilenet_v2",
        model_file=SSD_MODEL,
        input_tensor=SSD_INPUT,
        input_shape=SSD_SHAPE,
        output_tensors=["Squeeze", "concat_1"],
        final_outputs=SSD_FINALS,
        partition_id="split_before_postprocessor",
        num_tpu_ops=1573,
        num_cpu_ops=6422,
        boundary_tensor_shapes=[[1, 1917, 4], [1, 1917, 91]],
        has_skip_crossing=True,
        bandwidth_kib=178.0,
    ),
    # --------------------------------------------------------------- DeepLab
    dict(
        model_name="deeplab_v3_mobilenetv2",
        model_file=DEEPLAB_MODEL,
        input_tensor=DEEPLAB_INPUT,
        input_shape=DEEPLAB_SHAPE,
        output_tensors=["MobilenetV2/expanded_conv_5/output"],
        final_outputs=DEEPLAB_FINALS,
        partition_id="split_after_expanded_conv_5",
        num_tpu_ops=352,
        num_cpu_ops=620,
        boundary_tensor_shapes=[[1, 65, 65, 32]],
        has_skip_crossing=True,
        bandwidth_kib=132.0,
    ),
    dict(
        model_name="deeplab_v3_mobilenetv2",
        model_file=DEEPLAB_MODEL,
        input_tensor=DEEPLAB_INPUT,
        input_shape=DEEPLAB_SHAPE,
        output_tensors=["MobilenetV2/expanded_conv_10/output"],
        final_outputs=DEEPLAB_FINALS,
        partition_id="split_after_expanded_conv_10",
        num_tpu_ops=594,
        num_cpu_ops=378,
        boundary_tensor_shapes=[[1, 65, 65, 96]],
        has_skip_crossing=True,
        bandwidth_kib=396.0,
    ),
    dict(
        model_name="deeplab_v3_mobilenetv2",
        model_file=DEEPLAB_MODEL,
        input_tensor=DEEPLAB_INPUT,
        input_shape=DEEPLAB_SHAPE,
        output_tensors=["MobilenetV2/expanded_conv_13/output"],
        final_outputs=DEEPLAB_FINALS,
        partition_id="split_after_expanded_conv_13",
        num_tpu_ops=743,
        num_cpu_ops=229,
        boundary_tensor_shapes=[[1, 65, 65, 160]],
        has_skip_crossing=True,
        bandwidth_kib=660.0,
    ),
    dict(
        model_name="deeplab_v3_mobilenetv2",
        model_file=DEEPLAB_MODEL,
        input_tensor=DEEPLAB_INPUT,
        input_shape=DEEPLAB_SHAPE,
        output_tensors=["MobilenetV2/expanded_conv_16/output"],
        final_outputs=DEEPLAB_FINALS,
        partition_id="split_after_expanded_conv_16",
        num_tpu_ops=892,
        num_cpu_ops=80,
        boundary_tensor_shapes=[[1, 65, 65, 320]],
        has_skip_crossing=True,
        bandwidth_kib=1320.0,
    ),
    dict(
        model_name="deeplab_v3_mobilenetv2",
        model_file=DEEPLAB_MODEL,
        input_tensor=DEEPLAB_INPUT,
        input_shape=DEEPLAB_SHAPE,
        output_tensors=["concat_projection/Relu"],
        final_outputs=DEEPLAB_FINALS,
        partition_id="split_after_aspp",
        num_tpu_ops=936,
        num_cpu_ops=36,
        boundary_tensor_shapes=[[1, 65, 65, 256]],
        has_skip_crossing=True,
        bandwidth_kib=1056.0,
    ),
    dict(
        model_name="deeplab_v3_mobilenetv2",
        model_file=DEEPLAB_MODEL,
        input_tensor=DEEPLAB_INPUT,
        input_shape=DEEPLAB_SHAPE,
        output_tensors=["logits/semantic/BiasAdd"],
        final_outputs=DEEPLAB_FINALS,
        partition_id="split_after_logits",
        num_tpu_ops=943,
        num_cpu_ops=29,
        boundary_tensor_shapes=[[1, 65, 65, 21]],
        has_skip_crossing=True,
        bandwidth_kib=86.0,
    ),
    dict(
        model_name="deeplab_v3_mobilenetv2",
        model_file=DEEPLAB_MODEL,
        input_tensor=DEEPLAB_INPUT,
        input_shape=DEEPLAB_SHAPE,
        output_tensors=["ResizeBilinear_3"],
        final_outputs=DEEPLAB_FINALS,
        partition_id="split_after_resize",
        num_tpu_ops=951,
        num_cpu_ops=21,
        boundary_tensor_shapes=[[1, 513, 513, 21]],
        has_skip_crossing=True,
        bandwidth_kib=5398.0,
    ),
]

# Group by model name prefix for filtering
MODEL_GROUPS = {
    "dlc": "dlc_resnet50",
    "ssd": "ssd_mobilenet_v2",
    "deeplab": "deeplab_v3_mobilenetv2",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def artifact_dir(c: dict) -> Path:
    model_prefix = c["model_name"].split("_")[0]
    return ARTIFACTS / f"{model_prefix}_{c['partition_id']}"


def run(cmd: list, dry_run: bool, desc: str = "") -> bool:
    """Run a subprocess; return True on success."""
    if desc:
        print(f"  {desc}")
    print("  $", " ".join(str(x) for x in cmd))
    if dry_run:
        return True
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        print(f"  ERROR: command exited with code {r.returncode}")
        return False
    return True


def write_metadata(c: dict, adir: Path) -> None:
    tpu_out_tensors = [t + ":0" for t in c["output_tensors"]]
    meta = {
        "model": c["model_name"],
        "partition_id": c["partition_id"],
        "tpu_output_tensors": tpu_out_tensors,
        "cpu_input_tensors": tpu_out_tensors,
        "tpu_tflite_path": str(adir / "tpu_int8_pure.tflite"),
        "cpu_graph_path": str(adir / "cpu_savedmodel"),
        "num_tpu_ops": c["num_tpu_ops"],
        "num_cpu_ops": c["num_cpu_ops"],
        "boundary_tensor_shapes": c["boundary_tensor_shapes"],
        # optional but useful fields
        "boundary_bandwidth_bytes": int(c["bandwidth_kib"] * 1024),
        "has_skip_crossing": c["has_skip_crossing"],
        "quant_mode": None,  # filled in after edgetpu_compiler succeeds
    }
    path = adir / "metadata.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running them")
    parser.add_argument("--model", choices=list(MODEL_GROUPS) + ["all"],
                        default="all",
                        help="Restrict to one model family (default: all)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip candidates whose tpu_savedmodel/ already exists")
    args = parser.parse_args()

    # Filter candidate list
    target_model = None if args.model == "all" else MODEL_GROUPS[args.model]
    work = [c for c in CANDIDATES
            if target_model is None or c["model_name"] == target_model]

    print(f"Processing {len(work)} candidate(s) "
          f"({'dry run' if args.dry_run else 'live'})\n")

    failures: list = []

    for i, c in enumerate(work, 1):
        adir = artifact_dir(c)
        tpu_sm = adir / "tpu_savedmodel"
        cpu_sm = adir / "cpu_savedmodel"
        pid = c["partition_id"]
        model_name = c["model_name"]
        print(f"[{i}/{len(work)}] {model_name} / {pid}")
        print(f"  artifact dir: {adir}")

        if args.skip_existing and tpu_sm.exists():
            print("  tpu_savedmodel/ already exists — skipping (--skip-existing)")
            # Still write metadata.json if it's missing
            if not (adir / "metadata.json").exists():
                if not args.dry_run:
                    write_metadata(c, adir)
            print()
            continue

        # Create artifact directory
        if not args.dry_run:
            adir.mkdir(parents=True, exist_ok=True)

        # 1. TPU subgraph (gen_tflite.py)
        boundary_str = ",".join(c["output_tensors"])
        shape_str = ",".join(str(x) for x in c["input_shape"])
        ok = run(
            [sys.executable, "gen_tflite.py",
             "--model", str(ROOT / c["model_file"]),
             "--input-tensor", c["input_tensor"],
             "--input-shape", shape_str,
             "--output-tensors", boundary_str,
             "--output-dir", str(tpu_sm)],
            dry_run=args.dry_run,
            desc="TPU subgraph",
        )
        if not ok:
            failures.append((pid, "gen_tflite.py failed"))
            print()
            continue

        # 2. CPU subgraph (extract_cpu_subgraph.py)
        boundary_fq = ",".join(t + ":0" for t in c["output_tensors"])
        finals_str = ",".join(c["final_outputs"])
        ok = run(
            [sys.executable, "extract_cpu_subgraph.py",
             "--model", str(ROOT / c["model_file"]),
             "--boundary-tensors", boundary_fq,
             "--final-output-tensor", finals_str,
             "--output-dir", str(cpu_sm)],
            dry_run=args.dry_run,
            desc="CPU subgraph",
        )
        if not ok:
            failures.append((pid, "extract_cpu_subgraph.py failed"))
            print()
            continue

        # 3. metadata.json
        if not args.dry_run:
            write_metadata(c, adir)
        else:
            print(f"  (dry-run) would write {adir}/metadata.json")

        print()

    # Summary
    total = len(work)
    succeeded = total - len(failures)
    print(f"Done: {succeeded}/{total} candidates extracted successfully.")
    if failures:
        print("Failures:")
        for pid, reason in failures:
            print(f"  {pid}: {reason}")
        sys.exit(1)


if __name__ == "__main__":
    main()
