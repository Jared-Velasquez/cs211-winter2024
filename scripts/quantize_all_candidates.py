"""Batch-quantize every per-candidate TPU SavedModel to TFLite.

Runs convert.py in both int8_pure and int_fallback modes for every
candidate.  Writes results to a JSON report and prints a summary table.

Usage (from project root):
    python3 scripts/quantize_all_candidates.py
    python3 scripts/quantize_all_candidates.py --model dlc
    python3 scripts/quantize_all_candidates.py --mode int8_pure
    python3 scripts/quantize_all_candidates.py --skip-existing

Calibration sources used:
    DLC     — data/task_a/data/ap-10k/data/  (AP-10K animal JPEG images)
    SSD     — data/task_b/data/val2017/        (COCO val2017 images)
    DeepLab — data/task_c/data/pascal-voc-2012-DatasetNinja/val/
                                               (Pascal VOC 2012 val images)
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
ARTIFACTS = ROOT / "artifacts"
REPORT_PATH = ROOT / "artifacts" / "quantize_report.json"

# Calibration sources (image directories)
CALIB = {
    "dlc_resnet50": {
        "dir": str(ROOT / "data/task_a/data/ap-10k/data"),
        "w": 320, "h": 320,
    },
    "ssd_mobilenet_v2": {
        "dir": str(ROOT / "data/task_b/data/val2017"),
        "w": 300, "h": 300,
        # SSD input bypasses the Preprocessor; images must be in [-1,1]
        "normalize": "ssd",
    },
    "deeplab_v3_mobilenetv2": {
        "dir": str(ROOT / "data/task_c/data/pascal-voc-2012-DatasetNinja/val/img"),
        "w": 513, "h": 513,
    },
}

MODES = ("int8_pure", "int_fallback")


def find_candidates() -> list:
    """Return list of (artifact_dir, model_name) for all candidates."""
    out = []
    for d in sorted(ARTIFACTS.iterdir()):
        if not d.is_dir():
            continue
        meta_path = d / "metadata.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        if "model" not in meta:
            continue
        out.append((d, meta["model"]))
    return out


def run_convert(savedmodel: str, mode: str, calib: dict,
                output: str, dry_run: bool, max_calib: int = 200) -> tuple:
    """Run convert.py; return (success: bool, stderr: str)."""
    cmd = [
        sys.executable, "convert.py",
        "-m", savedmodel,
        "-O", mode,
        "-r", calib["dir"],
        "-w", str(calib["w"]),
        "-t", str(calib["h"]),
        "-o", output,
        "--max-calib", str(max_calib),
    ]
    if calib.get("normalize"):
        cmd += ["--calib-normalize", calib["normalize"]]
    print("    $", " ".join(str(x) for x in cmd))
    if dry_run:
        return True, ""
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.stdout:
        for line in r.stdout.splitlines()[-3:]:
            print("   ", line)
    if r.returncode != 0:
        print("    FAILED:")
        for line in r.stderr.splitlines()[-8:]:
            print("   ", line)
    return r.returncode == 0, r.stderr


def update_metadata_quant_mode(meta_path: Path, mode: str) -> None:
    with open(meta_path) as f:
        meta = json.load(f)
    meta["quant_mode"] = mode
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", choices=["dlc", "ssd", "deeplab", "all"],
                        default="all")
    parser.add_argument("--mode", choices=["int8_pure", "int_fallback", "both"],
                        default="both")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip candidates whose tpu_int8.tflite already exists")
    parser.add_argument("--max-calib", type=int, default=200,
                        help="Max calibration samples (default 200; use 50 for large models to limit runtime)")
    args = parser.parse_args()

    model_filter = {
        "dlc": "dlc_resnet50",
        "ssd": "ssd_mobilenet_v2",
        "deeplab": "deeplab_v3_mobilenetv2",
        "all": None,
    }[args.model]

    modes = [args.mode] if args.mode != "both" else list(MODES)

    candidates = find_candidates()
    if model_filter:
        candidates = [(d, m) for d, m in candidates if m == model_filter]

    print(f"Quantizing {len(candidates)} candidate(s), modes={modes}\n")

    report = {}
    failures = []

    for adir, model_name in candidates:
        pid = adir.name
        print(f"[{pid}]  model={model_name}")

        calib = CALIB.get(model_name)
        if not calib:
            print(f"  SKIP — no calibration config for model '{model_name}'")
            continue
        if not os.path.isdir(calib["dir"]):
            print(f"  SKIP — calibration dir not found: {calib['dir']}")
            continue

        tpu_sm = str(adir / "tpu_savedmodel")
        if not os.path.isdir(tpu_sm):
            print(f"  SKIP — tpu_savedmodel/ missing")
            continue

        meta_path = adir / "metadata.json"
        report[pid] = {}
        quant_succeeded = None

        for mode in modes:
            out_path = str(adir / f"tpu_{mode}.tflite")
            if args.skip_existing and os.path.exists(out_path):
                print(f"  [{mode}] already exists — skipping")
                report[pid][mode] = "skipped"
                if quant_succeeded is None and os.path.exists(out_path):
                    quant_succeeded = mode
                continue

            print(f"  [{mode}]")
            ok, stderr = run_convert(tpu_sm, mode, calib, out_path,
                                     args.dry_run, args.max_calib)
            report[pid][mode] = "ok" if ok else f"FAILED: {stderr[-200:]}"
            if ok and quant_succeeded is None:
                quant_succeeded = mode
            if not ok:
                failures.append((pid, mode))

        # Update metadata quant_mode to the first mode that succeeded
        if quant_succeeded and not args.dry_run:
            update_metadata_quant_mode(meta_path, quant_succeeded)
            print(f"  metadata.quant_mode = {quant_succeeded!r}")

        print()

    # Save JSON report
    if not args.dry_run:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(REPORT_PATH, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved: {REPORT_PATH}")

    # Print summary table
    print("\n=== Quantization summary ===")
    col_w = 34
    header = f"{'candidate':<{col_w}}" + "".join(f"  {m:<14}" for m in modes)
    print(header)
    print("-" * len(header))
    for pid, results in sorted(report.items()):
        row = f"{pid:<{col_w}}"
        for mode in modes:
            status = results.get(mode, "-")
            cell = "OK" if status in ("ok", "skipped") else "FAILED"
            row += f"  {cell:<14}"
        print(row)

    total = len(report) * len(modes)
    n_fail = len(failures)
    print(f"\n{total - n_fail}/{total} quant runs succeeded.")
    if failures:
        print("Failures:")
        for pid, mode in failures:
            print(f"  {pid} [{mode}]")
        sys.exit(1)


if __name__ == "__main__":
    main()
