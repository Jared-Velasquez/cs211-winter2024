from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from src.config_utils import get_boundary_tensors, load_config
from src.io_utils import save_json


QUANTIZATION_MODES = [
    {"label": "int_fallback", "convert_opt": "int_fallback"},
    {"label": "int8_pure", "convert_opt": "int8_pure"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full partition experiment for the current manual split: "
            "split once, prepare both quantization modes, and run each compiled Edge TPU variant when available."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/task_a_dlc.json",
        help="Path to the task config JSON.",
    )
    parser.add_argument(
        "--boundary-tensors",
        nargs="+",
        default=None,
        help="Override boundary tensor names for this experiment.",
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate during hybrid runs and representative calibration.",
    )
    parser.add_argument(
        "--force-split",
        action="store_true",
        help="Regenerate the prefix/suffix split artifacts even if they already exist.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only run split/conversion/compilation prep. Do not execute hybrid runs.",
    )
    parser.add_argument(
        "--skip-missing-images",
        action="store_true",
        help="Pass --skip-missing-images through to convert.py for AP-10K representative loading.",
    )
    parser.add_argument(
        "--rebuild-artifacts",
        action="store_true",
        help="Force regeneration of TFLite and compiled Edge TPU artifacts even if they already exist.",
    )
    return parser.parse_args()


def _repo_script(name: str) -> str:
    return str(Path(__file__).resolve().with_name(name))


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _supports_local_compile() -> bool:
    return platform.machine().lower() in {"x86_64", "amd64"} and shutil.which("edgetpu_compiler") is not None


def _tflite_path(config: dict, label: str) -> Path:
    return Path(config["artifacts_dir"]) / f"output_{label}.tflite"


def _compiled_tflite_path(config: dict, label: str) -> Path:
    return Path(config["artifacts_dir"]) / f"output_{label}_edgetpu.tflite"


def _summary_path(config: dict, label: str) -> Path:
    return Path(config["artifacts_dir"]) / f"hybrid_tpu_summary_{label}.json"


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    boundary_tensors = get_boundary_tensors(config, override=args.boundary_tensors)
    artifacts_dir = Path(config["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    split_required = (
        args.force_split
        or args.rebuild_artifacts
        or not Path(config["split_metadata_path"]).exists()
        or not Path(config["suffix_graph_path"]).exists()
        or not Path(config["prefix_saved_model_dir"]).exists()
    )
    if split_required:
        split_command = [sys.executable, _repo_script("split.py"), "--config", args.config]
        if args.force_split:
            split_command.append("--force")
        if args.boundary_tensors:
            split_command.extend(["--boundary-tensors", *args.boundary_tensors])
        _run(split_command)

    compiler_available = _supports_local_compile()
    results: dict[str, dict] = {}

    for mode in QUANTIZATION_MODES:
        label = mode["label"]
        tflite_path = _tflite_path(config, label)
        compiled_path = _compiled_tflite_path(config, label)

        mode_result = {
            "label": label,
            "convert_opt": mode["convert_opt"],
            "tflite_path": str(tflite_path),
            "compiled_tflite_path": str(compiled_path),
            "compiled_exists": compiled_path.exists(),
            "tflite_exists": tflite_path.exists(),
        }

        if args.rebuild_artifacts or not tflite_path.exists():
            convert_command = [
                sys.executable,
                _repo_script("convert.py"),
                "--config",
                args.config,
                "--model",
                config["prefix_saved_model_dir"],
                "--opt",
                mode["convert_opt"],
                "--output",
                str(tflite_path),
            ]
            if args.frame_limit is not None:
                convert_command.extend(["--frame-limit", str(args.frame_limit)])
            if args.skip_missing_images:
                convert_command.append("--skip-missing-images")
            try:
                _run(convert_command)
                mode_result["convert_succeeded"] = True
                mode_result["tflite_exists"] = tflite_path.exists()
            except subprocess.CalledProcessError as exc:
                mode_result["convert_succeeded"] = False
                mode_result["convert_error"] = str(exc)
        else:
            mode_result["convert_succeeded"] = True
            mode_result["reused_existing_tflite"] = True

        if compiler_available and tflite_path.exists() and (args.rebuild_artifacts or not compiled_path.exists()):
            try:
                _run(["edgetpu_compiler", str(tflite_path), "-o", str(artifacts_dir)])
                mode_result["compiled_exists"] = compiled_path.exists()
                mode_result["compiled_locally"] = True
            except subprocess.CalledProcessError as exc:
                mode_result["compiled_locally"] = False
                mode_result["compile_error"] = str(exc)
        elif compiled_path.exists():
            mode_result["compiled_exists"] = True
            mode_result["reused_existing_compiled"] = True
        else:
            mode_result["compiled_locally"] = False
            mode_result["compile_note"] = (
                "edgetpu_compiler was not run locally. Use an x86-64 host with the Edge TPU compiler, "
                "or reuse an existing compiled *_edgetpu.tflite artifact."
            )

        if not args.prepare_only and compiled_path.exists():
            hybrid_command = [
                sys.executable,
                _repo_script("run_hybrid.py"),
                "--config",
                args.config,
                "--compiled-tflite",
                str(compiled_path),
                "--artifact-tag",
                label,
                "--quant-mode",
                label,
            ]
            if args.frame_limit is not None:
                hybrid_command.extend(["--frame-limit", str(args.frame_limit)])
            if args.boundary_tensors:
                hybrid_command.extend(["--boundary-tensors", *args.boundary_tensors])
            try:
                _run(hybrid_command)
                mode_result["hybrid_summary_path"] = str(_summary_path(config, label))
                mode_result["hybrid_succeeded"] = True
            except subprocess.CalledProcessError as exc:
                mode_result["hybrid_summary_path"] = None
                mode_result["hybrid_succeeded"] = False
                mode_result["hybrid_error"] = str(exc)
        elif not args.prepare_only:
            mode_result["hybrid_summary_path"] = None
            mode_result["run_note"] = (
                "Hybrid TPU execution was skipped because the compiled Edge TPU model was not available."
            )

        results[label] = mode_result

    experiment_summary = {
        "task_name": config["task_name"],
        "config_path": config["_config_path"],
        "boundary_tensors": boundary_tensors,
        "frame_limit": args.frame_limit,
        "prepare_only": args.prepare_only,
        "compiler_available_locally": compiler_available,
        "quantization_modes": results,
    }
    summary_path = artifacts_dir / "partition_experiment_summary.json"
    save_json(summary_path, experiment_summary)

    print(f"Saved partition experiment summary to: {summary_path}")
    for label, result in results.items():
        print(f"[{label}] tflite={result['tflite_path']}")
        if result.get("hybrid_summary_path"):
            print(f"[{label}] hybrid summary={result['hybrid_summary_path']}")


if __name__ == "__main__":
    main()
