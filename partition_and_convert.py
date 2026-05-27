"""Unified partition-and-convert pipeline for Coral Edge TPU deployment.

Weeks 7–8 deliverable for Student B (Jared Velasquez, cs211-winter2024).

This script composes the four existing sub-scripts into a single CLI:

    gen_tflite.py           -- extract TPU-side SavedModel
    extract_cpu_subgraph.py -- extract CPU-side SavedModel
    convert.py              -- quantize to TFLite (int8_pure and/or int_fallback)
    edgetpu_compiler        -- compile to Coral Edge TPU binary (x86-64 only)

After running those steps it writes ``metadata.json`` and validates it via
``scripts/validate_metadata.py``.

Pipeline position::

    partition_and_convert.py
        -> gen_tflite.py        -> tpu_savedmodel/
        -> extract_cpu_subgraph.py -> cpu_savedmodel/
        -> convert.py           -> tpu_int8_pure.tflite  (and/or int_fallback)
        -> edgetpu_compiler     -> tpu_int8_pure_edgetpu.tflite
        -> metadata.json        (validated by scripts/validate_metadata.py)

Example usage (SSD MobileNet V2, boundary after expanded_conv_10)::

    python3 partition_and_convert.py \\
      --model ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb \\
      --input-tensor FeatureExtractor/MobilenetV2/MobilenetV2/input:0 \\
      --input-shape 1,300,300,3 \\
      --boundary-tensors FeatureExtractor/MobilenetV2/expanded_conv_10/output \\
      --final-outputs Squeeze,concat_1 \\
      --quant-mode int8_pure \\
      --calib-dir data/task_b/data/val2017 \\
      --calib-normalize ssd \\
      --calib-width 300 --calib-height 300 --max-calib 50 \\
      --partition-id split_after_expanded_conv_10 \\
      --model-name ssd_mobilenet_v2 \\
      --num-tpu-ops 764 --num-cpu-ops 7230 \\
      --output-dir artifacts/ssd_split_after_expanded_conv_10_v2

Notes:
- ``edgetpu_compiler`` is x86-64 only (Google Colab / Debian x86-64).
  Pass ``--skip-compile`` when running on ARM64 Raspberry Pi.
- Use ``--dry-run`` to print all commands without executing them.
- Sub-scripts are called via ``subprocess.run()`` with relative paths
  from the project root (same convention as gen_all_candidates.py).
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_colon_suffix(name: str) -> str:
    """Remove trailing ':N' from a tensor name (e.g. 'foo:0' -> 'foo').

    gen_tflite.py and extract_cpu_subgraph.py expect bare op names; they add
    the ':0' themselves.
    """
    return re.sub(r":\d+$", "", name)


def _add_colon_zero(name: str) -> str:
    """Ensure a tensor name ends with ':0'."""
    if ":" not in name:
        return name + ":0"
    return name


def _parse_csv(s: str) -> list:
    """Split a comma-separated string, stripping whitespace."""
    return [item.strip() for item in s.split(",") if item.strip()]


def _file_size_kb(path: str) -> int:
    """Return file size in KB (rounded), or -1 if file does not exist."""
    try:
        return round(os.path.getsize(path) / 1024)
    except OSError:
        return -1


def _run(cmd: list, dry_run: bool, desc: str = "",
         capture_output: bool = False) -> subprocess.CompletedProcess:
    """Run a subprocess command.

    Parameters
    ----------
    cmd:
        List of command tokens (same as subprocess.run).
    dry_run:
        If True, print the command but do not execute it.  Returns a
        synthetic CompletedProcess with returncode=0.
    desc:
        Short human-readable description printed before the command.
    capture_output:
        If True, capture stdout+stderr (returned on the result object).

    Returns
    -------
    subprocess.CompletedProcess
    """
    if desc:
        print(f"  {desc}")
    print("  $", " ".join(str(tok) for tok in cmd))
    if dry_run:
        return subprocess.CompletedProcess(cmd, returncode=0, stdout=b"", stderr=b"")

    kwargs = {}
    if capture_output:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE

    result = subprocess.run(cmd, **kwargs)
    return result


# ---------------------------------------------------------------------------
# Step 1: TPU subgraph extraction (gen_tflite.py)
# ---------------------------------------------------------------------------

def step_gen_tflite(args: argparse.Namespace, boundary_tensors_bare: list,
                    tpu_sm_dir: str) -> bool:
    """Call gen_tflite.py to extract the TPU-side SavedModel.

    Returns True on success (or dry_run).
    """
    print("\n[Step 1] TPU subgraph extraction (gen_tflite.py)")

    # gen_tflite.py expects the input tensor name WITHOUT ':0'
    input_tensor_bare = _strip_colon_suffix(args.input_tensor)

    cmd = [
        sys.executable, "gen_tflite.py",
        "--model", args.model,
        "--input-tensor", input_tensor_bare,
        "--input-shape", args.input_shape,
        "--output-tensors", ",".join(boundary_tensors_bare),
        "--output-dir", tpu_sm_dir,
    ]
    result = _run(cmd, args.dry_run, desc="Extract TPU subgraph")
    if result.returncode != 0:
        print(f"  ERROR: gen_tflite.py failed (exit code {result.returncode})")
        return False
    return True


# ---------------------------------------------------------------------------
# Step 2: CPU subgraph extraction (extract_cpu_subgraph.py)
# ---------------------------------------------------------------------------

def step_extract_cpu(args: argparse.Namespace, boundary_tensors_bare: list,
                     cpu_sm_dir: str) -> bool:
    """Call extract_cpu_subgraph.py to extract the CPU-side SavedModel.

    Returns True on success (or dry_run).
    """
    print("\n[Step 2] CPU subgraph extraction (extract_cpu_subgraph.py)")

    # extract_cpu_subgraph.py expects tensor names WITH ':0'
    boundary_fq = ",".join(_add_colon_zero(t) for t in boundary_tensors_bare)

    cmd = [
        sys.executable, "extract_cpu_subgraph.py",
        "--model", args.model,
        "--boundary-tensors", boundary_fq,
        "--final-output-tensor", args.final_outputs,
        "--output-dir", cpu_sm_dir,
    ]
    result = _run(cmd, args.dry_run, desc="Extract CPU subgraph")
    if result.returncode != 0:
        print(f"  ERROR: extract_cpu_subgraph.py failed (exit code {result.returncode})")
        return False
    return True


# ---------------------------------------------------------------------------
# Step 3: Quantization (convert.py)
# ---------------------------------------------------------------------------

def step_quantize(args: argparse.Namespace, tpu_sm_dir: str,
                  output_dir: str, quant_mode: str) -> Tuple[bool, str]:
    """Call convert.py for a single quantization mode.

    Returns (success: bool, tflite_path: str).
    """
    print(f"\n[Step 3] Quantization — mode={quant_mode} (convert.py)")

    tflite_filename = f"tpu_{quant_mode}.tflite"
    tflite_path = os.path.join(output_dir, tflite_filename)

    cmd = [
        sys.executable, "convert.py",
        "--model", tpu_sm_dir,
        "--opt", quant_mode,
        "--output", tflite_path,
    ]

    # Calibration args are required for int8_pure and int_fallback
    if quant_mode in ("int8_pure", "int_fallback"):
        if args.calib_dir is None:
            print("  ERROR: --calib-dir is required for int8_pure / int_fallback")
            return False, tflite_path
        cmd += ["--rep", args.calib_dir]
        if args.calib_width is not None:
            cmd += ["--width", str(args.calib_width)]
        if args.calib_height is not None:
            cmd += ["--height", str(args.calib_height)]
        if args.max_calib is not None:
            cmd += ["--max-calib", str(args.max_calib)]
        if args.calib_normalize is not None:
            cmd += ["--calib-normalize", args.calib_normalize]

    result = _run(cmd, args.dry_run, desc=f"Convert + quantize ({quant_mode})")
    if result.returncode != 0:
        print(f"  ERROR: convert.py failed for {quant_mode} (exit code {result.returncode})")
        return False, tflite_path
    return True, tflite_path


# ---------------------------------------------------------------------------
# Step 4: edgetpu_compiler
# ---------------------------------------------------------------------------

def _find_edgetpu_compiler() -> Optional[str]:
    """Return the path to edgetpu_compiler, or None if not found."""
    return shutil.which("edgetpu_compiler")


def _parse_compiler_ops(stdout: str) -> Optional[str]:
    """Parse edgetpu_compiler stdout to extract 'X/Y' ops-on-TPU string.

    Looks for lines like:
        Number of operations that will run on Edge TPU: 39
        Number of operations that will run on CPU: 2

    Returns a string like '39/41', or None if parsing failed.
    """
    on_tpu_match = re.search(
        r"Number of operations that will run on Edge TPU:\s*(\d+)", stdout)
    on_cpu_match = re.search(
        r"Number of operations that will run on CPU:\s*(\d+)", stdout)
    if on_tpu_match and on_cpu_match:
        n_tpu = int(on_tpu_match.group(1))
        n_cpu = int(on_cpu_match.group(1))
        total = n_tpu + n_cpu
        return f"{n_tpu}/{total}"
    # Fallback: only TPU count available
    if on_tpu_match:
        return on_tpu_match.group(1)
    return None


def step_edgetpu_compile(args: argparse.Namespace, tflite_path: str,
                         output_dir: str) -> dict:
    """Run edgetpu_compiler on the given TFLite file.

    Returns a dict with keys:
        edgetpu_compiled      : True / False / None
        tpu_edgetpu_path      : str path or None
        tpu_ops_mapped_edgetpu: 'X/Y' string or None
        edgetpu_rejection_reason: str or None
    """
    result_meta = {
        "edgetpu_compiled": None,
        "tpu_edgetpu_path": None,
        "tpu_ops_mapped_edgetpu": None,
        "edgetpu_rejection_reason": None,
    }

    compiler = _find_edgetpu_compiler()

    if compiler is None:
        print(
            "\n[Step 4] edgetpu_compiler — NOT FOUND on PATH\n"
            "         (edgetpu_compiler is x86-64 only — run on Colab or an\n"
            "          x86-64 Debian host with the Coral toolchain installed)"
        )
        # Leave edgetpu_compiled as None (not attempted)
        return result_meta

    print(f"\n[Step 4] edgetpu_compiler — found at {compiler}")

    # edgetpu_compiler writes <stem>_edgetpu.tflite in the current working dir
    # by default, but accepts -o <output_dir>.
    cmd = [compiler, tflite_path, "-o", output_dir]

    result = _run(cmd, args.dry_run, desc="Compile for Edge TPU",
                  capture_output=True)

    if args.dry_run:
        # In dry_run mode we cannot inspect real output; leave as None.
        return result_meta

    stdout = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
    stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""

    # Print combined output so the user sees it
    combined = (stdout + "\n" + stderr).strip()
    if combined:
        print("  --- compiler output ---")
        for line in combined.splitlines():
            print(f"  {line}")
        print("  -----------------------")

    if result.returncode == 0:
        # Derive the expected output filename: edgetpu_compiler appends
        # '_edgetpu' before the .tflite extension.
        tflite_stem = Path(tflite_path).stem          # e.g. 'tpu_int8_pure'
        edgetpu_filename = f"{tflite_stem}_edgetpu.tflite"
        edgetpu_path = str(Path(output_dir) / edgetpu_filename)
        result_meta["edgetpu_compiled"] = True
        result_meta["tpu_edgetpu_path"] = edgetpu_path
        result_meta["tpu_ops_mapped_edgetpu"] = _parse_compiler_ops(stdout)
    else:
        # Capture rejection reason from stderr (first meaningful line)
        rejection = ""
        for line in (stderr + stdout).splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                rejection = line
                break
        result_meta["edgetpu_compiled"] = False
        result_meta["edgetpu_rejection_reason"] = rejection or "unknown error"

    return result_meta


# ---------------------------------------------------------------------------
# Step 5: Write metadata.json
# ---------------------------------------------------------------------------

def write_metadata(args: argparse.Namespace, output_dir: str,
                   boundary_tensors_bare: list,
                   tflite_paths: dict,
                   edgetpu_meta: dict) -> str:
    """Write metadata.json to output_dir.

    Parameters
    ----------
    args:
        Parsed CLI arguments.
    output_dir:
        Destination directory (must exist).
    boundary_tensors_bare:
        Boundary tensor names without ':0' suffix.
    tflite_paths:
        Dict mapping quant_mode -> tflite path (only modes that ran).
    edgetpu_meta:
        Dict from step_edgetpu_compile().

    Returns
    -------
    str: path to the written metadata.json
    """
    print("\n[Step 5] Writing metadata.json")

    # Fully-qualified tensor names (with :0) used in tpu_output_tensors /
    # cpu_input_tensors — this matches the convention in gen_all_candidates.py.
    tpu_output_tensors = [_add_colon_zero(t) for t in boundary_tensors_bare]

    # Determine the primary TFLite path for tpu_tflite_path.
    # Prefer int8_pure; fall back to int_fallback; then whatever is available.
    primary_tflite = None
    for preferred in ("int8_pure", "int_fallback"):
        if preferred in tflite_paths:
            primary_tflite = tflite_paths[preferred]
            break
    if primary_tflite is None and tflite_paths:
        primary_tflite = next(iter(tflite_paths.values()))

    # quant_mode: use the mode that was actually run (or 'both')
    modes_run = list(tflite_paths.keys())
    if len(modes_run) == 1:
        quant_mode_str = modes_run[0]
    elif len(modes_run) > 1:
        quant_mode_str = "both"
    else:
        quant_mode_str = args.quant_mode  # dry_run fallback

    # boundary_tensor_shapes: populated from --boundary-shapes if provided
    boundary_shapes = None
    if args.boundary_shapes is not None:
        # Format: shapes separated by '|', each shape comma-separated ints
        # e.g. "1,65,65,32|1,65,65,96"
        boundary_shapes = [
            [int(x) for x in shape_str.split(",")]
            for shape_str in args.boundary_shapes.split("|")
            if shape_str.strip()
        ]

    meta = {
        "model": args.model_name,
        "partition_id": args.partition_id,
        "tpu_output_tensors": tpu_output_tensors,
        "cpu_input_tensors": tpu_output_tensors,
        "tpu_tflite_path": primary_tflite,
        "cpu_graph_path": os.path.join(output_dir, "cpu_savedmodel"),
        "num_tpu_ops": args.num_tpu_ops,
        "num_cpu_ops": args.num_cpu_ops,
        "boundary_tensor_shapes": boundary_shapes,
        "boundary_bandwidth_bytes": None,
        "has_skip_crossing": None,
        "quant_mode": quant_mode_str,
        # edgetpu fields (populated by step 4)
        "edgetpu_compiled": edgetpu_meta["edgetpu_compiled"],
        "tpu_edgetpu_path": edgetpu_meta["tpu_edgetpu_path"],
        "tpu_ops_mapped_edgetpu": edgetpu_meta["tpu_ops_mapped_edgetpu"],
        "edgetpu_rejection_reason": edgetpu_meta["edgetpu_rejection_reason"],
    }

    meta_path = os.path.join(output_dir, "metadata.json")
    if not args.dry_run:
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  wrote {meta_path}")
    else:
        print(f"  (dry-run) would write {meta_path}")
        print("  contents preview:")
        print(json.dumps(meta, indent=4))

    return meta_path


# ---------------------------------------------------------------------------
# Step 6: Validate metadata.json
# ---------------------------------------------------------------------------

def step_validate_metadata(args: argparse.Namespace,
                            meta_path: str) -> bool:
    """Run scripts/validate_metadata.py and print the result.

    Returns True if validation passed (or dry_run).
    """
    print("\n[Step 6] Validating metadata.json (scripts/validate_metadata.py)")

    # Derive the artifacts root from the output_dir so the validator can find
    # this candidate's metadata.json even when --root is the candidate dir.
    # We pass the parent of output_dir as the root so validate_metadata walks
    # up one level (to artifacts/) and finds other candidates too.  In the
    # common case --output-dir is artifacts/<id>/ so the parent is artifacts/.
    artifacts_root = str(Path(args.output_dir).parent)

    cmd = [
        sys.executable, "scripts/validate_metadata.py",
        "--root", artifacts_root,
    ]
    result = _run(cmd, args.dry_run, desc="Validate metadata")
    if args.dry_run:
        return True

    if result.returncode != 0:
        print("  WARNING: metadata validation reported errors (see above)")
        return False
    return True


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _status_symbol(ok: Optional[bool], skipped: bool = False) -> str:
    """Return a one-character status symbol."""
    if skipped:
        return "skip"
    if ok is True:
        return "OK"
    if ok is False:
        return "FAIL"
    return "n/a"


def print_summary(args: argparse.Namespace, output_dir: str,
                  tpu_sm_ok: Optional[bool], cpu_sm_ok: Optional[bool],
                  quant_results: dict,
                  edgetpu_meta: dict,
                  meta_valid: Optional[bool]) -> None:
    """Print a human-readable pipeline summary."""

    print("\n" + "=" * 52)
    print("=== partition_and_convert.py summary ===")
    print("=" * 52)
    print(f"Model:          {args.model_name}")
    print(f"Partition:      {args.partition_id}")
    print(f"Output dir:     {output_dir}")

    # TPU subgraph
    tpu_sm_dir = os.path.join(output_dir, "tpu_savedmodel")
    if tpu_sm_ok is None:
        tpu_status = "skip (--skip-existing)"
    elif tpu_sm_ok:
        tpu_status = f"OK  {tpu_sm_dir}/"
    else:
        tpu_status = f"FAIL"
    print(f"TPU subgraph:   {tpu_status}")

    # CPU subgraph
    cpu_sm_dir = os.path.join(output_dir, "cpu_savedmodel")
    if cpu_sm_ok is None:
        cpu_status = "skip (--skip-existing)"
    elif cpu_sm_ok:
        cpu_status = f"OK  {cpu_sm_dir}/"
    else:
        cpu_status = f"FAIL"
    print(f"CPU subgraph:   {cpu_status}")

    # Quantization modes
    for mode in ("int8_pure", "int_fallback"):
        result = quant_results.get(mode)
        if result is None:
            label = "skip"
        elif result["ok"]:
            kb = _file_size_kb(result["path"])
            kb_str = f"({kb:,} KB)" if kb >= 0 else ""
            label = f"OK  {kb_str}"
        else:
            label = "FAIL"
        print(f"Quantize {mode:<14}: {label}")

    # edgetpu_compiler
    compiled = edgetpu_meta.get("edgetpu_compiled")
    if compiled is None:
        # Not attempted
        compiler_path = _find_edgetpu_compiler()
        if compiler_path is None:
            edgetpu_status = "not found (x86-64 only) -- run on Colab"
        elif args.skip_compile:
            edgetpu_status = "skipped (--skip-compile)"
        else:
            edgetpu_status = "n/a"
    elif compiled is True:
        ops_str = edgetpu_meta.get("tpu_ops_mapped_edgetpu") or ""
        edgetpu_status = f"OK  ops on TPU: {ops_str}"
    else:
        reason = edgetpu_meta.get("edgetpu_rejection_reason") or "unknown"
        edgetpu_status = f"FAIL  {reason}"
    print(f"edgetpu_compiler:    {edgetpu_status}")

    # Metadata
    if meta_valid is None:
        meta_status = "skip (dry-run)"
    elif meta_valid:
        meta_status = "valid"
    else:
        meta_status = "INVALID (see above)"
    print(f"Metadata:       {meta_status}")
    print("=" * 52 + "\n")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="partition_and_convert.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Model / graph args ---------------------------------------------------
    g_model = p.add_argument_group("Model / graph")
    g_model.add_argument("--model", required=True,
                         help="Path to the frozen TF .pb graph file")
    g_model.add_argument("--input-tensor", required=True,
                         help="Name of the model's input placeholder "
                              "(e.g. 'FeatureExtractor/MobilenetV2/MobilenetV2/input:0'). "
                              "Trailing ':0' is stripped automatically for sub-scripts "
                              "that expect bare op names.")
    g_model.add_argument("--input-shape", required=True,
                         help="Comma-separated input shape, e.g. '1,300,300,3'")
    g_model.add_argument("--boundary-tensors", required=True,
                         help="Comma-separated list of boundary tensor names at the "
                              "TPU/CPU split point. ':0' suffixes are handled automatically.")
    g_model.add_argument("--final-outputs", required=True,
                         help="Comma-separated final output tensor names of the full graph "
                              "(used by extract_cpu_subgraph.py), "
                              "e.g. 'Squeeze,concat_1' or 'SemanticPredictions'")

    # --- Metadata args --------------------------------------------------------
    g_meta = p.add_argument_group("Partition metadata")
    g_meta.add_argument("--partition-id", required=True,
                        help="Kebab-case slug for this partition candidate "
                             "(e.g. 'split_after_expanded_conv_10')")
    g_meta.add_argument("--model-name", required=True,
                        help="Model name for metadata (e.g. 'ssd_mobilenet_v2')")
    g_meta.add_argument("--num-tpu-ops", required=True, type=int,
                        help="Static TPU op count for metadata")
    g_meta.add_argument("--num-cpu-ops", required=True, type=int,
                        help="Static CPU op count for metadata")
    g_meta.add_argument("--boundary-shapes", default=None,
                        help="Optional: '|'-separated list of shapes for each boundary "
                             "tensor (comma-separated ints per shape), "
                             "e.g. '1,65,65,32' (single) or '1,65,65,32|1,65,65,96' "
                             "(multiple). Populates boundary_tensor_shapes in metadata.json.")

    # --- Quantization args ----------------------------------------------------
    g_quant = p.add_argument_group("Quantization (convert.py)")
    g_quant.add_argument("--quant-mode",
                         choices=["int8_pure", "int_fallback", "both"],
                         default="int8_pure",
                         help="Quantization mode. 'both' runs int8_pure then int_fallback. "
                              "(default: int8_pure)")
    g_quant.add_argument("--calib-dir", default=None,
                         help="Path to calibration image directory (required for "
                              "int8_pure / int_fallback)")
    g_quant.add_argument("--calib-normalize", default=None,
                         choices=["none", "ssd"],
                         help="Normalization for calibration frames: 'ssd' applies "
                              "pixel/128-1 for SSD MobileNet V2")
    g_quant.add_argument("--calib-width", type=int, default=None,
                         help="Resize width for calibration frames (default: convert.py default)")
    g_quant.add_argument("--calib-height", type=int, default=None,
                         help="Resize height for calibration frames (default: convert.py default)")
    g_quant.add_argument("--max-calib", type=int, default=None,
                         help="Max calibration samples to use (default: convert.py default)")

    # --- Output / control args ------------------------------------------------
    g_out = p.add_argument_group("Output / control")
    g_out.add_argument("--output-dir", required=True,
                       help="Destination directory for all outputs "
                            "(tpu_savedmodel/, cpu_savedmodel/, *.tflite, metadata.json)")
    g_out.add_argument("--skip-compile", action="store_true",
                       help="Skip the edgetpu_compiler step (useful on ARM64 / non-Coral hosts)")
    g_out.add_argument("--skip-existing", action="store_true",
                       help="Skip extraction and quantization steps if "
                            "tpu_savedmodel/ already exists in --output-dir")
    g_out.add_argument("--dry-run", action="store_true",
                       help="Print all commands without executing them")

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Entry point. Returns 0 on success, non-zero on failure."""
    parser = build_parser()
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve paths and parse inputs
    # ------------------------------------------------------------------
    output_dir = os.path.abspath(args.output_dir)

    # Parse boundary tensors; strip ':0' for sub-scripts that add it themselves
    boundary_tensors_raw = _parse_csv(args.boundary_tensors)
    boundary_tensors_bare = [_strip_colon_suffix(t) for t in boundary_tensors_raw]

    if not boundary_tensors_bare:
        print("ERROR: --boundary-tensors produced an empty list after parsing")
        return 1

    tpu_sm_dir = os.path.join(output_dir, "tpu_savedmodel")
    cpu_sm_dir = os.path.join(output_dir, "cpu_savedmodel")

    print(f"partition_and_convert.py")
    print(f"  model:          {args.model}")
    print(f"  partition-id:   {args.partition_id}")
    print(f"  output-dir:     {output_dir}")
    print(f"  boundary:       {boundary_tensors_bare}")
    print(f"  quant-mode:     {args.quant_mode}")
    print(f"  dry-run:        {args.dry_run}")
    print(f"  skip-compile:   {args.skip_compile}")
    print(f"  skip-existing:  {args.skip_existing}")

    # ------------------------------------------------------------------
    # Create output directory
    # ------------------------------------------------------------------
    if not args.dry_run:
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Decide whether to skip extraction (--skip-existing)
    # ------------------------------------------------------------------
    skip_extract = args.skip_existing and os.path.isdir(tpu_sm_dir)
    if skip_extract:
        print(f"\n  tpu_savedmodel/ exists and --skip-existing set — skipping extraction")
    # tpu_sm_ok / cpu_sm_ok: None means "skipped", True/False = result
    tpu_sm_ok = None if skip_extract else True  # updated below if not skip_extract
    cpu_sm_ok = None if skip_extract else True

    # ------------------------------------------------------------------
    # Step 1: TPU subgraph extraction
    # ------------------------------------------------------------------
    if not skip_extract:
        tpu_sm_ok = step_gen_tflite(args, boundary_tensors_bare, tpu_sm_dir)
        if not tpu_sm_ok and not args.dry_run:
            print("Aborting: TPU subgraph extraction failed.")
            print_summary(args, output_dir, tpu_sm_ok, None, {}, {}, None)
            return 1

    # ------------------------------------------------------------------
    # Step 2: CPU subgraph extraction
    # ------------------------------------------------------------------
    if not skip_extract:
        cpu_sm_ok = step_extract_cpu(args, boundary_tensors_bare, cpu_sm_dir)
        if not cpu_sm_ok and not args.dry_run:
            print("Aborting: CPU subgraph extraction failed.")
            print_summary(args, output_dir, tpu_sm_ok, cpu_sm_ok, {}, {}, None)
            return 1

    # ------------------------------------------------------------------
    # Step 3: Quantization
    # ------------------------------------------------------------------
    # Determine which modes to run
    if args.quant_mode == "both":
        modes = ["int8_pure", "int_fallback"]
    else:
        modes = [args.quant_mode]

    # quant_results: mode -> {"ok": bool, "path": str}
    quant_results: Dict[str, dict] = {}

    # The primary TFLite path (int8_pure preferred) used for edgetpu_compiler
    primary_tflite_for_compile = None  # Optional[str]; set below

    for mode in modes:
        ok, tflite_path = step_quantize(args, tpu_sm_dir, output_dir, mode)
        quant_results[mode] = {"ok": ok, "path": tflite_path}
        if not ok and not args.dry_run:
            print(f"  WARNING: quantization failed for mode '{mode}' — continuing")
        # Track primary tflite (first successful int8_pure, else first successful)
        if primary_tflite_for_compile is None and ok:
            primary_tflite_for_compile = tflite_path
        if mode == "int8_pure" and ok:
            primary_tflite_for_compile = tflite_path  # always prefer int8_pure

    # In dry_run, synthesise a plausible path for metadata
    if args.dry_run and primary_tflite_for_compile is None:
        primary_tflite_for_compile = os.path.join(output_dir, "tpu_int8_pure.tflite")

    # ------------------------------------------------------------------
    # Step 4: edgetpu_compiler
    # ------------------------------------------------------------------
    edgetpu_meta = {
        "edgetpu_compiled": None,
        "tpu_edgetpu_path": None,
        "tpu_ops_mapped_edgetpu": None,
        "edgetpu_rejection_reason": None,
    }  # Dict[str, object]

    if args.skip_compile:
        print("\n[Step 4] edgetpu_compiler — skipped (--skip-compile)")
        # edgetpu_compiled stays None (not attempted)
    elif primary_tflite_for_compile is None:
        print("\n[Step 4] edgetpu_compiler — skipped (no tflite file produced)")
    else:
        edgetpu_meta = step_edgetpu_compile(args, primary_tflite_for_compile, output_dir)

    # ------------------------------------------------------------------
    # Step 5: Write metadata.json
    # ------------------------------------------------------------------
    # Build tflite_paths dict for metadata
    tflite_paths_for_meta: Dict[str, str] = {}
    for mode, res in quant_results.items():
        if res["ok"] or args.dry_run:
            tflite_paths_for_meta[mode] = res["path"]

    meta_path = write_metadata(
        args=args,
        output_dir=output_dir,
        boundary_tensors_bare=boundary_tensors_bare,
        tflite_paths=tflite_paths_for_meta,
        edgetpu_meta=edgetpu_meta,
    )

    # ------------------------------------------------------------------
    # Step 6: Validate metadata.json
    # ------------------------------------------------------------------
    meta_valid = step_validate_metadata(args, meta_path)  # Optional[bool]

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print_summary(
        args=args,
        output_dir=output_dir,
        tpu_sm_ok=tpu_sm_ok,
        cpu_sm_ok=cpu_sm_ok,
        quant_results=quant_results,
        edgetpu_meta=edgetpu_meta,
        meta_valid=meta_valid,
    )

    # Return non-zero if any critical step failed
    extraction_ok = (skip_extract or (tpu_sm_ok and cpu_sm_ok))
    any_quant_ok = any(r["ok"] for r in quant_results.values())

    if not args.dry_run:
        if not extraction_ok:
            return 1
        if quant_results and not any_quant_ok:
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
