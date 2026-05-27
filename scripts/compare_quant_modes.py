#!/usr/bin/env python3
"""Compare int8_pure vs. int_fallback boundary activations for all 16 compiled candidates.

For each compiled candidate (those with edgetpu_compiled = true in metadata.json):
  1. Loads tpu_int8_pure.tflite and tpu_int_fallback.tflite via tflite.Interpreter (CPU only).
  2. Generates one random calibration sample matching the boundary input shape.
  3. Runs both interpreters and collects the boundary tensor output (last output tensor).
  4. Computes:
       - MSE between int8_pure and int_fallback boundary outputs (dequantized float32)
       - Cosine similarity
  5. Writes results to artifacts/quant_mode_comparison.json for Student D.

Usage:
    python3 scripts/compare_quant_modes.py [--artifacts ARTIFACTS_DIR]
                                            [--seed SEED]
                                            [--output OUTPUT_JSON]
                                            [--verbose]

No Coral hardware required — uses standard tflite.Interpreter on CPU.
Python/TF version: use the project venv (TF 2.21).
"""
import argparse
import json
import os
import sys

import numpy as np

try:
    import tensorflow as tf
    TFLiteInterpreter = tf.lite.Interpreter
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        TFLiteInterpreter = tflite.Interpreter
    except ImportError:
        print("ERROR: Neither tensorflow nor tflite_runtime is available.")
        print("       Activate the project venv: source venv/bin/activate")
        sys.exit(1)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_interpreter(tflite_path: str) -> "tf.lite.Interpreter":
    """Load a TFLite model into a CPU interpreter and allocate tensors."""
    interp = TFLiteInterpreter(model_path=tflite_path)
    interp.allocate_tensors()
    return interp


def run_inference(interp: "tf.lite.Interpreter",
                  input_data: np.ndarray) -> np.ndarray:
    """Run one inference pass. Returns the first output tensor as float32.

    If the model output is quantized (int8/uint8), dequantizes it using the
    output tensor's scale and zero_point metadata.
    """
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()

    # Set the input tensor
    inp = input_details[0]
    inp_dtype = np.dtype(inp["dtype"])

    # Quantize the float32 input if the model expects int8/uint8
    if inp_dtype in (np.int8, np.uint8):
        scale, zero_point = inp["quantization"]
        if scale != 0:
            data = (input_data / scale + zero_point).astype(inp_dtype)
        else:
            data = input_data.astype(inp_dtype)
    else:
        data = input_data.astype(inp_dtype)

    interp.set_tensor(inp["index"], data)
    interp.invoke()

    # Collect first output tensor and dequantize if quantized
    out_detail = output_details[0]
    raw = interp.get_tensor(out_detail["index"])
    out_dtype = np.dtype(out_detail["dtype"])

    if out_dtype in (np.int8, np.uint8):
        scale, zero_point = out_detail["quantization"]
        if scale != 0:
            result = (raw.astype(np.float32) - zero_point) * scale
        else:
            result = raw.astype(np.float32)
    else:
        result = raw.astype(np.float32)

    return result


def compute_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    """Compute MSE and cosine similarity between two float32 arrays."""
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)

    mse = float(np.mean((a_flat - b_flat) ** 2))

    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a > 0 and norm_b > 0:
        cosine_sim = float(np.dot(a_flat, b_flat) / (norm_a * norm_b))
    else:
        cosine_sim = None

    return {"mse": mse, "cosine_similarity": cosine_sim}


def make_calibration_input(interp: "tf.lite.Interpreter",
                            rng: np.random.Generator) -> np.ndarray:
    """Generate a random float32 calibration input matching the model's input shape.

    Inputs are drawn from U[0, 1] (float32 range expected by most TPU subgraphs
    after int8 dequantization at the boundary is undone). The interpreter's own
    quantization metadata handles the int8 conversion internally.
    """
    input_details = interp.get_input_details()
    shape = input_details[0]["shape"]
    return rng.random(shape).astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts", default=os.path.join(ROOT, "artifacts"),
                        help="Path to artifacts/ directory (default: ./artifacts)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for calibration input generation (default: 42)")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: artifacts/quant_mode_comparison.json)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-tensor stats in addition to summary")
    args = parser.parse_args()

    artifacts_dir = args.artifacts
    output_path = args.output or os.path.join(artifacts_dir, "quant_mode_comparison.json")

    rng = np.random.default_rng(args.seed)

    # Find all candidate dirs with both tflite variants
    results = []
    skipped = []

    candidate_dirs = sorted([
        d for d in os.listdir(artifacts_dir)
        if os.path.isdir(os.path.join(artifacts_dir, d))
        and d.startswith(("dlc_", "ssd_", "deeplab_"))
    ])

    print(f"Scanning {len(candidate_dirs)} candidate dirs in {artifacts_dir}/\n")

    for cid in candidate_dirs:
        cdir = os.path.join(artifacts_dir, cid)
        meta_path = os.path.join(cdir, "metadata.json")
        pure_path = os.path.join(cdir, "tpu_int8_pure.tflite")
        fallback_path = os.path.join(cdir, "tpu_int_fallback.tflite")

        # Load metadata to check if edgetpu_compiled
        if not os.path.exists(meta_path):
            skipped.append((cid, "no metadata.json"))
            continue
        with open(meta_path) as f:
            meta = json.load(f)

        compiled = meta.get("edgetpu_compiled", None)
        if not compiled:
            skipped.append((cid, "edgetpu_compiled=false (failed candidates skipped)"))
            continue

        if not os.path.exists(pure_path):
            skipped.append((cid, "tpu_int8_pure.tflite missing"))
            continue
        if not os.path.exists(fallback_path):
            skipped.append((cid, "tpu_int_fallback.tflite missing"))
            continue

        print(f"  Processing {cid} ...")

        try:
            interp_pure = load_interpreter(pure_path)
            interp_fallback = load_interpreter(fallback_path)
        except Exception as e:
            skipped.append((cid, f"interpreter load error: {e}"))
            print(f"    ❌ load error: {e}")
            continue

        # Generate a calibration input (same random sample for both modes)
        try:
            calib_input = make_calibration_input(interp_pure, rng)
        except Exception as e:
            skipped.append((cid, f"input generation error: {e}"))
            print(f"    ❌ input error: {e}")
            continue

        # Run int8_pure
        try:
            out_pure = run_inference(interp_pure, calib_input)
        except Exception as e:
            skipped.append((cid, f"int8_pure inference error: {e}"))
            print(f"    ❌ int8_pure inference error: {e}")
            continue

        # Run int_fallback
        try:
            out_fallback = run_inference(interp_fallback, calib_input)
        except Exception as e:
            skipped.append((cid, f"int_fallback inference error: {e}"))
            print(f"    ❌ int_fallback inference error: {e}")
            continue

        # Compute metrics
        metrics = compute_metrics(out_pure, out_fallback)

        entry = {
            "partition_id": meta.get("partition_id", cid),
            "model": meta.get("model", "?"),
            "tpu_ops_mapped_edgetpu": meta.get("tpu_ops_mapped_edgetpu"),
            "boundary_tensor_shapes": meta.get("boundary_tensor_shapes"),
            "int8_pure_output_shape": list(out_pure.shape),
            "int_fallback_output_shape": list(out_fallback.shape),
            "mse_pure_vs_fallback": metrics["mse"],
            "cosine_similarity_pure_vs_fallback": metrics["cosine_similarity"],
            "int8_pure_output_stats": {
                "min": float(out_pure.min()),
                "max": float(out_pure.max()),
                "mean": float(out_pure.mean()),
                "std": float(out_pure.std()),
            },
            "int_fallback_output_stats": {
                "min": float(out_fallback.min()),
                "max": float(out_fallback.max()),
                "mean": float(out_fallback.mean()),
                "std": float(out_fallback.std()),
            },
        }
        results.append(entry)

        mse_str = f"{metrics['mse']:.6f}"
        cos_str = f"{metrics['cosine_similarity']:.6f}" if metrics["cosine_similarity"] is not None else "n/a"
        print(f"    ✅ MSE={mse_str}  cosine={cos_str}  "
              f"shapes pure={list(out_pure.shape)} fallback={list(out_fallback.shape)}")
        if args.verbose:
            print(f"       int8_pure:    min={out_pure.min():.4f}  max={out_pure.max():.4f}  "
                  f"mean={out_pure.mean():.4f}  std={out_pure.std():.4f}")
            print(f"       int_fallback: min={out_fallback.min():.4f}  max={out_fallback.max():.4f}  "
                  f"mean={out_fallback.mean():.4f}  std={out_fallback.std():.4f}")

    # Write output JSON
    output_doc = {
        "description": (
            "Boundary tensor comparison: int8_pure vs. int_fallback TFLite models "
            "for the 16 compiled Edge TPU candidates. "
            "Both models run on CPU (no Coral hardware). "
            "MSE and cosine similarity measure how much quantization mode affects "
            "the boundary activations fed to the CPU subgraph."
        ),
        "date": "2026-05-27",
        "random_seed": args.seed,
        "num_candidates_compared": len(results),
        "num_candidates_skipped": len(skipped),
        "skipped": [{"id": s[0], "reason": s[1]} for s in skipped],
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output_doc, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results written to: {output_path}")
    print(f"  Compared: {len(results)} candidates")
    if skipped:
        print(f"  Skipped:  {len(skipped)} candidates")
        for sid, reason in skipped:
            print(f"    - {sid}: {reason}")

    if results:
        mses = [r["mse_pure_vs_fallback"] for r in results]
        coses = [r["cosine_similarity_pure_vs_fallback"] for r in results
                 if r["cosine_similarity_pure_vs_fallback"] is not None]
        print(f"\n  MSE summary:    min={min(mses):.6f}  max={max(mses):.6f}  "
              f"mean={sum(mses)/len(mses):.6f}")
        if coses:
            print(f"  Cosine summary: min={min(coses):.6f}  max={max(coses):.6f}  "
                  f"mean={sum(coses)/len(coses):.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
