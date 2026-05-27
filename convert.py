"""Convert a TF1 SavedModel to TFLite with optional quantization.

Supports a video file or an image directory as the representative / calibration
dataset.  The calibration dtype (float32 vs uint8) is auto-detected from the
SavedModel's first input tensor so DLC (float32) and SSD/DeepLab (uint8) are
handled without extra flags.

Usage:
    # DLC — video file
    python3 convert.py -m artifacts/dlc_split_after_block1/tpu_savedmodel \
        -O int8_pure -r data/task_a/data/ap-10k/data \
        -w 320 -t 320 -o artifacts/dlc_split_after_block1/tpu_int8.tflite

    # SSD — image directory
    python3 convert.py -m artifacts/ssd_split_after_expanded_conv_5/tpu_savedmodel \
        -O int8_pure -r data/task_b/data/val2017 \
        -w 300 -t 300 -o artifacts/ssd_split_after_expanded_conv_5/tpu_int8.tflite

    # float16 (no calibration needed)
    python3 convert.py -m <savedmodel_dir> -O float16 -o out.tflite
"""
import argparse
import glob
import os

import cv2
import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------------------
# Calibration data loading
# ---------------------------------------------------------------------------

def _load_calib_data(source: str, width: int, height: int,
                     max_samples: int, dtype: str) -> list:
    """Return a list of arrays shaped [1, height, width, 3].

    source — path to a video file OR a directory containing JPEG/PNG images.
    dtype  — 'float32' or 'uint8'.
    """
    as_uint8 = (dtype == "uint8")
    dataset: list = []

    if os.path.isdir(source):
        exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
        files: list = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(source, ext)))
        files = sorted(files)[:max_samples]
        for f in files:
            frame = cv2.imread(f)
            if frame is None:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (width, height))
            arr = np.expand_dims(frame, axis=0)
            dataset.append(arr.astype(np.uint8 if as_uint8 else np.float32))
    else:
        # Treat as video file
        cap = cv2.VideoCapture(source)
        for _ in range(max_samples):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (width, height))
            arr = np.expand_dims(frame, axis=0)
            dataset.append(arr.astype(np.uint8 if as_uint8 else np.float32))
        cap.release()

    return dataset


def _make_rep_dataset(dataset: list):
    def func():
        for item in dataset:
            yield [item]
    return func


def _get_input_dtype(model_path: str) -> str:
    """Detect the SavedModel's first input dtype ('float32' or 'uint8')."""
    try:
        sm = tf.saved_model.load(model_path)
        sig = sm.signatures.get("serving_default")
        if sig:
            for spec in sig.structured_input_signature[1].values():
                return "uint8" if spec.dtype == tf.uint8 else "float32"
    except Exception:
        pass
    return "float32"  # safe default for DLC


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="convert.py",
        description="Convert a TF1 SavedModel to TFLite with optional quantization.",
    )
    parser.add_argument("-m", "--model", required=True,
                        help="Path to the input TF1 SavedModel directory")
    parser.add_argument("-O", "--opt", default="none",
                        choices=["none", "drange", "float16", "int_fallback", "int8_pure"],
                        help="Quantization/optimisation mode")
    parser.add_argument("-r", "--rep", default=None,
                        help="Representative dataset: path to a video file or an image directory")
    parser.add_argument("-w", "--width", type=int, default=320,
                        help="Resize width for calibration frames (default 320)")
    parser.add_argument("-t", "--height", type=int, default=320,
                        help="Resize height for calibration frames (default 320)")
    parser.add_argument("-o", "--output", type=str, default="output.tflite",
                        help="Output .tflite path (default output.tflite)")
    parser.add_argument("--max-calib", type=int, default=200,
                        help="Max calibration samples to load (default 200)")
    parser.add_argument("--calib-dtype", choices=["auto", "float32", "uint8"],
                        default="auto",
                        help="Dtype for calibration frames ('auto' detects from SavedModel input)")
    parser.add_argument("--calib-normalize", choices=["none", "ssd"],
                        default="none",
                        help="Normalize calibration frames after loading: "
                             "'ssd' applies (x-128)/128 to put pixels in [-1, 1] "
                             "for SSD MobileNet V2 bypassing the Preprocessor cluster")
    args = parser.parse_args()

    # Argument validation
    if args.opt in ("int_fallback", "int8_pure") and args.rep is None:
        parser.error("--rep is required for int_fallback / int8_pure quantization")
    if args.opt == "drange" and args.rep is not None:
        parser.error("--rep should not be provided for drange optimisation")

    # Set threading BEFORE any TF operation that could initialize the runtime.
    # (threading config must be applied before the first Session/eager call)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(8)
        tf.config.threading.set_inter_op_parallelism_threads(8)
    except RuntimeError:
        pass  # TF already initialized (e.g. in a long-running process) — ignore

    # Determine calibration dtype
    calib_dtype = (
        _get_input_dtype(args.model)
        if args.calib_dtype == "auto"
        else args.calib_dtype
    )

    # Load calibration data
    dataset: list = []
    if args.rep is not None:
        print(f"Loading calibration data from: {args.rep}  (dtype={calib_dtype})")
        dataset = _load_calib_data(
            args.rep, args.width, args.height, args.max_calib, calib_dtype
        )
        # Optional post-load normalization (must be float32 first)
        if args.calib_normalize == "ssd":
            dataset = [arr.astype(np.float32) / 128.0 - 1.0 for arr in dataset]
            print(f"  applied SSD normalization (x/128-1, range ≈ [-1, 1])")
        print(f"  {len(dataset)} samples loaded")
        if not dataset:
            raise RuntimeError(
                f"No calibration samples loaded from {args.rep!r}. "
                "Check that the path exists and contains images / video frames."
            )

    converter = tf.lite.TFLiteConverter.from_saved_model(args.model)

    # https://www.tensorflow.org/lite/performance/post_training_quantization
    if args.opt != "none":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        if args.opt == "float16":
            converter.target_spec.supported_types = [tf.float16]

        elif args.opt == "int_fallback":
            # Integer where possible; float fallback for unsupported ops.
            # Good for models with ops the Edge TPU doesn't support.
            converter.representative_dataset = _make_rep_dataset(dataset)

        elif args.opt == "int8_pure":
            # Full integer quantization — required for Edge TPU compilation.
            # Keep inference_output_type as float32 so the TFLite runtime
            # auto-dequantizes boundary tensors; the CPU subgraph then
            # receives float32 without an explicit dequant step.
            converter.representative_dataset = _make_rep_dataset(dataset)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            if calib_dtype == "uint8":
                # SSD / DeepLab: keep uint8 input so the Edge TPU runtime
                # doesn't need to quantize from float32 at runtime.
                converter.inference_input_type = tf.uint8
            # inference_output_type intentionally left as default (float32):
            # boundary tensor values are fed as float32 to the CPU subgraph.

    print(f"Converting SavedModel '{args.model}' with opt={args.opt!r} ...")
    tflite_model = converter.convert()

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "wb") as f:
        f.write(tflite_model)
    print(f"Wrote {len(tflite_model):,} bytes → {args.output}")


if __name__ == "__main__":
    main()
