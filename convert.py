from __future__ import annotations

import argparse
import glob
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from src.config_utils import load_config
from src.data_loaders import load_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a SavedModel prefix into TFLite.")
    parser.add_argument(
        "--config",
        default="configs/task_a_dlc.json",
        help="Path to the task config JSON.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to the input SavedModel. Defaults to prefix_saved_model_dir from config.",
    )
    parser.add_argument(
        "--opt",
        default="none",
        choices=["none", "drange", "float16", "int_fallback", "int8_pure"],
        help="Optimization type.",
    )
    parser.add_argument(
        "--rep-video",
        default=None,
        help="Optional representative video override for int quantization. Defaults to the config data loader.",
    )
    parser.add_argument(
        "--rep",
        default=None,
        help="Candidate mode representative data source: image directory or video file.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Candidate mode representative resize width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Candidate mode representative resize height.",
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=100,
        help="Maximum number of representative frames to load.",
    )
    parser.add_argument(
        "--max-calib",
        type=int,
        default=None,
        help="Candidate mode maximum number of representative samples.",
    )
    parser.add_argument(
        "--calib-dtype",
        choices=["auto", "float32", "uint8"],
        default="auto",
        help="Candidate mode representative input dtype.",
    )
    parser.add_argument(
        "--calib-normalize",
        choices=["none", "ssd"],
        default="none",
        help="Candidate mode calibration normalization. `ssd` applies x/128-1.",
    )
    parser.add_argument(
        "--output",
        default="output.tflite",
        help="Output TFLite path.",
    )
    parser.add_argument(
        "--allow-dynamic",
        action="store_true",
        help="Allow dynamic tensor shapes in the generated TFLite model. This is not suitable for Edge TPU compilation.",
    )
    parser.add_argument(
        "--skip-missing-images",
        action="store_true",
        help="Skip missing AP-10K image files while loading representative samples for quantization.",
    )
    return parser.parse_args()


def _get_savedmodel_input_dtype(model_path: str) -> str:
    try:
        saved_model = tf.saved_model.load(model_path)
        signature = saved_model.signatures.get("serving_default")
        if signature:
            for spec in signature.structured_input_signature[1].values():
                return "uint8" if spec.dtype == tf.uint8 else "float32"
    except Exception:
        pass
    return "float32"


def _load_calibration_arrays(source: str, width: int, height: int, max_samples: int, dtype: str) -> list[np.ndarray]:
    target_dtype = np.uint8 if dtype == "uint8" else np.float32
    source_path = Path(source)
    samples: list[np.ndarray] = []

    if source_path.is_dir():
        files: list[str] = []
        for pattern in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            files.extend(glob.glob(str(source_path / pattern)))
        for image_path in sorted(files)[:max_samples]:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (width, height))
            samples.append(np.expand_dims(image, axis=0).astype(target_dtype))
        return samples

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open representative data source: {source}")
    for _ in range(max_samples):
        ok, frame = capture.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (width, height))
        samples.append(np.expand_dims(frame, axis=0).astype(target_dtype))
    capture.release()
    return samples


def make_array_representative_dataset(dataset: list[np.ndarray]):
    if not dataset:
        raise RuntimeError("No representative samples were loaded for TFLite quantization.")

    def generator():
        for sample in dataset:
            yield [sample]

    return generator, len(dataset)


def make_representative_dataset(config: dict, frame_limit: int):
    dataset = load_samples(config, frame_limit=frame_limit)
    if not dataset:
        raise RuntimeError(
            "No representative samples were loaded for TFLite quantization. "
            "Check the configured data paths; if AP-10K images are missing, "
            "--skip-missing-images can skip missing files but still requires at least one valid sample."
        )

    def generator():
        for sample in dataset:
            yield [sample["input"]]

    return generator, len(dataset)


def validate_static_tflite_model(model_content: bytes) -> None:
    interpreter = tf.lite.Interpreter(model_content=model_content)
    dynamic_tensors = []
    for detail in interpreter.get_input_details() + interpreter.get_output_details():
        shape_signature = detail.get("shape_signature")
        if shape_signature is not None and any(dim < 0 for dim in shape_signature):
            dynamic_tensors.append(
                {
                    "name": detail.get("name"),
                    "shape_signature": [int(dim) for dim in shape_signature],
                }
            )

    if dynamic_tensors:
        formatted = ", ".join(
            f"{item['name']} shape_signature={item['shape_signature']}"
            for item in dynamic_tensors
        )
        raise ValueError(
            "Generated TFLite model has dynamic input/output tensor shapes, which Edge TPU compilation rejects. "
            "Set matching `resize` and `fixed_input_shape` values in the task config, regenerate the split artifacts, "
            f"and rerun conversion. Dynamic tensors: {formatted}. "
            "Use --allow-dynamic only for non-TPU debugging."
        )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    model_path = args.model or config["prefix_saved_model_dir"]
    candidate_rep_mode = args.rep is not None

    if args.rep_video:
        config = dict(config)
        config["data_loader"] = "video_frames"
        config["video_path"] = args.rep_video
    elif args.skip_missing_images:
        config = dict(config)
        config["skip_missing_images"] = True

    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)

    if args.opt != "none":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        if args.opt == "float16":
            converter.target_spec.supported_types = [tf.float16]
        elif args.opt in {"int_fallback", "int8_pure"}:
            if candidate_rep_mode:
                width = args.width or config.get("resize", [320, 320])[0]
                height = args.height or config.get("resize", [320, 320])[1]
                max_calib = args.max_calib or args.frame_limit
                calib_dtype = _get_savedmodel_input_dtype(model_path) if args.calib_dtype == "auto" else args.calib_dtype
                dataset = _load_calibration_arrays(args.rep, width, height, max_calib, calib_dtype)
                if args.calib_normalize == "ssd":
                    dataset = [sample.astype(np.float32) / 128.0 - 1.0 for sample in dataset]
                    calib_dtype = "float32"
                representative_dataset, num_frames = make_array_representative_dataset(dataset)
                data_source = args.rep
            else:
                representative_dataset, num_frames = make_representative_dataset(config, frame_limit=args.frame_limit)
                data_source = config.get("video_path") or config.get("images_dir")
                calib_dtype = "float32"
            print(f"Loaded {num_frames} representative samples from {data_source}")
            converter.representative_dataset = representative_dataset
            if args.opt == "int8_pure":
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.uint8 if calib_dtype == "uint8" else tf.float32
                converter.inference_output_type = tf.float32

    tflite_model = converter.convert()
    if not args.allow_dynamic:
        validate_static_tflite_model(tflite_model)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        handle.write(tflite_model)

    print(f"Saved TFLite model to: {output_path}")


if __name__ == "__main__":
    main()
