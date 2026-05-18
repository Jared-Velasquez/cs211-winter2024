from __future__ import annotations

import argparse
from pathlib import Path

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
        "--frame-limit",
        type=int,
        default=100,
        help="Maximum number of representative frames to load.",
    )
    parser.add_argument(
        "--output",
        default="output.tflite",
        help="Output TFLite path.",
    )
    return parser.parse_args()


def make_representative_dataset(config: dict, frame_limit: int):
    dataset = load_samples(config, frame_limit=frame_limit)
    if not dataset:
        raise RuntimeError("No representative samples were loaded for TFLite quantization.")

    def generator():
        for sample in dataset:
            yield [sample["input"]]

    return generator, len(dataset)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    model_path = args.model or config["prefix_saved_model_dir"]

    if args.rep_video:
        config = dict(config)
        config["data_loader"] = "video_frames"
        config["video_path"] = args.rep_video

    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)

    if args.opt != "none":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        if args.opt == "float16":
            converter.target_spec.supported_types = [tf.float16]
        elif args.opt in {"int_fallback", "int8_pure"}:
            representative_dataset, num_frames = make_representative_dataset(config, frame_limit=args.frame_limit)
            data_source = config.get("video_path") or config.get("images_dir")
            print(f"Loaded {num_frames} representative samples from {data_source}")
            converter.representative_dataset = representative_dataset
            if args.opt == "int8_pure":
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.float32
                converter.inference_output_type = tf.float32

    tflite_model = converter.convert()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        handle.write(tflite_model)

    print(f"Saved TFLite model to: {output_path}")


if __name__ == "__main__":
    main()
