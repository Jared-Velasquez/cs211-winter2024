from __future__ import annotations

import argparse

import tensorflow as tf

from src.config_utils import load_config
from src.data_loaders import load_video_frames


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
        help="Representative video to use for int quantization. Defaults to the task video path.",
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


def make_representative_dataset(video_path: str, resize: list[int] | None, frame_limit: int):
    dataset = load_video_frames(video_path, resize=resize, frame_limit=frame_limit)

    def generator():
        for sample in dataset:
            yield [sample["input"]]

    return generator, len(dataset)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    model_path = args.model or config["prefix_saved_model_dir"]
    rep_video = args.rep_video or config["video_path"]

    if args.opt in {"int_fallback", "int8_pure"} and not rep_video:
        raise ValueError("A representative video is required for full int quantization.")

    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)

    if args.opt != "none":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        if args.opt == "float16":
            converter.target_spec.supported_types = [tf.float16]
        elif args.opt in {"int_fallback", "int8_pure"}:
            representative_dataset, num_frames = make_representative_dataset(
                rep_video,
                resize=config.get("resize"),
                frame_limit=args.frame_limit,
            )
            print(f"Loaded {num_frames} representative frames from {rep_video}")
            converter.representative_dataset = representative_dataset
            if args.opt == "int8_pure":
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.float32
                converter.inference_output_type = tf.float32

    tflite_model = converter.convert()
    with open(args.output, "wb") as handle:
        handle.write(tflite_model)

    print(f"Saved TFLite model to: {args.output}")


if __name__ == "__main__":
    main()
