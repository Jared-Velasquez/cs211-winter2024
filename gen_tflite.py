from __future__ import annotations

import argparse
import shutil

from src.config_utils import get_boundary_tensors, load_config
from src.graph_utils import extract_prefix_graph_def, export_saved_model_from_graph_def
from src.io_utils import ensure_directory, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a prefix subgraph from a frozen .pb and export it as a SavedModel."
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
        help="Override boundary tensor names for this export.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the output SavedModel directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing SavedModel directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    boundary_tensors = get_boundary_tensors(config, override=args.boundary_tensors)
    output_dir = args.output_dir or config["prefix_saved_model_dir"]

    if args.force:
        shutil.rmtree(output_dir, ignore_errors=True)
    elif ensure_directory(output_dir).exists() and any(ensure_directory(output_dir).iterdir()):
        raise FileExistsError(f"Output directory already exists: {output_dir}. Use --force to overwrite it.")

    prefix_graph_def = extract_prefix_graph_def(
        graph_path=config["model_path"],
        boundary_tensors=boundary_tensors,
        input_tensor_name=config["input_tensor"],
        fixed_input_shape=config.get("fixed_input_shape"),
    )
    metadata = export_saved_model_from_graph_def(
        prefix_graph_def,
        export_dir=output_dir,
        input_tensor_name=config["input_tensor"],
        output_tensor_names=boundary_tensors,
    )
    metadata["boundary_tensors"] = boundary_tensors
    metadata["resize"] = config.get("resize")
    metadata["fixed_input_shape"] = config.get("fixed_input_shape")
    metadata_path = f"{output_dir.rstrip('/')}_metadata.json"
    save_json(metadata_path, metadata)

    print(f"Saved prefix SavedModel to: {output_dir}")
    print(f"Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    main()
