from __future__ import annotations

import argparse
import os
import shutil

from src.config_utils import get_boundary_tensors, load_config
from src.graph_utils import (
    build_suffix_graph_def,
    extract_prefix_graph_def,
    export_saved_model_from_graph_def,
    write_graph_def,
)
from src.io_utils import ensure_directory, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generalized graph split for the current task config.")
    parser.add_argument(
        "--config",
        default="configs/task_a_dlc.json",
        help="Path to the task config JSON.",
    )
    parser.add_argument(
        "--boundary-tensors",
        nargs="+",
        default=None,
        help="Override boundary tensor names for this split.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite prior split artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    boundary_tensors = get_boundary_tensors(config, override=args.boundary_tensors)

    prefix_dir = config["prefix_saved_model_dir"]
    suffix_graph_path = config["suffix_graph_path"]
    split_metadata_path = config["split_metadata_path"]

    if args.force:
        shutil.rmtree(prefix_dir, ignore_errors=True)
        if os.path.exists(suffix_graph_path):
            os.remove(suffix_graph_path)
        if os.path.exists(split_metadata_path):
            os.remove(split_metadata_path)
    elif any(ensure_directory(prefix_dir).iterdir()):
        raise FileExistsError(f"Prefix SavedModel directory already exists: {prefix_dir}. Use --force to overwrite it.")

    ensure_directory(config["artifacts_dir"])
    shutil.rmtree(prefix_dir, ignore_errors=True)
    prefix_graph_def = extract_prefix_graph_def(
        graph_path=config["model_path"],
        boundary_tensors=boundary_tensors,
        input_tensor_name=config["input_tensor"],
        fixed_input_shape=config.get("fixed_input_shape"),
    )
    prefix_metadata = export_saved_model_from_graph_def(
        prefix_graph_def,
        export_dir=prefix_dir,
        input_tensor_name=config["input_tensor"],
        output_tensor_names=boundary_tensors,
    )

    suffix_graph_def, placeholder_map = build_suffix_graph_def(
        graph_path=config["model_path"],
        boundary_tensors=boundary_tensors,
        output_tensors=list(config["output_tensors"]),
    )
    write_graph_def(suffix_graph_def, suffix_graph_path)

    split_metadata = {
        "task_name": config["task_name"],
        "model_path": config["model_path"],
        "prefix_saved_model_dir": prefix_dir,
        "suffix_graph_path": suffix_graph_path,
        "input_tensor": config["input_tensor"],
        "boundary_tensors": boundary_tensors,
        "suffix_placeholder_map": placeholder_map,
        "output_tensors": list(config["output_tensors"]),
        "resize": config.get("resize"),
        "fixed_input_shape": config.get("fixed_input_shape"),
        "prefix_saved_model": prefix_metadata,
    }
    save_json(split_metadata_path, split_metadata)

    print(f"Saved prefix model to: {prefix_dir}")
    print(f"Saved suffix graph to: {suffix_graph_path}")
    print(f"Saved split metadata to: {split_metadata_path}")


if __name__ == "__main__":
    main()
