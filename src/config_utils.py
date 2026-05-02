from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PATH_KEYS = {
    "model_path",
    "video_path",
    "images_dir",
    "annotations_path",
    "annotations_dir",
    "meta_path",
    "tflite_model_path",
    "artifacts_dir",
    "prefix_saved_model_dir",
    "suffix_graph_path",
    "split_metadata_path",
    "baseline_output_path",
    "baseline_summary_path",
    "candidate_output_path",
}


def _resolve_repo_root(config_path: Path) -> Path:
    if config_path.parent.name == "configs":
        return config_path.parent.parent
    return config_path.parent


def _resolve_path(repo_root: Path, value: str) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path.resolve())
    return str((repo_root / path).resolve())


def _with_defaults(config: dict[str, Any]) -> dict[str, Any]:
    artifacts_dir = Path(config["artifacts_dir"])
    config.setdefault("prefix_saved_model_dir", str(artifacts_dir / "prefix_saved_model"))
    config.setdefault("suffix_graph_path", str(artifacts_dir / "suffix_graph.pb"))
    config.setdefault("split_metadata_path", str(artifacts_dir / "split_metadata.json"))
    config.setdefault("baseline_output_path", str(artifacts_dir / "full_graph_outputs.npz"))
    config.setdefault("baseline_summary_path", str(artifacts_dir / "full_graph_summary.json"))
    config.setdefault("baseline_results_path", str(artifacts_dir / "baseline_results.json"))
    config.setdefault("candidate_output_path", str(artifacts_dir / "partition_candidates.json"))
    config.setdefault("data_loader", "video_frames")
    return config


def load_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    repo_root = _resolve_repo_root(path)

    config: dict[str, Any] = dict(data)
    config["_config_path"] = str(path)
    config["_repo_root"] = str(repo_root)

    for key in PATH_KEYS:
        if key in config:
            config[key] = _resolve_path(repo_root, config[key])

    config = _with_defaults(config)
    for key in PATH_KEYS:
        if key in config:
            config[key] = _resolve_path(repo_root, config[key])
    return config


def get_boundary_tensors(config: dict[str, Any], override: list[str] | None = None) -> list[str]:
    if override:
        return list(override)
    boundary_tensors = config.get("boundary_tensors", [])
    if not boundary_tensors:
        raise ValueError("No boundary tensors were provided in the config or CLI.")
    return list(boundary_tensors)
