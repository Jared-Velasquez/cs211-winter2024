from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def tensor_name_to_key(name: str) -> str:
    return name.replace(":", "_").replace("/", "_")


def squeeze_leading_batch(array: np.ndarray) -> np.ndarray:
    if array.ndim > 0 and array.shape[0] == 1:
        return array[0]
    return array


def stack_named_outputs(per_sample_outputs: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not per_sample_outputs:
        return {}

    keys = per_sample_outputs[0].keys()
    stacked: dict[str, np.ndarray] = {}
    for key in keys:
        values = [squeeze_leading_batch(sample[key]) for sample in per_sample_outputs]
        try:
            stacked[key] = np.stack(values, axis=0)
        except ValueError:
            stacked[key] = np.array(values, dtype=object)
    return stacked


def save_npz(path: str | Path, tensors: dict[str, np.ndarray]) -> None:
    target = Path(path)
    ensure_directory(target.parent)
    np.savez(target, **tensors)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    ensure_directory(target.parent)
    target.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")
