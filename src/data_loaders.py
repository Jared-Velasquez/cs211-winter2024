from __future__ import annotations

import base64
import json
import zlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def load_video_frames(video_path: str, resize: list[int] | None = None, frame_limit: int | None = None) -> list[dict[str, Any]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    samples: list[dict[str, Any]] = []
    frame_index = 0
    while True:
        if frame_limit is not None and frame_index >= frame_limit:
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        original_shape = list(frame.shape)
        if resize:
            width, height = resize
            frame = cv2.resize(frame, (width, height))

        samples.append(
            {
                "input": np.expand_dims(frame.astype(np.float32), axis=0),
                "sample_id": f"frame_{frame_index:06d}",
                "sample_index": frame_index,
                "original_shape": original_shape,
                "input_shape": list(frame.shape),
            }
        )
        frame_index += 1

    cap.release()
    return samples


def _load_image_as_rgb(image_path: Path, resize: list[int] | None = None) -> tuple[np.ndarray, list[int]]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to load image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_shape = list(image.shape)
    if resize:
        width, height = resize
        image = cv2.resize(image, (width, height))
    return image.astype(np.float32), original_shape


def load_coco_images(
    images_dir: str,
    annotations_path: str,
    resize: list[int] | None = None,
    frame_limit: int | None = None,
) -> list[dict[str, Any]]:
    images_root = Path(images_dir)
    annotations = json.loads(Path(annotations_path).read_text(encoding="utf-8"))

    annotations_by_image_id: dict[int, list[dict[str, Any]]] = {}
    for annotation in annotations.get("annotations", []):
        annotations_by_image_id.setdefault(annotation["image_id"], []).append(annotation)

    samples: list[dict[str, Any]] = []
    for sample_index, image_info in enumerate(annotations.get("images", [])):
        if frame_limit is not None and sample_index >= frame_limit:
            break

        image_path = images_root / image_info["file_name"]
        image, original_shape = _load_image_as_rgb(image_path, resize=resize)
        samples.append(
            {
                "input": np.expand_dims(image, axis=0),
                "sample_id": image_info["file_name"],
                "sample_index": sample_index,
                "image_id": image_info["id"],
                "original_shape": original_shape,
                "input_shape": list(image.shape),
                "annotations": annotations_by_image_id.get(image_info["id"], []),
                "source_path": str(image_path),
            }
        )
    return samples


def load_ap10k_pose_dataset(
    images_dir: str,
    annotations_path: str,
    resize: list[int] | None = None,
    frame_limit: int | None = None,
    require_single_instance: bool = True,
    annotation_strategy: str = "largest_instance",
) -> list[dict[str, Any]]:
    images_root = Path(images_dir)
    annotations = json.loads(Path(annotations_path).read_text(encoding="utf-8"))

    image_by_id = {image["id"]: image for image in annotations.get("images", [])}
    category_by_id = {category["id"]: category for category in annotations.get("categories", [])}
    annotations_by_image_id: dict[int, list[dict[str, Any]]] = {}
    for annotation in annotations.get("annotations", []):
        annotations_by_image_id.setdefault(annotation["image_id"], []).append(annotation)

    samples: list[dict[str, Any]] = []
    for image_id, image_info in image_by_id.items():
        image_annotations = annotations_by_image_id.get(image_id, [])
        if not image_annotations:
            continue
        if require_single_instance and len(image_annotations) != 1:
            continue

        if annotation_strategy == "largest_instance":
            annotation = max(image_annotations, key=lambda ann: ann.get("area", 0.0))
        else:
            annotation = image_annotations[0]

        image_path = images_root / image_info["file_name"]
        image, original_shape = _load_image_as_rgb(image_path, resize=resize)
        keypoints = np.asarray(annotation.get("keypoints", []), dtype=np.float32).reshape(-1, 3)
        category = category_by_id[annotation["category_id"]]

        samples.append(
            {
                "input": np.expand_dims(image, axis=0),
                "sample_id": image_info["file_name"],
                "sample_index": len(samples),
                "image_id": image_id,
                "original_shape": original_shape,
                "input_shape": list(image.shape),
                "annotation": annotation,
                "bbox": annotation.get("bbox"),
                "gt_keypoints": keypoints,
                "category_name": category["name"],
                "gt_keypoint_names": category.get("keypoints", []),
                "source_path": str(image_path),
            }
        )
        if frame_limit is not None and len(samples) >= frame_limit:
            break

    return samples


VOC_CLASS_NAME_TO_ID = {
    "aeroplane": 1,
    "bicycle": 2,
    "bird": 3,
    "boat": 4,
    "bottle": 5,
    "bus": 6,
    "car": 7,
    "cat": 8,
    "chair": 9,
    "cow": 10,
    "diningtable": 11,
    "dog": 12,
    "horse": 13,
    "motorbike": 14,
    "person": 15,
    "pottedplant": 16,
    "sheep": 17,
    "sofa": 18,
    "train": 19,
    "tvmonitor": 20,
    "neutral": 255,
}


def _decode_datasetninja_bitmap(bitmap: dict[str, Any]) -> np.ndarray:
    compressed = base64.b64decode(bitmap["data"])
    png_bytes = zlib.decompress(compressed)
    encoded = np.frombuffer(png_bytes, dtype=np.uint8)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise ValueError("Failed to decode DatasetNinja bitmap payload.")

    if decoded.ndim == 2:
        return decoded > 0
    if decoded.shape[2] == 4:
        return decoded[:, :, 3] > 0
    return decoded[:, :, 0] > 0


def _datasetninja_json_to_mask(annotation_path: Path, class_name_to_id: dict[str, int]) -> np.ndarray:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    height = payload["size"]["height"]
    width = payload["size"]["width"]
    mask = np.zeros((height, width), dtype=np.uint8)

    objects = payload.get("objects", [])
    objects = sorted(objects, key=lambda obj: obj.get("classTitle") != "neutral")
    for obj in objects:
        class_title = obj.get("classTitle", "")
        class_id = class_name_to_id.get(class_title, 255)
        bitmap = obj.get("bitmap")
        if not bitmap:
            continue
        binary_mask = _decode_datasetninja_bitmap(bitmap)
        origin_x, origin_y = bitmap["origin"]
        mask_h, mask_w = binary_mask.shape
        target = mask[origin_y:origin_y + mask_h, origin_x:origin_x + mask_w]
        target[binary_mask] = class_id

    return mask


def load_voc_segmentation_dataset(
    images_dir: str,
    annotations_dir: str,
    resize: list[int] | None = None,
    frame_limit: int | None = None,
) -> list[dict[str, Any]]:
    images_root = Path(images_dir)
    annotations_root = Path(annotations_dir)

    image_paths = sorted(path for path in images_root.iterdir() if path.is_file())
    samples: list[dict[str, Any]] = []
    for sample_index, image_path in enumerate(image_paths):
        if frame_limit is not None and sample_index >= frame_limit:
            break

        annotation_path = annotations_root / f"{image_path.name}.json"
        image, original_shape = _load_image_as_rgb(image_path, resize=resize)
        label_mask = _datasetninja_json_to_mask(annotation_path, VOC_CLASS_NAME_TO_ID)

        samples.append(
            {
                "input": np.expand_dims(image, axis=0),
                "sample_id": image_path.name,
                "sample_index": sample_index,
                "original_shape": original_shape,
                "input_shape": list(image.shape),
                "label_mask": label_mask,
                "annotation_path": str(annotation_path),
                "source_path": str(image_path),
            }
        )
    return samples


def load_samples(config: dict[str, Any], frame_limit: int | None = None) -> list[dict[str, Any]]:
    loader = config.get("data_loader", "video_frames")
    if loader == "video_frames":
        return load_video_frames(
            config["video_path"],
            resize=config.get("resize"),
            frame_limit=frame_limit,
        )
    if loader == "coco_images":
        return load_coco_images(
            config["images_dir"],
            config["annotations_path"],
            resize=config.get("resize"),
            frame_limit=frame_limit,
        )
    if loader == "voc_segmentation":
        return load_voc_segmentation_dataset(
            config["images_dir"],
            config["annotations_dir"],
            resize=config.get("resize"),
            frame_limit=frame_limit,
        )
    if loader == "ap10k_pose":
        return load_ap10k_pose_dataset(
            config["images_dir"],
            config["annotations_path"],
            resize=config.get("resize"),
            frame_limit=frame_limit,
            require_single_instance=config.get("require_single_instance", True),
            annotation_strategy=config.get("annotation_strategy", "largest_instance"),
        )

    raise ValueError(f"Unsupported data loader '{loader}'.")
