from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


TASK_A_DLC_KEYPOINT_NAMES = [
    "nose",
    "upper_jaw",
    "lower_jaw",
    "mouth_end_right",
    "mouth_end_left",
    "right_eye",
    "right_earbase",
    "right_earend",
    "right_antler_base",
    "right_antler_end",
    "left_eye",
    "left_earbase",
    "left_earend",
    "left_antler_base",
    "left_antler_end",
    "neck_base",
    "neck_end",
    "throat_base",
    "throat_end",
    "back_base",
    "back_end",
    "back_middle",
    "tail_base",
    "tail_end",
    "front_left_thai",
    "front_left_knee",
    "front_left_paw",
    "front_right_thai",
    "front_right_knee",
    "front_right_paw",
    "back_left_paw",
    "back_left_thai",
    "back_right_thai",
    "back_left_knee",
    "back_right_knee",
    "back_right_paw",
    "belly_bottom",
    "body_middle_right",
    "body_middle_left",
]


TASK_A_AP10K_TO_DLC = {
    "left_eye": "left_eye",
    "right_eye": "right_eye",
    "nose": "nose",
    "neck": "neck_base",
    "root_of_tail": "tail_base",
    "left_shoulder": "front_left_thai",
    "left_elbow": "front_left_knee",
    "left_front_paw": "front_left_paw",
    "right_shoulder": "front_right_thai",
    "right_elbow": "front_right_knee",
    "right_front_paw": "front_right_paw",
    "left_hip": "back_left_thai",
    "left_knee": "back_left_knee",
    "left_back_paw": "back_left_paw",
    "right_hip": "back_right_thai",
    "right_knee": "back_right_knee",
    "right_back_paw": "back_right_paw",
}


VOC_CLASS_NAMES = {
    0: "background",
    1: "aeroplane",
    2: "bicycle",
    3: "bird",
    4: "boat",
    5: "bottle",
    6: "bus",
    7: "car",
    8: "cat",
    9: "chair",
    10: "cow",
    11: "diningtable",
    12: "dog",
    13: "horse",
    14: "motorbike",
    15: "person",
    16: "pottedplant",
    17: "sheep",
    18: "sofa",
    19: "train",
    20: "tvmonitor",
}


def _output_tensor(outputs: dict[str, np.ndarray], tensor_name: str) -> np.ndarray:
    key = tensor_name.replace(":", "_").replace("/", "_")
    return outputs[key]


def evaluate_pose(samples: list[dict[str, Any]], outputs: dict[str, np.ndarray], config: dict[str, Any]) -> dict[str, Any]:
    predictions = _output_tensor(outputs, config["output_tensors"][0])
    name_to_index = {name: index for index, name in enumerate(TASK_A_DLC_KEYPOINT_NAMES)}
    thresholds = config.get("pck_thresholds", [0.05, 0.10, 0.20])

    distances = []
    correct_by_threshold = {threshold: [] for threshold in thresholds}
    per_keypoint_distances: dict[str, list[float]] = {}

    for sample, prediction in zip(samples, predictions):
        gt_keypoints = sample["gt_keypoints"]
        bbox = sample.get("bbox") or [0, 0, sample["original_shape"][1], sample["original_shape"][0]]
        scale = max(float(bbox[2]), float(bbox[3]), 1.0)
        gt_name_to_index = {name: index for index, name in enumerate(sample["gt_keypoint_names"])}

        # Predictions come out in the resized network-input space (input_shape) and in
        # (row, col) = (y, x) order; ground truth is in original-image space. Map
        # predictions back to original-image space so both, and the bbox normalizer
        # above, share one coordinate frame.
        orig_h, orig_w = sample["original_shape"][0], sample["original_shape"][1]
        in_h, in_w = sample["input_shape"][0], sample["input_shape"][1]
        scale_x, scale_y = orig_w / in_w, orig_h / in_h

        for gt_name, pred_name in TASK_A_AP10K_TO_DLC.items():
            gt_index = gt_name_to_index.get(gt_name)
            pred_index = name_to_index.get(pred_name)
            if gt_index is None or pred_index is None:
                continue

            x, y, visibility = gt_keypoints[gt_index]
            if visibility <= 0:
                continue

            pred_row, pred_col, _ = prediction[pred_index]
            pred_x, pred_y = pred_col * scale_x, pred_row * scale_y
            distance = float(np.linalg.norm(np.array([pred_x - x, pred_y - y], dtype=np.float32)))
            distances.append(distance)
            per_keypoint_distances.setdefault(gt_name, []).append(distance)
            for threshold in thresholds:
                correct_by_threshold[threshold].append(distance <= threshold * scale)

    if not distances:
        return {
            "num_evaluated_samples": len(samples),
            "num_evaluated_keypoints": 0,
            "mapping_note": "No valid mapped keypoints were available for evaluation.",
        }

    return {
        "num_evaluated_samples": len(samples),
        "num_evaluated_keypoints": len(distances),
        "mapping_note": (
            "AP-10K keypoints are evaluated through an approximate AP-10K-to-DLC mapping because "
            "the DLC model predicts 39 landmarks while AP-10K labels 17 landmarks."
        ),
        "rmse": float(math.sqrt(np.mean(np.square(distances)))),
        "pck": {str(threshold): float(np.mean(correct_by_threshold[threshold])) for threshold in thresholds},
        "per_keypoint_rmse": {
            name: float(math.sqrt(np.mean(np.square(values))))
            for name, values in per_keypoint_distances.items()
        },
    }


def _boxes_to_coco_detections(samples: list[dict[str, Any]], outputs: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    boxes = outputs["detection_boxes_0"]
    scores = outputs["detection_scores_0"]
    classes = outputs["detection_classes_0"]
    num_detections = outputs["num_detections_0"]

    detections: list[dict[str, Any]] = []
    for sample, sample_boxes, sample_scores, sample_classes, sample_num in zip(
        samples, boxes, scores, classes, num_detections
    ):
        height, width = sample["original_shape"][:2]
        limit = int(sample_num)
        for box, score, category in zip(sample_boxes[:limit], sample_scores[:limit], sample_classes[:limit]):
            ymin, xmin, ymax, xmax = box.tolist()
            x = xmin * width
            y = ymin * height
            w = (xmax - xmin) * width
            h = (ymax - ymin) * height
            detections.append(
                {
                    "image_id": sample["image_id"],
                    "category_id": int(round(float(category))),
                    "bbox": [x, y, w, h],
                    "score": float(score),
                }
            )
    return detections


def evaluate_detection(samples: list[dict[str, Any]], outputs: dict[str, np.ndarray], config: dict[str, Any]) -> dict[str, Any]:
    detections = _boxes_to_coco_detections(samples, outputs)
    coco_gt = COCO(config["annotations_path"])
    if detections:
        coco_dt = coco_gt.loadRes(detections)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.params.imgIds = [sample["image_id"] for sample in samples]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats = coco_eval.stats.tolist()
    else:
        stats = [0.0] * 12

    return {
        "num_evaluated_samples": len(samples),
        "num_detections": len(detections),
        "map_50_95": float(stats[0]),
        "map_50": float(stats[1]),
        "coco_eval_stats": stats,
    }


def _boundary_mask(mask: np.ndarray, ignore_value: int = 255) -> np.ndarray:
    valid = mask != ignore_value
    safe = mask.copy()
    safe[~valid] = 0
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(safe.astype(np.uint8), kernel, iterations=1)
    boundary = (safe != eroded) & valid
    return boundary.astype(np.uint8)


def _boundary_f1(pred_mask: np.ndarray, gt_mask: np.ndarray, ignore_value: int = 255) -> float:
    pred_boundary = _boundary_mask(pred_mask, ignore_value=ignore_value)
    gt_boundary = _boundary_mask(gt_mask, ignore_value=ignore_value)
    kernel = np.ones((3, 3), dtype=np.uint8)
    pred_dil = cv2.dilate(pred_boundary, kernel, iterations=1)
    gt_dil = cv2.dilate(gt_boundary, kernel, iterations=1)

    pred_sum = int(pred_boundary.sum())
    gt_sum = int(gt_boundary.sum())
    if pred_sum == 0 and gt_sum == 0:
        return 1.0
    if pred_sum == 0 or gt_sum == 0:
        return 0.0

    precision = float((pred_boundary & gt_dil).sum()) / pred_sum
    recall = float((gt_boundary & pred_dil).sum()) / gt_sum
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def evaluate_segmentation(samples: list[dict[str, Any]], outputs: dict[str, np.ndarray], config: dict[str, Any]) -> dict[str, Any]:
    predictions = _output_tensor(outputs, config["output_tensors"][0])
    class_ids = sorted(VOC_CLASS_NAMES.keys())
    ignore_value = 255

    confusion = np.zeros((len(class_ids), len(class_ids)), dtype=np.int64)
    boundary_scores = []

    for sample, pred_mask in zip(samples, predictions):
        gt_mask = sample["label_mask"]
        pred_mask = np.asarray(pred_mask, dtype=np.uint8)
        # Partitioned/hybrid runs emit the segmentation map at the model input size
        # (e.g. 513x513), while the GT label mask is at the original image resolution.
        # Resize the prediction (nearest-neighbour, label-preserving) to the GT shape so
        # per-pixel comparison is valid. The full-graph baseline already emits original
        # size, so this is a no-op there.
        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(
                pred_mask,
                (gt_mask.shape[1], gt_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        valid = gt_mask != ignore_value

        gt_flat = gt_mask[valid].astype(np.int64)
        pred_flat = pred_mask[valid].astype(np.int64)
        for gt_value, pred_value in zip(gt_flat, pred_flat):
            if gt_value in VOC_CLASS_NAMES and pred_value in VOC_CLASS_NAMES:
                confusion[gt_value, pred_value] += 1

        boundary_scores.append(_boundary_f1(pred_mask, gt_mask, ignore_value=ignore_value))

    per_class_iou = {}
    ious = []
    for class_id, class_name in VOC_CLASS_NAMES.items():
        tp = confusion[class_id, class_id]
        fp = confusion[:, class_id].sum() - tp
        fn = confusion[class_id, :].sum() - tp
        denom = tp + fp + fn
        if denom == 0:
            continue
        iou = float(tp / denom)
        per_class_iou[class_name] = iou
        ious.append(iou)

    return {
        "num_evaluated_samples": len(samples),
        "miou": float(np.mean(ious)) if ious else 0.0,
        "per_class_iou": per_class_iou,
        "boundary_f1": float(np.mean(boundary_scores)) if boundary_scores else 0.0,
    }


def evaluate_outputs(config: dict[str, Any], samples: list[dict[str, Any]], outputs: dict[str, np.ndarray]) -> dict[str, Any]:
    task_type = config["task_type"]
    if task_type == "pose_estimation":
        return evaluate_pose(samples, outputs, config)
    if task_type == "object_detection":
        return evaluate_detection(samples, outputs, config)
    if task_type == "semantic_segmentation":
        return evaluate_segmentation(samples, outputs, config)
    raise ValueError(f"Unsupported task_type '{task_type}' for evaluation.")
