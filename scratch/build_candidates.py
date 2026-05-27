"""Master driver: for each model, extract TPU + CPU SavedModels per
candidate, compute metadata, write metadata.json.

Run from repo root:
    venv/bin/python scratch/build_candidates.py [dlc|ssd|deeplab|all]
"""
import argparse
import json
import os
import sys
from collections import deque

import tensorflow as tf

sys.path.insert(0, "/Users/jaredvelasquez/projects/cs211-winter2024")
from gen_tflite import extract as extract_tpu
from extract_cpu_subgraph import extract as extract_cpu

tf1 = tf.compat.v1
REPO = "/Users/jaredvelasquez/projects/cs211-winter2024"


# ---------- candidate definitions ----------

DLC_CANDIDATES = {
    "model_path": f"{REPO}/snapshot-1000.pb",
    "model_id": "dlc_resnet50",
    "id_prefix": "dlc_",
    "input_tensor": "Placeholder",
    "input_shape": [1, 320, 320, 3],
    "final_outputs": ["concat_1"],
    "candidates": [
        {
            "id": "split_after_block1",
            "boundary": ["resnet_v1_50/block1/unit_3/bottleneck_v1/Relu"],
            "rationale": "End of ResNet stage 1; small TPU footprint, 256-channel boundary at 80x80.",
        },
        {
            "id": "split_after_block2",
            "boundary": ["resnet_v1_50/block2/unit_4/bottleneck_v1/Relu"],
            "rationale": "End of ResNet stage 2; 512-channel boundary at 40x40.",
        },
        {
            "id": "split_after_block3",
            "boundary": ["resnet_v1_50/block3/unit_6/bottleneck_v1/Relu"],
            "rationale": "End of ResNet stage 3; last clean cut before block4's dilated convs (which introduce non-TPU SpaceToBatchND/BatchToSpaceND ops).",
        },
        {
            "id": "split_in_block4_unit1",
            "boundary": ["resnet_v1_50/block4/unit_1/bottleneck_v1/Relu"],
            "rationale": "End of block4 unit 1; TPU subgraph now spans dilated-conv pattern (red SpaceToBatchND/BatchToSpaceND ops upstream).",
        },
        {
            "id": "split_after_block4",
            "boundary": ["resnet_v1_50/block4/unit_3/bottleneck_v1/Relu"],
            "rationale": "End of backbone; entire dilated-conv zone on TPU side.",
        },
        {
            "id": "split_at_conv2d_transpose",
            "boundary": [
                "pose/part_pred/block4/conv2d_transpose",
                "pose/locref_pred/block4/conv2d_transpose",
            ],
            "rationale": "Output of the pose-head transposed conv, one step before BiasAdd.",
        },
        {
            "id": "split_at_bias_add",
            "boundary": [
                "pose/part_pred/block4/BiasAdd",
                "pose/locref_pred/block4/BiasAdd",
            ],
            "rationale": "Existing baseline (matches the original split.py); maximal TPU partition.",
        },
    ],
}

SSD_CANDIDATES = {
    "model_path": f"{REPO}/artifacts/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb",
    "model_id": "ssd_mobilenet_v2",
    "id_prefix": "ssd_",
    "input_tensor": "image_tensor",
    "input_shape": [1, 300, 300, 3],
    "final_outputs": [
        "detection_boxes",
        "detection_scores",
        "detection_classes",
        "num_detections",
    ],
    "candidates": [
        {
            "id": "split_after_expanded_conv_5",
            "boundary": ["FeatureExtractor/MobilenetV2/expanded_conv_5/output"],
            "shape_override": [[1, 38, 38, 32]],
            "rationale": "Early MobileNet block exit; small TPU partition, 32-channel boundary at 38x38.",
        },
        {
            "id": "split_after_expanded_conv_10",
            "boundary": ["FeatureExtractor/MobilenetV2/expanded_conv_10/output"],
            "shape_override": [[1, 19, 19, 96]],
            "rationale": "Mid MobileNet block exit; 96-channel boundary at 19x19.",
        },
        {
            "id": "split_after_expanded_conv_13",
            "boundary": ["FeatureExtractor/MobilenetV2/expanded_conv_13/output"],
            "shape_override": [[1, 10, 10, 160]],
            "rationale": "First SSD feature-pyramid level after downsample; 160-channel boundary at 10x10.",
        },
        {
            "id": "split_after_conv_1",
            "boundary": ["FeatureExtractor/MobilenetV2/Conv_1/Relu6"],
            "shape_override": [[1, 10, 10, 1280]],
            "rationale": "Post-backbone projection; 1280-channel boundary at 10x10.",
        },
        {
            "id": "split_at_box_predictor_biasadds",
            "boundary": [
                "BoxPredictor_0/ClassPredictor/BiasAdd",
                "BoxPredictor_0/BoxEncodingPredictor/BiasAdd",
                "BoxPredictor_1/ClassPredictor/BiasAdd",
                "BoxPredictor_1/BoxEncodingPredictor/BiasAdd",
                "BoxPredictor_2/ClassPredictor/BiasAdd",
                "BoxPredictor_2/BoxEncodingPredictor/BiasAdd",
                "BoxPredictor_3/ClassPredictor/BiasAdd",
                "BoxPredictor_3/BoxEncodingPredictor/BiasAdd",
                "BoxPredictor_4/ClassPredictor/BiasAdd",
                "BoxPredictor_4/BoxEncodingPredictor/BiasAdd",
                "BoxPredictor_5/ClassPredictor/BiasAdd",
                "BoxPredictor_5/BoxEncodingPredictor/BiasAdd",
            ],
            "shape_override": [
                [1, 19, 19, 273], [1, 19, 19, 12],
                [1, 10, 10, 546], [1, 10, 10, 24],
                [1, 5, 5, 546],   [1, 5, 5, 24],
                [1, 3, 3, 546],   [1, 3, 3, 24],
                [1, 2, 2, 546],   [1, 2, 2, 24],
                [1, 1, 1, 546],   [1, 1, 1, 24],
            ],
            "rationale": "All 12 SSD prediction-head BiasAdds (6 box, 6 class). Wide frontier across the 6 feature-pyramid levels (19, 10, 5, 3, 2, 1).",
        },
        {
            "id": "split_before_postprocessor",
            "boundary": ["Squeeze", "concat_1"],
            "rationale": "Natural cut just before Postprocessor (decode + NMS) — matches what Coral's pre-compiled ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite does; 1917x4 boxes + 1917x91 class scores.",
        },
    ],
}

DEEPLAB_CANDIDATES = {
    "model_path": f"{REPO}/artifacts/deeplabv3_mnv2_pascal_trainval/frozen_inference_graph.pb",
    "model_id": "deeplab_v3_mnv2",
    "id_prefix": "deeplab_",
    "input_tensor": "ImageTensor",
    "input_shape": [1, 513, 513, 3],
    "final_outputs": ["SemanticPredictions"],
    "candidates": [
        {
            "id": "split_after_expanded_conv_5",
            "boundary": ["MobilenetV2/expanded_conv_5/output"],
            "rationale": "Early MobileNet backbone block exit; 32-channel boundary at 65x65.",
        },
        {
            "id": "split_after_expanded_conv_10",
            "boundary": ["MobilenetV2/expanded_conv_10/output"],
            "rationale": "Mid backbone exit; 96-channel boundary.",
        },
        {
            "id": "split_after_expanded_conv_13",
            "boundary": ["MobilenetV2/expanded_conv_13/output"],
            "rationale": "Late backbone exit; 160-channel boundary at 65x65.",
        },
        {
            "id": "split_after_expanded_conv_16",
            "boundary": ["MobilenetV2/expanded_conv_16/output"],
            "rationale": "End of MobileNet backbone; 320-channel boundary at 65x65 (before ASPP).",
        },
        {
            "id": "split_after_aspp",
            "boundary": ["concat_projection/Relu"],
            "rationale": "After ASPP module + 1x1 concat projection; 256-channel boundary at 65x65.",
        },
        {
            "id": "split_after_logits",
            "boundary": ["logits/semantic/BiasAdd"],
            "rationale": "After final 1x1 prediction conv; 21-channel logits at 65x65 (before final upsample).",
        },
        {
            "id": "split_after_resize",
            "boundary": ["ResizeBilinear_3"],
            "rationale": "After the final bilinear upsample to input resolution; 21-channel logits at 513x513 (before ArgMax).",
        },
    ],
}


# ---------- helpers ----------

def load_graph(path):
    gdef = tf1.GraphDef()
    with tf1.io.gfile.GFile(path, "rb") as f:
        gdef.ParseFromString(f.read())
    g = tf.Graph()
    with g.as_default():
        tf.graph_util.import_graph_def(gdef, name="")
    return g


def count_savedmodel_ops(savedmodel_dir):
    loaded = tf.Graph()
    with loaded.as_default():
        with tf1.Session() as sess:
            tf1.saved_model.loader.load(sess, ["serve"], savedmodel_dir)
            return len(loaded.get_operations())


def boundary_shapes(graph, input_tensor, input_shape, boundary_tensors):
    """Re-do the input shape replacement to get concrete shapes for boundary tensors."""
    gdef = graph.as_graph_def()
    new_g = tf.Graph()
    with new_g.as_default():
        orig_input = graph.get_tensor_by_name(input_tensor + ":0")
        new_input = tf1.placeholder(
            dtype=orig_input.dtype, shape=input_shape, name=input_tensor)
        tf1.import_graph_def(
            gdef, name="", input_map={input_tensor: new_input})
    shapes = []
    for t_name in boundary_tensors:
        full = t_name if ":" in t_name else t_name + ":0"
        t = new_g.get_tensor_by_name(full)
        shape = t.shape.as_list()
        # Replace None with 1 conservatively (shouldn't happen with fixed input).
        shape = [s if s is not None else 1 for s in shape]
        shapes.append(shape)
    return shapes


def downstream_op_names(graph, start_op_name):
    """BFS downstream from a single op; returns all reachable op names."""
    seen = {start_op_name}
    queue = deque([graph.get_operation_by_name(start_op_name)])
    while queue:
        op = queue.popleft()
        for out in op.outputs:
            for consumer in out.consumers():
                if consumer.name not in seen:
                    seen.add(consumer.name)
                    queue.append(consumer)
    return seen


def detect_skip_crossing(graph, input_tensor, boundary_tensor_names, upstream_set):
    """A skip crossing = an edge (X -> Y) where:
      - X is on a compute path from model input to a boundary tensor (i.e.
        in upstream_set AND downstream of the input)
      - Y is NOT on any such path (i.e. CPU side)
      - the carried tensor is not in the boundary list

    This filters out shared Consts and shape-ops that fan out to both
    sides — those are not real skip connections.
    """
    boundary_set = set(t if ":" in t else t + ":0" for t in boundary_tensor_names)
    downstream_of_input = downstream_op_names(graph, input_tensor)
    # Compute path ops: upstream of boundary AND downstream of input.
    compute_path = upstream_set & downstream_of_input
    for op_name in compute_path:
        op = graph.get_operation_by_name(op_name)
        for out in op.outputs:
            for consumer in out.consumers():
                if consumer.name not in compute_path:
                    if out.name not in boundary_set:
                        return True
    return False


def upstream_op_names(graph, boundary_tensor_names):
    """BFS upstream from each boundary tensor's producing op; collect all reachable op names."""
    seen = set()
    queue = deque()
    for t_name in boundary_tensor_names:
        op_name = t_name.split(":")[0]
        op = graph.get_operation_by_name(op_name)
        if op.name not in seen:
            seen.add(op.name)
            queue.append(op)
    while queue:
        op = queue.popleft()
        for inp in op.inputs:
            up = inp.op
            if up.name not in seen:
                seen.add(up.name)
                queue.append(up)
        for ctrl in op.control_inputs:
            if ctrl.name not in seen:
                seen.add(ctrl.name)
                queue.append(ctrl)
    return seen


def prod(seq):
    out = 1
    for x in seq:
        out *= x
    return out


# ---------- per-model driver ----------

def process_model(spec, dry_run=False):
    print(f"\n{'=' * 70}\n=== Model: {spec['model_id']}  ({spec['model_path']})\n{'=' * 70}")
    g = load_graph(spec["model_path"])
    print(f"  Loaded {len(g.get_operations())} ops")

    results = []
    for c in spec["candidates"]:
        partition_id = c["id"]
        artifact_dir = f"{REPO}/artifacts/{spec['id_prefix']}{partition_id}"
        tpu_dir = f"{artifact_dir}/tpu_savedmodel"
        cpu_dir = f"{artifact_dir}/cpu_savedmodel"
        meta_path = f"{artifact_dir}/metadata.json"

        print(f"\n--- {partition_id} ---")
        print(f"    boundary: {c['boundary']}")

        try:
            if not dry_run:
                extract_tpu(
                    model_path=spec["model_path"],
                    input_tensor=spec["input_tensor"],
                    input_shape=spec["input_shape"],
                    output_tensors=c["boundary"],
                    output_dir=tpu_dir,
                )
                extract_cpu(
                    model_path=spec["model_path"],
                    boundary_tensors=c["boundary"],
                    final_output_tensor=spec["final_outputs"],
                    output_dir=cpu_dir,
                )
        except Exception as e:
            print(f"    EXTRACT FAILED: {type(e).__name__}: {e}")
            results.append({"id": partition_id, "status": "FAILED", "error": str(e)})
            continue

        try:
            n_tpu = count_savedmodel_ops(tpu_dir)
            n_cpu = count_savedmodel_ops(cpu_dir)
            shapes = c.get("shape_override") or boundary_shapes(
                g, spec["input_tensor"], spec["input_shape"], c["boundary"])
            up_ops = upstream_op_names(g, c["boundary"])
            has_skip = detect_skip_crossing(g, spec["input_tensor"], c["boundary"], up_ops)
            bandwidth_bytes = sum(prod(s) for s in shapes)  # int8 = 1 byte/element
            tpu_tensors_full = [t if ":" in t else t + ":0" for t in c["boundary"]]

            metadata = {
                "model": spec["model_id"],
                "partition_id": partition_id,
                "rationale": c["rationale"],
                "tpu_output_tensors": tpu_tensors_full,
                "cpu_input_tensors": tpu_tensors_full,
                "tpu_tflite_path": f"artifacts/{spec['id_prefix']}{partition_id}/tpu_edgetpu.tflite",
                "cpu_graph_path": f"artifacts/{spec['id_prefix']}{partition_id}/cpu_savedmodel/saved_model.pb",
                "num_tpu_ops": n_tpu,
                "num_cpu_ops": n_cpu,
                "boundary_tensor_shapes": shapes,
                "boundary_bandwidth_bytes": bandwidth_bytes,
                "has_skip_crossing": bool(has_skip),
                "quant_mode": None,
            }
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)
            print(f"    OK  num_tpu_ops={n_tpu}  num_cpu_ops={n_cpu}  shapes={shapes}  bandwidth={bandwidth_bytes}B  skip={has_skip}")
            results.append({"id": partition_id, "status": "OK", **metadata})
        except Exception as e:
            print(f"    METADATA FAILED: {type(e).__name__}: {e}")
            results.append({"id": partition_id, "status": "META_FAILED", "error": str(e)})

    print(f"\n=== Summary for {spec['model_id']} ===")
    for r in results:
        if r["status"] == "OK":
            print(f"  ✓ {r['id']:40s} tpu={r['num_tpu_ops']:4d} cpu={r['num_cpu_ops']:4d} bw={r['boundary_bandwidth_bytes']:>9d}B skip={r['has_skip_crossing']}")
        else:
            print(f"  ✗ {r['id']:40s} {r['status']}: {r.get('error', '')[:120]}")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", nargs="?", default="all",
                   choices=["dlc", "ssd", "deeplab", "all"])
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    by_name = {
        "dlc": DLC_CANDIDATES,
        "ssd": SSD_CANDIDATES,
        "deeplab": DEEPLAB_CANDIDATES,
    }
    todo = list(by_name.keys()) if args.model == "all" else [args.model]

    all_results = {}
    for name in todo:
        all_results[name] = process_model(by_name[name], dry_run=args.dry_run)

    print(f"\n\n{'=' * 70}\n=== GRAND SUMMARY\n{'=' * 70}")
    for name, results in all_results.items():
        oks = sum(1 for r in results if r["status"] == "OK")
        print(f"  {name}: {oks}/{len(results)} OK")


if __name__ == "__main__":
    main()
