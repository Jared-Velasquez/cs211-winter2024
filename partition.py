"""
Generalized graph partition script: splits a frozen TF graph into a TPU-runnable prefix
and a CPU-runnable suffix, producing split_metadata.json compatible with hybrid_runner.

Algorithm: forward topological walk over all ops; any tensor produced by a TPU-compatible
op whose output is consumed by a CPU-only op becomes a boundary tensor.

Usage (Task B):
  python partition.py \
    --pb-path data/task_b/models/frozen_inference_graph.pb \
    --artifacts-dir artifacts/task_b/detection \
    --input-tensor image_tensor:0 \
    --output-tensors "detection_boxes:0,detection_scores:0,detection_classes:0,num_detections:0" \
    --input-shape 1,300,300,3 \
    --task-name task_b_detection

Usage (Task C):
  python partition.py \
    --pb-path data/task_c/models/frozen_inference_graph.pb \
    --artifacts-dir artifacts/task_c/segmentation \
    --input-tensor ImageTensor:0 \
    --output-tensors "SemanticPredictions:0" \
    --input-shape 1,513,513,3 \
    --task-name task_c_segmentation
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import tensorflow as tf

tfv1 = tf.compat.v1

# TF op types supported by the Edge TPU.
# Checked against op.type (the kernel type), not op.name, for reliability across models.
# Source: https://coral.ai/docs/edgetpu/models-intro/#supported-operations
_TPU_OP_TYPES: frozenset[str] = frozenset({
    "Add", "AddV2", "AddN",
    "AvgPool",
    "BiasAdd",
    "BatchMatMul", "BatchMatMulV2",
    "Cast",
    "Concat", "ConcatV2",
    "Conv2D",
    "Conv2DBackpropInput",       # TFLite TransposeConv
    "DepthwiseConv2dNative",
    "DepthToSpace",
    "ExpandDims",
    "FusedBatchNorm", "FusedBatchNormV3",
    "GatherV2",
    "L2Normalize",
    "LeakyRelu",
    "MatMul",
    "Max", "Maximum",
    "MaxPool",
    "Mean",
    "Min", "Minimum",
    "Mul",
    "Pack",
    "Pad", "PadV2", "MirrorPad",
    "Pow",
    "RealDiv",
    "Relu", "Relu6",
    "Reshape",
    "ResizeBilinear", "ResizeNearestNeighbor",
    "Rsqrt", "Sqrt",
    "Sigmoid",
    "Slice", "StridedSlice",
    "Softmax",
    "SpaceToBatchND", "BatchToSpaceND",
    "SpaceToDepth",
    "Split", "SplitV",
    "Squeeze",
    "Sub",
    "Sum",
    "Tanh",
    "Tile",
    "Transpose",
    "Unpack",
    "ZerosLike",
    # Infrastructure — present on both sides; excluded from boundary detection separately.
    "Const", "Identity", "NoOp", "Placeholder", "Shape", "StopGradient",
    "QuantizeV2", "Dequantize",
})

# These op types produce tensors that carry weights/constants, not computed features.
# Exclude them as potential boundary sources so anchor boxes and NMS thresholds
# don't become spurious boundary tensors.
_NON_COMPUTE_OP_TYPES: frozenset[str] = frozenset({
    "Const", "NoOp", "Placeholder", "Identity",
    "Shape", "StopGradient", "QuantizeV2", "Dequantize",
})


def is_tpu_compatible(op: tf.Operation) -> bool:
    return op.type in _TPU_OP_TYPES


def load_graph(pb_path: str) -> tuple[tf.Graph, tfv1.GraphDef]:
    gdef = tfv1.GraphDef()
    with tfv1.io.gfile.GFile(pb_path, "rb") as f:
        gdef.ParseFromString(f.read())
    graph = tf.Graph()
    with graph.as_default():
        tf.graph_util.import_graph_def(gdef, name="")
    return graph, gdef


def find_partition_boundary(graph: tf.Graph) -> list[str]:
    """
    Forward topological walk to find boundary tensors (TPU→CPU interface).

    A tensor qualifies as a boundary tensor when:
      1. Its producing op is TPU-compatible (and not a pure infrastructure op), AND
      2. At least one consuming op is NOT TPU-compatible.

    tf.Graph.get_operations() returns ops in topological order for graphs imported
    via import_graph_def, so no explicit sort is needed.
    """
    boundary: list[str] = []
    for op in graph.get_operations():
        if op.type in _NON_COMPUTE_OP_TYPES:
            continue
        if not is_tpu_compatible(op):
            continue
        for out_tensor in op.outputs:
            for consumer in out_tensor.consumers():
                if not is_tpu_compatible(consumer):
                    if out_tensor.name not in boundary:
                        boundary.append(out_tensor.name)
                    break
    return boundary


def extract_prefix_saved_model(
    gdef: tfv1.GraphDef,
    input_tensor_name: str,
    boundary_tensor_names: list[str],
    export_dir: str,
) -> None:
    """
    Extract the prefix subgraph (inputs → boundary tensors) and save as a SavedModel.

    Uses the same extract_sub_graph + simple_save pattern as gen_tflite.py so the
    prefix can be quantized and compiled for the Edge TPU with existing tooling.
    """
    boundary_op_names = [t.split(":")[0] for t in boundary_tensor_names]
    prefix_gdef = tfv1.graph_util.extract_sub_graph(gdef, boundary_op_names)

    prefix_graph = tf.Graph()
    with prefix_graph.as_default():
        tf.graph_util.import_graph_def(prefix_gdef, name="")

    input_tensor = prefix_graph.get_tensor_by_name(input_tensor_name)
    outputs = {
        t.replace("/", "_").replace(":", "_"): prefix_graph.get_tensor_by_name(t)
        for t in boundary_tensor_names
    }

    os.makedirs(export_dir, exist_ok=True)
    with tfv1.Session(graph=prefix_graph) as sess:
        tfv1.saved_model.simple_save(
            session=sess,
            export_dir=export_dir,
            inputs={"input": input_tensor},
            outputs=outputs,
        )


def build_suffix_graph(
    graph: tf.Graph,
    gdef: tfv1.GraphDef,
    boundary_tensor_names: list[str],
    output_tensor_names: list[str],
) -> tuple[tfv1.GraphDef, dict[str, str]]:
    """
    Build a suffix graph where boundary tensors are replaced by Placeholder inputs.

    The returned placeholder_map matches the split_metadata.json schema used by
    hybrid_runner: {original_tensor_name: placeholder_tensor_name}.
    """
    suffix_graph = tf.Graph()
    placeholder_map: dict[str, str] = {}

    with suffix_graph.as_default():
        input_map: dict[str, tf.Tensor] = {}
        for i, b_name in enumerate(boundary_tensor_names):
            orig = graph.get_tensor_by_name(b_name)
            shape = orig.shape.as_list()
            if shape and shape[0] is not None:
                shape[0] = None  # dynamic batch dimension
            ph = tfv1.placeholder(
                dtype=orig.dtype,
                shape=shape,
                name=f"cpu_inputs/boundary_{i}",
            )
            input_map[b_name] = ph
            placeholder_map[b_name] = ph.name

        tf.graph_util.import_graph_def(gdef, input_map=input_map, name="")

    output_op_names = [t.split(":")[0] for t in output_tensor_names]
    with suffix_graph.as_default():
        suffix_gdef = tfv1.graph_util.extract_sub_graph(
            suffix_graph.as_graph_def(), output_op_names
        )

    return suffix_gdef, placeholder_map


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Partition a frozen TF graph into a TPU prefix and CPU suffix."
    )
    parser.add_argument("--pb-path", required=True, help="Path to frozen_inference_graph.pb")
    parser.add_argument("--artifacts-dir", required=True, help="Output directory")
    parser.add_argument("--input-tensor", required=True, help="e.g. image_tensor:0")
    parser.add_argument(
        "--output-tensors", required=True,
        help="Comma-separated full graph output tensor names, e.g. detection_boxes:0,num_detections:0",
    )
    parser.add_argument(
        "--input-shape", required=True,
        help="Comma-separated ints matching the input tensor shape, e.g. 1,300,300,3",
    )
    parser.add_argument("--task-name", default=None)
    args = parser.parse_args()

    output_tensor_names = [t.strip() for t in args.output_tensors.split(",")]
    input_shape = [int(x) for x in args.input_shape.split(",")]
    task_name = args.task_name or Path(args.artifacts_dir).parent.name
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading graph: {args.pb_path}")
    graph, gdef = load_graph(args.pb_path)
    print(f"  {len(graph.get_operations())} ops loaded")

    print("\nFinding partition boundary (forward topological walk)...")
    boundary_tensors = find_partition_boundary(graph)
    if not boundary_tensors:
        raise RuntimeError(
            "No boundary tensors found. The model may be fully TPU-compatible "
            "or the output tensors may all be TPU-compatible."
        )
    print(f"  {len(boundary_tensors)} boundary tensor(s):")
    for t in boundary_tensors:
        tensor = graph.get_tensor_by_name(t)
        print(f"    {t}  shape={tensor.shape.as_list()}  dtype={tensor.dtype.name}")

    prefix_dir = str(artifacts_dir / "prefix_saved_model")
    suffix_path = str(artifacts_dir / "suffix_graph.pb")

    print(f"\nExtracting prefix SavedModel → {prefix_dir}")
    extract_prefix_saved_model(gdef, args.input_tensor, boundary_tensors, prefix_dir)

    print(f"Building suffix graph → {suffix_path}")
    suffix_gdef, placeholder_map = build_suffix_graph(
        graph, gdef, boundary_tensors, output_tensor_names
    )
    with open(suffix_path, "wb") as f:
        f.write(suffix_gdef.SerializeToString())

    metadata = {
        "task_name": task_name,
        "model_path": str(Path(args.pb_path).resolve()),
        "input_tensor": args.input_tensor,
        "fixed_input_shape": input_shape,
        "output_tensors": output_tensor_names,
        "boundary_tensors": boundary_tensors,
        "prefix_saved_model_dir": prefix_dir,
        "prefix_saved_model": {
            "input_tensor": args.input_tensor,
            "output_tensors": boundary_tensors,
            "saved_model_dir": prefix_dir,
        },
        "suffix_graph_path": suffix_path,
        "suffix_placeholder_map": placeholder_map,
    }
    metadata_path = str(artifacts_dir / "split_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nWrote split_metadata.json → {metadata_path}")
    print("\nNext steps:")
    print(f"  1. Quantize prefix for Edge TPU (adapt gen_tflite.py for {task_name})")
    print(f"     Prefix SavedModel: {prefix_dir}")
    print(f"  2. edgetpu_compiler output_int8.tflite")
    print(f"  3. Run hybrid inference via src/hybrid_runner.py with the split metadata")


if __name__ == "__main__":
    main()
