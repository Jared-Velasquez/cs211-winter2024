from __future__ import annotations

import argparse
from collections import deque

from src.config_utils import load_config
from src.graph_utils import import_graph, load_graph_def
from src.io_utils import save_json


POTENTIALLY_TPU_COMPATIBLE_OP_TYPES = {
    "Add",
    "AddV2",
    "AvgPool",
    "BiasAdd",
    "ConcatV2",
    "Conv2D",
    "DepthwiseConv2dNative",
    "ExpandDims",
    "FusedBatchNorm",
    "FusedBatchNormV3",
    "Identity",
    "MatMul",
    "MaxPool",
    "Mean",
    "Mul",
    "Pack",
    "Pad",
    "Relu",
    "Relu6",
    "Reshape",
    "ResizeBilinear",
    "Rsqrt",
    "Shape",
    "Sigmoid",
    "Softmax",
    "Squeeze",
    "StridedSlice",
    "Sub",
    "Sum",
    "Tanh",
    "Transpose",
}

SKIP_CANDIDATE_OP_TYPES = {
    "Assert",
    "Assign",
    "Const",
    "NoOp",
    "Placeholder",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Enumerate graph partition candidates and report basic TPU-compatibility scaffolding. "
            "This script intentionally does not choose a best split."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/task_a_dlc.json",
        help="Path to the task config JSON.",
    )
    parser.add_argument(
        "--compatible-only",
        action="store_true",
        help="Only emit candidate tensors from potentially TPU-compatible ops.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Optional limit on the number of emitted candidates.",
    )
    return parser.parse_args()


def is_potentially_tpu_compatible_op(op) -> bool:
    return op.type in POTENTIALLY_TPU_COMPATIBLE_OP_TYPES


def find_all_potentially_tpu_compatible_ops(graph):
    return [op for op in graph.get_operations() if is_potentially_tpu_compatible_op(op)]


def find_max_tpu_compatible_backward_closure(graph) -> dict:
    compatible_ops = find_all_potentially_tpu_compatible_ops(graph)
    if not compatible_ops:
        return {
            "size": 0,
            "seed_op": None,
            "op_names": [],
        }

    best_seed = None
    best_visited: set[str] = set()

    for seed in compatible_ops:
        visited: set[str] = set()
        queue = deque([seed])

        while queue:
            op = queue.popleft()
            if op.name in visited:
                continue
            if not is_potentially_tpu_compatible_op(op):
                continue

            visited.add(op.name)
            for input_tensor in op.inputs:
                producer = input_tensor.op
                if producer.name not in visited:
                    queue.append(producer)

        if len(visited) > len(best_visited):
            best_seed = seed.name
            best_visited = visited

    return {
        "size": len(best_visited),
        "seed_op": best_seed,
        "op_names": sorted(best_visited),
    }


def enumerate_partition_candidates(graph, compatible_only: bool = False, max_candidates: int | None = None) -> list[dict]:
    candidates = []

    for index, op in enumerate(graph.get_operations()):
        if op.type in SKIP_CANDIDATE_OP_TYPES:
            continue
        if compatible_only and not is_potentially_tpu_compatible_op(op):
            continue
        if not op.outputs:
            continue

        output_tensors = []
        for tensor in op.outputs:
            shape = tensor.shape.as_list() if tensor.shape.rank is not None else None
            output_tensors.append(
                {
                    "name": tensor.name,
                    "shape": shape,
                    "dtype": tensor.dtype.name,
                }
            )

        candidate = {
            "partition_id": f"candidate_{index}_{op.name.replace('/', '_')}",
            "op_name": op.name,
            "op_type": op.type,
            "topological_index": index,
            "num_inputs": len(op.inputs),
            "num_outputs": len(op.outputs),
            "potential_tpu_compatible": is_potentially_tpu_compatible_op(op),
            "output_tensors": output_tensors,
        }
        candidates.append(candidate)

        if max_candidates is not None and len(candidates) >= max_candidates:
            break

    return candidates


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    graph = import_graph(load_graph_def(config["model_path"]))

    compatible_ops = find_all_potentially_tpu_compatible_ops(graph)
    max_subgraph = find_max_tpu_compatible_backward_closure(graph)
    candidates = enumerate_partition_candidates(
        graph,
        compatible_only=args.compatible_only,
        max_candidates=args.max_candidates,
    )

    payload = {
        "task_name": config["task_name"],
        "model_path": config["model_path"],
        "num_ops": len(graph.get_operations()),
        "num_potentially_tpu_compatible_ops": len(compatible_ops),
        "max_tpu_compatible_backward_closure": max_subgraph,
        "num_candidates": len(candidates),
        "candidates": candidates,
    }
    save_json(config["candidate_output_path"], payload)

    print(f"Enumerated {len(candidates)} candidate partition points.")
    print(f"Potentially TPU-compatible ops: {len(compatible_ops)} / {len(graph.get_operations())}")
    print(
        "Largest TPU-compatible backward closure:",
        max_subgraph["size"],
        "ops",
        f"(seed={max_subgraph['seed_op']})",
    )
    print(f"Saved candidate list to: {config['candidate_output_path']}")


if __name__ == "__main__":
    main()
