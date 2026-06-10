from __future__ import annotations

from pathlib import Path
from typing import Any

import tensorflow as tf

tf1 = tf.compat.v1


def tensor_name_to_op_name(tensor_name: str) -> str:
    return tensor_name.split(":", 1)[0]


def load_graph_def(graph_path: str) -> tf1.GraphDef:
    graph_def = tf1.GraphDef()
    with tf1.io.gfile.GFile(graph_path, "rb") as handle:
        graph_def.ParseFromString(handle.read())
    return graph_def


def import_graph(graph_def: tf1.GraphDef, name: str = "") -> tf.Graph:
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name=name)
    return graph


def get_tensor_shapes(graph_path: str, tensor_names: list[str]) -> dict[str, list[int | None]]:
    graph = import_graph(load_graph_def(graph_path))
    shapes: dict[str, list[int | None]] = {}
    for tensor_name in tensor_names:
        tensor = graph.get_tensor_by_name(tensor_name)
        shapes[tensor_name] = tensor.shape.as_list()
    return shapes


def apply_fixed_input_shape(
    graph_def: tf1.GraphDef,
    input_tensor_name: str,
    fixed_input_shape: list[int] | None,
) -> tf1.GraphDef:
    if not fixed_input_shape:
        return graph_def

    input_op_name = tensor_name_to_op_name(input_tensor_name)
    original_graph = import_graph(graph_def)
    last_error: Exception | None = None

    for key in (input_tensor_name, input_op_name):
        remapped_graph = tf.Graph()
        with remapped_graph.as_default():
            original_input = original_graph.get_tensor_by_name(input_tensor_name)
            new_input = tf1.placeholder(
                dtype=original_input.dtype,
                shape=fixed_input_shape,
                name=input_op_name,
            )
            try:
                tf1.import_graph_def(
                    original_graph.as_graph_def(),
                    name="",
                    input_map={key: new_input},
                )
                return remapped_graph.as_graph_def()
            except Exception as exc:  # noqa: BLE001
                last_error = exc

    raise RuntimeError(
        f"Failed to apply fixed input shape to {input_tensor_name}: {last_error}"
    )


def extract_prefix_graph_def(
    graph_path: str,
    boundary_tensors: list[str],
    input_tensor_name: str,
    fixed_input_shape: list[int] | None = None,
) -> tf1.GraphDef:
    graph_def = load_graph_def(graph_path)
    graph_def = apply_fixed_input_shape(graph_def, input_tensor_name, fixed_input_shape)
    output_op_names = [tensor_name_to_op_name(name) for name in boundary_tensors]
    return tf1.graph_util.extract_sub_graph(graph_def, output_op_names)


def export_saved_model_from_graph_def(
    graph_def: tf1.GraphDef,
    export_dir: str,
    input_tensor_name: str,
    output_tensor_names: list[str],
) -> dict[str, Any]:
    graph = import_graph(graph_def)
    input_tensor = graph.get_tensor_by_name(input_tensor_name)
    outputs = {tensor_name_to_op_name(name).replace("/", "_"): graph.get_tensor_by_name(name) for name in output_tensor_names}

    with tf1.Session(graph=graph) as session:
        tf1.saved_model.simple_save(
            session=session,
            export_dir=export_dir,
            inputs={"input": input_tensor},
            outputs=outputs,
        )

    return {
        "input_tensor": input_tensor_name,
        "output_tensors": output_tensor_names,
        "saved_model_dir": export_dir,
    }


def build_suffix_graph_def(
    graph_path: str,
    boundary_tensors: list[str],
    output_tensors: list[str],
) -> tuple[tf1.GraphDef, dict[str, str]]:
    original_graph = import_graph(load_graph_def(graph_path))
    placeholder_map: dict[str, str] = {}
    input_map = {}

    remapped_graph = tf.Graph()
    with remapped_graph.as_default():
        for index, boundary_tensor in enumerate(boundary_tensors):
            tensor = original_graph.get_tensor_by_name(boundary_tensor)
            placeholder_name = f"cpu_inputs/boundary_{index}"
            placeholder = tf1.placeholder(
                dtype=tensor.dtype,
                shape=tensor.shape.as_list(),
                name=placeholder_name,
            )
            placeholder_map[boundary_tensor] = f"{placeholder_name}:0"
            input_map[boundary_tensor] = placeholder

        tf1.import_graph_def(
            original_graph.as_graph_def(),
            name="",
            input_map=input_map,
        )

    suffix_graph_def = tf1.graph_util.extract_sub_graph(
        remapped_graph.as_graph_def(),
        [tensor_name_to_op_name(name) for name in output_tensors],
    )
    return suffix_graph_def, placeholder_map


def write_graph_def(graph_def: tf1.GraphDef, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tf1.io.gfile.GFile(str(path), "wb") as handle:
        handle.write(graph_def.SerializeToString())
