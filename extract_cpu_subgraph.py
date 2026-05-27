"""Extract the CPU-side subgraph of a partition into a SavedModel.

The CPU subgraph takes the partition boundary tensors as Placeholder
inputs (float32) and produces the model's final output tensor. Mirrors
gen_tflite.py on the TPU side; together they fully decompose the model
across the Coral TPU + host-CPU pipeline.
"""
import argparse
import os
import shutil

import tensorflow as tf

tf1 = tf.compat.v1


def parse_tensor_list(s):
    return [t.strip() for t in s.split(",") if t.strip()]


def op_name(tensor_name):
    return tensor_name.split(":")[0]


def fully_qualified(tensor_name):
    return tensor_name if ":" in tensor_name else tensor_name + ":0"


def extract(model_path, boundary_tensors, final_output_tensor,
            output_dir):
    # Accept either a single tensor name or a list. Single name kept as
    # the signature for backwards compatibility; comma-separated string
    # or list both work.
    if isinstance(final_output_tensor, str):
        final_outputs = parse_tensor_list(final_output_tensor)
    else:
        final_outputs = list(final_output_tensor)
    gdef = tf1.GraphDef()
    with tf1.io.gfile.GFile(model_path, "rb") as f:
        gdef.ParseFromString(f.read())

    g = tf.Graph()
    with g.as_default():
        tf.graph_util.import_graph_def(gdef, name="")

    # Look up boundary tensor shapes and dtypes so the replacement
    # Placeholders match what the downstream ops expect.
    boundary_info = []
    for t_name in boundary_tensors:
        t = g.get_tensor_by_name(fully_qualified(t_name))
        boundary_info.append({
            "tensor": t_name,
            "shape": t.shape.as_list(),
            "dtype": t.dtype,
        })

    # Build new graph with Placeholders standing in for each boundary
    # tensor, then re-import the original graph mapping each boundary
    # tensor to its Placeholder.
    new_graph = tf.Graph()
    input_map = {}
    new_inputs_by_orig = {}
    with new_graph.as_default():
        for info in boundary_info:
            ph_name = "cpu_in/" + op_name(info["tensor"]).replace("/", "_")
            ph = tf1.placeholder(
                dtype=info["dtype"], shape=info["shape"], name=ph_name)
            input_map[fully_qualified(info["tensor"])] = ph
            new_inputs_by_orig[info["tensor"]] = ph
        tf1.import_graph_def(g.as_graph_def(), name="", input_map=input_map)

    # Prune everything not on a path to the final outputs.
    gdef_sub = tf1.graph_util.extract_sub_graph(
        new_graph.as_graph_def(),
        [op_name(t) for t in final_outputs])

    g2 = tf.Graph()
    with g2.as_default():
        tf.graph_util.import_graph_def(gdef_sub, name="")

    # Re-fetch the Placeholders and the final output from g2.
    inputs = {}
    for info in boundary_info:
        ph_name = "cpu_in/" + op_name(info["tensor"]).replace("/", "_")
        # input key matches the original boundary tensor name without :0
        key = op_name(info["tensor"]).replace("/", "_")
        inputs[key] = g2.get_tensor_by_name(ph_name + ":0")

    outputs = {}
    for t in final_outputs:
        key = op_name(t).replace("/", "_") if len(final_outputs) > 1 else "output"
        outputs[key] = g2.get_tensor_by_name(fully_qualified(t))

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)

    with tf1.Session(graph=g2) as s2:
        tf1.saved_model.simple_save(
            session=s2,
            export_dir=output_dir,
            inputs=inputs,
            outputs=outputs)

    return list(inputs.keys())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True,
                   help="Path to frozen .pb")
    p.add_argument("--boundary-tensors", required=True,
                   help="Comma-separated boundary tensor names; these "
                        "become Placeholder inputs of the CPU subgraph "
                        "(e.g. pose/part_pred/block4/BiasAdd,"
                        "pose/locref_pred/block4/BiasAdd)")
    p.add_argument("--final-output-tensor", required=True,
                   help="Final output tensor(s) of the model. Single name "
                        "or comma-separated list (e.g. concat_1, or "
                        "detection_boxes,detection_scores,detection_classes,"
                        "num_detections for SSD).")
    p.add_argument("--output-dir", required=True,
                   help="Where to write the SavedModel")
    args = p.parse_args()

    keys = extract(
        model_path=args.model,
        boundary_tensors=parse_tensor_list(args.boundary_tensors),
        final_output_tensor=args.final_output_tensor,
        output_dir=args.output_dir,
    )
    print(f"Wrote CPU SavedModel to {args.output_dir} with inputs {keys}")


if __name__ == "__main__":
    main()
