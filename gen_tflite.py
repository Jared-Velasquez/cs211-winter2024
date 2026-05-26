"""Extract a subgraph from a frozen TF .pb into a SavedModel.

Pre-conversion stage of the partition pipeline:

    gen_tflite.py -> convert.py -> edgetpu_compiler -> updated_edgetpu_test.py

Generalised to accept any partition boundary tensors via CLI args, so the
same script handles DLC, SSD MobileNet V2 and DeepLab V3.
"""
import argparse
import os

import tensorflow as tf

tf1 = tf.compat.v1


def parse_shape(s):
    return [int(x) for x in s.split(",") if x.strip()]


def parse_tensor_list(s):
    """Comma-separated tensor names. Strip the trailing :N if present —
    extract_sub_graph wants op names, not tensor names."""
    return [t.strip() for t in s.split(",") if t.strip()]


def op_name(tensor_name):
    return tensor_name.split(":")[0]


def extract(model_path, input_tensor, input_shape, output_tensors,
            output_dir):
    gdef = tf1.GraphDef()
    with tf1.io.gfile.GFile(model_path, "rb") as f:
        gdef.ParseFromString(f.read())

    g = tf.Graph()
    with g.as_default():
        tf.graph_util.import_graph_def(gdef, name="")

    # Match the dtype of the existing input placeholder (e.g. uint8 for
    # SSD/DeepLab object detection graphs, float32 for DLC).
    orig_input = g.get_tensor_by_name(input_tensor + ":0")
    input_dtype = orig_input.dtype

    # Replace the original input placeholder with one of the requested
    # fixed shape, via the input_map trick.
    new_graph = tf.Graph()
    with new_graph.as_default():
        new_input = tf1.placeholder(
            dtype=input_dtype, shape=input_shape, name=input_tensor)
        tf1.import_graph_def(
            g.as_graph_def(), name="",
            input_map={input_tensor: new_input})

    # Backward reachability from the requested output ops.
    output_op_names = [op_name(t) for t in output_tensors]
    gdef_sub = tf1.graph_util.extract_sub_graph(
        new_graph.as_graph_def(), output_op_names)

    g2 = tf.Graph()
    with g2.as_default():
        tf.graph_util.import_graph_def(gdef_sub, name="")

    g2_input = g2.get_tensor_by_name(input_tensor + ":0")
    # Map output tensor names back into g2. If the caller passed bare
    # op names, append :0; otherwise use the tensor name as-is.
    outputs = {}
    for t in output_tensors:
        if ":" not in t:
            t_full = t + ":0"
        else:
            t_full = t
        outputs[t.replace("/", "_").replace(":", "_")] = (
            g2.get_tensor_by_name(t_full))

    os.makedirs(os.path.dirname(output_dir.rstrip("/")) or ".",
                exist_ok=True)
    # simple_save refuses to overwrite an existing dir.
    if os.path.isdir(output_dir):
        import shutil
        shutil.rmtree(output_dir)

    with tf1.Session(graph=g2) as s2:
        tf1.saved_model.simple_save(
            session=s2,
            export_dir=output_dir,
            inputs={"input": g2_input},
            outputs=outputs)

    return list(outputs.keys())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True,
                   help="Path to frozen .pb")
    p.add_argument("--input-tensor", default="Placeholder",
                   help="Name of input placeholder in the frozen graph")
    p.add_argument("--input-shape", default="1,320,320,3",
                   help="Comma-separated shape for the new input "
                        "placeholder (e.g. 1,320,320,3)")
    p.add_argument("--output-tensors", required=True,
                   help="Comma-separated boundary tensor names to use "
                        "as outputs of the extracted subgraph "
                        "(e.g. pose/part_pred/block4/BiasAdd,"
                        "pose/locref_pred/block4/BiasAdd)")
    p.add_argument("--output-dir", required=True,
                   help="Where to write the SavedModel")
    args = p.parse_args()

    keys = extract(
        model_path=args.model,
        input_tensor=args.input_tensor,
        input_shape=parse_shape(args.input_shape),
        output_tensors=parse_tensor_list(args.output_tensors),
        output_dir=args.output_dir,
    )
    print(f"Wrote SavedModel to {args.output_dir} with outputs {keys}")


if __name__ == "__main__":
    main()
