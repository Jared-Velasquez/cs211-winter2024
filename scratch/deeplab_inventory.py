"""Inventory DeepLab V3 (MobileNet V2 backbone) frozen graph."""
import sys
from collections import Counter
import tensorflow as tf

sys.path.insert(0, "/Users/jaredvelasquez/projects/cs211-winter2024")
from auto_partition import enumerate_blue_cuts, is_blue_node, _TPU_BLUE_TF_OPS

tf1 = tf.compat.v1
PB = "/Users/jaredvelasquez/projects/cs211-winter2024/artifacts/deeplabv3_mnv2_pascal_trainval/frozen_inference_graph.pb"

gdef = tf1.GraphDef()
with tf1.io.gfile.GFile(PB, "rb") as f:
    gdef.ParseFromString(f.read())

g = tf.Graph()
with g.as_default():
    tf.graph_util.import_graph_def(gdef, name="")

ops = g.get_operations()
print(f"TOTAL OPS: {len(ops)}")

# Placeholders
print("\n=== PLACEHOLDERS ===")
for op in ops:
    if op.type == "Placeholder":
        try:
            shape = op.outputs[0].shape.as_list()
        except Exception:
            shape = None
        print(f"  {op.name}  dtype={op.outputs[0].dtype}  shape={shape}")

# Leaf ops
print("\n=== LEAF OPS ===")
consumers = {op.name: set() for op in ops}
for op in ops:
    for inp in op.inputs:
        consumers[inp.op.name].add(op.name)
for op in ops:
    if not consumers[op.name] and op.type not in ("NoOp", "Assert"):
        try:
            shape = op.outputs[0].shape.as_list() if op.outputs else None
        except Exception:
            shape = None
        print(f"  {op.type:25s} shape={shape} {op.name}")

type_counts = Counter(op.type for op in ops)
print("\n=== OP TYPE COUNTS ===")
for t, c in type_counts.most_common():
    blue = "BLUE" if t in _TPU_BLUE_TF_OPS else "red"
    print(f"  {c:5d}  {t:35s} [{blue}]")

# MobileNet block exits
print("\n=== MobilenetV2/expanded_conv_*/output (and base conv outputs) ===")
exits = set()
for op in ops:
    if "MobilenetV2/expanded_conv" in op.name and op.name.endswith("/output"):
        exits.add(op.name)
    if op.name == "MobilenetV2/Conv/Relu6" or op.name == "MobilenetV2/Conv_1/Relu6":
        exits.add(op.name)
for name in sorted(exits):
    op = g.get_operation_by_name(name)
    try:
        shape = op.outputs[0].shape.as_list()
    except Exception:
        shape = None
    blue = "B" if is_blue_node(op) else "R"
    print(f"  {blue} {op.type:25s} shape={shape} {name}")

# ASPP ops
print("\n=== ASPP/concat_projection/logits ops ===")
keywords = ["aspp", "image_pooling", "concat_projection", "logits"]
seen = set()
for op in ops:
    if any(k in op.name.lower() for k in keywords):
        prefix = op.name.split("/")[0]
        if prefix not in seen and len(seen) < 30:
            pass
        try:
            shape = op.outputs[0].shape.as_list() if op.outputs else None
        except Exception:
            shape = None
        blue = "B" if is_blue_node(op) else "R"
        # limit verbosity
        print(f"  {blue} {op.type:25s} shape={shape} {op.name}")
        if op.type in ("Conv2D", "BiasAdd", "Relu", "Relu6", "ConcatV2"):
            seen.add(op.name)

# ResizeBilinear
print("\n=== ResizeBilinear ops ===")
for op in ops:
    if op.type == "ResizeBilinear":
        try:
            shape = op.outputs[0].shape.as_list()
        except Exception:
            shape = None
        blue = "B" if is_blue_node(op) else "R"
        print(f"  {blue} {op.type:20s} shape={shape} {op.name}")

# ArgMax
print("\n=== ArgMax ops ===")
for op in ops:
    if op.type == "ArgMax":
        try:
            shape = op.outputs[0].shape.as_list()
        except Exception:
            shape = None
        blue = "B" if is_blue_node(op) else "R"
        print(f"  {blue} {op.type:20s} shape={shape} {op.name}")

# Red ops
red = [op for op in ops if not is_blue_node(op)]
red_types = Counter(op.type for op in red)
print(f"\n=== RED OP TYPES (n={len(red)}) ===")
for t, c in red_types.most_common():
    print(f"  {c:5d}  {t}")

# Enumerate cuts
print("\n=== enumerate_blue_cuts ===")
cuts = enumerate_blue_cuts(g)
print(f"Total cuts: {len(cuts)}")
top = sorted(cuts, key=lambda c: -c['num_tpu_ops'])[:10]
print("\nTop 10 by num_tpu_ops:")
for i, c in enumerate(top):
    print(f"  [{i}] tpu={c['num_tpu_ops']:4d} cpu={c['num_cpu_ops']:4d} "
          f"frontier_size={len(c['frontier_tensors']):3d} skip={c['has_skip_crossing']}")
    for ft in c['frontier_tensors'][:5]:
        print(f"        {ft}")
    if len(c['frontier_tensors']) > 5:
        print(f"        ... +{len(c['frontier_tensors'])-5} more")

sorted_cuts = sorted(cuts, key=lambda c: c['num_tpu_ops'])
n = len(sorted_cuts)
print("\nSpread sample (by num_tpu_ops):")
for label, idx in [("min", 0), ("p25", n // 4), ("p50", n // 2), ("p75", 3 * n // 4), ("max", n - 1)]:
    c = sorted_cuts[idx]
    print(f"  {label}: tpu={c['num_tpu_ops']:4d} cpu={c['num_cpu_ops']:4d} "
          f"frontier_size={len(c['frontier_tensors']):3d}")
    for ft in c['frontier_tensors'][:3]:
        print(f"        {ft}")
