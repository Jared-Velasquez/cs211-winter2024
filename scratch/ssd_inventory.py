"""Inventory SSD MobileNet V2 frozen graph."""
import sys
from collections import Counter
import tensorflow as tf

sys.path.insert(0, "/Users/jaredvelasquez/projects/cs211-winter2024")
from auto_partition import enumerate_blue_cuts, is_blue_node, _TPU_BLUE_TF_OPS

tf1 = tf.compat.v1
PB = "/Users/jaredvelasquez/projects/cs211-winter2024/artifacts/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb"

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

# Leaf ops (no consumers)
print("\n=== LEAF OPS (no downstream consumers) ===")
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
print("\n=== TOP OP TYPES (with blue/red status) ===")
for t, c in type_counts.most_common(40):
    blue = "BLUE" if t in _TPU_BLUE_TF_OPS else "red"
    print(f"  {c:5d}  {t:35s} [{blue}]")

# Architectural boundaries: expanded_conv block outputs
print("\n=== expanded_conv block outputs ===")
for op in ops:
    if "expanded_conv" in op.name and op.name.endswith("/output"):
        try:
            shape = op.outputs[0].shape.as_list()
        except Exception:
            shape = None
        blue = "B" if is_blue_node(op) else "R"
        print(f"  {blue} {op.type:25s} shape={shape} {op.name}")

# Predictor heads
print("\n=== BoxPredictor heads (last BiasAdd of each) ===")
for op in ops:
    if op.name.startswith("BoxPredictor_") and op.type == "BiasAdd":
        try:
            shape = op.outputs[0].shape.as_list()
        except Exception:
            shape = None
        blue = "B" if is_blue_node(op) else "R"
        print(f"  {blue} {op.type:15s} shape={shape} {op.name}")

# Post-backbone Conv_1
print("\n=== Post-backbone Conv_1/Relu6 etc ===")
for op in ops:
    if op.name.startswith("FeatureExtractor/MobilenetV2/Conv_1/"):
        try:
            shape = op.outputs[0].shape.as_list()
        except Exception:
            shape = None
        blue = "B" if is_blue_node(op) else "R"
        print(f"  {blue} {op.type:25s} shape={shape} {op.name}")

# Postprocessor boundary
print("\n=== First few Postprocessor ops (boundary) ===")
post_ops = [op for op in ops if op.name.startswith("Postprocessor/")]
print(f"Total Postprocessor ops: {len(post_ops)}")
for op in post_ops[:15]:
    blue = "B" if is_blue_node(op) else "R"
    print(f"  {blue} {op.type:25s} {op.name}")

# Concat/Squeeze ops feeding Postprocessor (likely the boundary tensors)
print("\n=== Tensors feeding Postprocessor (boundary candidates) ===")
post_op_names = {op.name for op in post_ops}
feeders = set()
for op in post_ops:
    for inp in op.inputs:
        if inp.op.name not in post_op_names:
            feeders.add((inp.op.name, inp.op.type, str(inp.shape.as_list()) if inp.shape.rank else "?"))
for name, typ, shape in sorted(feeders):
    print(f"  {typ:25s} shape={shape} {name}")

# Red ops globally (anything non-blue)
red = [op for op in ops if not is_blue_node(op)]
red_types = Counter(op.type for op in red)
print(f"\n=== RED OP TYPES (n={len(red)}) ===")
for t, c in red_types.most_common():
    print(f"  {c:5d}  {t}")

# Enumerate cuts
print("\n=== enumerate_blue_cuts ===")
cuts = enumerate_blue_cuts(g)
print(f"Total cuts: {len(cuts)}")

# Top 15 cuts by num_tpu_ops
top = sorted(cuts, key=lambda c: -c['num_tpu_ops'])[:15]
print("\nTop 15 by num_tpu_ops:")
for i, c in enumerate(top):
    print(f"  [{i}] tpu={c['num_tpu_ops']:4d} cpu={c['num_cpu_ops']:4d} "
          f"frontier_size={len(c['frontier_tensors']):3d} skip={c['has_skip_crossing']}")
    for ft in c['frontier_tensors'][:5]:
        print(f"        {ft}")
    if len(c['frontier_tensors']) > 5:
        print(f"        ... +{len(c['frontier_tensors'])-5} more")

# Spread sample
print("\nSpread sample (by num_tpu_ops): smallest, 25%, median, 75%, largest")
sorted_cuts = sorted(cuts, key=lambda c: c['num_tpu_ops'])
n = len(sorted_cuts)
for label, idx in [("min", 0), ("p25", n // 4), ("p50", n // 2), ("p75", 3 * n // 4), ("max", n - 1)]:
    c = sorted_cuts[idx]
    print(f"  {label}: tpu={c['num_tpu_ops']:4d} cpu={c['num_cpu_ops']:4d} "
          f"frontier_size={len(c['frontier_tensors']):3d}")
