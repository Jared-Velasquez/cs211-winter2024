"""Inventory DLC frozen graph: count ops, list block/pose ops, top blue cuts."""
import sys
from collections import Counter
import tensorflow as tf

sys.path.insert(0, "/Users/jaredvelasquez/projects/cs211-winter2024")
from auto_partition import enumerate_blue_cuts, is_blue_node, _TPU_BLUE_TF_OPS

tf1 = tf.compat.v1
PB = "/Users/jaredvelasquez/projects/cs211-winter2024/snapshot-1000.pb"

gdef = tf1.GraphDef()
with tf1.io.gfile.GFile(PB, "rb") as f:
    gdef.ParseFromString(f.read())

g = tf.Graph()
with g.as_default():
    tf.graph_util.import_graph_def(gdef, name="")

ops = g.get_operations()
print(f"TOTAL OPS: {len(ops)}")

type_counts = Counter(op.type for op in ops)
print("\n=== OP TYPE COUNTS ===")
for t, c in type_counts.most_common():
    blue = "BLUE" if t in _TPU_BLUE_TF_OPS else "red"
    print(f"  {c:5d}  {t:35s} [{blue}]")

print("\n=== BLOCK ops by block (only Relu/exit-relevant) ===")
for b in (1, 2, 3, 4):
    block_ops = [op for op in ops if op.name.startswith(f"resnet_v1_50/block{b}/")]
    print(f"\n--- block{b}: {len(block_ops)} ops total ---")
    # Show only Relu ops (block exits) plus surrounding context
    for op in block_ops:
        if op.type == "Relu":
            blue_flag = "B" if is_blue_node(op) else "R"
            try:
                shape = op.outputs[0].shape.as_list()
            except Exception:
                shape = None
            print(f"  {blue_flag} {op.type:8s} shape={shape} {op.name}")

print("\n=== POSE OPS (pose/*) ===")
pose_ops = [op for op in ops if op.name.startswith("pose/")]
print(f"Total pose ops: {len(pose_ops)}")
for op in pose_ops:
    blue_flag = "B" if is_blue_node(op) else "R"
    try:
        shape = op.outputs[0].shape.as_list() if op.outputs else None
    except Exception:
        shape = None
    print(f"  {blue_flag} {op.type:25s} shape={shape} {op.name}")

print("\n=== NON-BLUE OPS GLOBALLY (red ops) ===")
red_ops = [op for op in ops if not is_blue_node(op)]
print(f"Red op count: {len(red_ops)}")
for op in red_ops:
    print(f"  {op.type:25s} {op.name}")

print("\n=== TOP 20 BLUE CUTS (sorted by num_tpu_ops desc) ===")
cuts = enumerate_blue_cuts(g)
print(f"Total cuts enumerated: {len(cuts)}")
for i, c in enumerate(cuts[:20]):
    print(f"\n  [{i}] tpu={c['num_tpu_ops']} cpu={c['num_cpu_ops']} "
          f"frontier_size={len(c['frontier_tensors'])} skip={c['has_skip_crossing']}")
    for ft in c['frontier_tensors'][:8]:
        print(f"        {ft}")
    if len(c['frontier_tensors']) > 8:
        print(f"        ... +{len(c['frontier_tensors'])-8} more")

print("\n-- Smallest 10 by num_tpu_ops --")
small = sorted(cuts, key=lambda c: c['num_tpu_ops'])[:10]
for c in small:
    print(f"  tpu={c['num_tpu_ops']} cpu={c['num_cpu_ops']} "
          f"frontier_size={len(c['frontier_tensors'])}")
    for ft in c['frontier_tensors'][:4]:
        print(f"        {ft}")
