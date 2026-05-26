from collections import deque
import tensorflow as tf


# TF GraphDef op types whose TFLite equivalent is in the Edge TPU
# supported-ops list (https://coral.ai/docs/edgetpu/models-intro/), OR
# which fold away during TFLite conversion and therefore do not block a
# cut: BiasAdd fuses into the preceding Conv2D; FusedBatchNorm* folds
# into Conv weights; Identity is stripped; Const is inlined; Placeholder
# is the graph input.
#
# This is a heuristic on the pre-conversion TF graph. Final blue/red
# classification happens after convert.py produces the .tflite.
_TPU_BLUE_TF_OPS = frozenset({
    # TF op                   -> TFLite op (Coral allowlist)
    "Conv2D",                 # CONV_2D
    "DepthwiseConv2dNative",  # DEPTHWISE_CONV_2D
    "Conv2DBackpropInput",    # TRANSPOSE_CONV
    "MatMul",                 # FULLY_CONNECTED
    "MaxPool",                # MAX_POOL_2D
    "AvgPool",                # AVERAGE_POOL_2D
    "Mean",                   # MEAN
    "Sum",                    # SUM
    "Max",                    # REDUCE_MAX
    "Min",                    # REDUCE_MIN
    "Add", "AddV2",           # ADD
    "Sub",                    # SUB
    "Mul",                    # MUL
    "Maximum",                # MAXIMUM
    "Minimum",                # MINIMUM
    "SquaredDifference",      # SQUARED_DIFFERENCE
    "Rsqrt",                  # RSQRT
    "Relu",                   # RELU
    "Relu6",                  # RELU6
    "Sigmoid",                # LOGISTIC
    "Tanh",                   # TANH
    "Softmax",                # SOFTMAX
    "ResizeBilinear",         # RESIZE_BILINEAR (some shapes rejected)
    "ResizeNearestNeighbor",  # RESIZE_NEAREST_NEIGHBOR
    "Concat", "ConcatV2",     # CONCATENATION
    "Pad",                    # PAD
    "Reshape",                # RESHAPE
    "Squeeze",                # SQUEEZE
    "ExpandDims",             # EXPAND_DIMS
    "StridedSlice",           # STRIDED_SLICE
    "Slice",                  # SLICE
    "Transpose",              # TRANSPOSE
    "Pack",                   # PACK
    "Split",                  # SPLIT
    "SpaceToDepth",           # SPACE_TO_DEPTH
    # Transparent under TFLite conversion
    "Identity",
    "Const",
    "Placeholder",
    "BiasAdd",
    "FusedBatchNorm",
    "FusedBatchNormV2",
    "FusedBatchNormV3",
})


def is_blue_node(node):
    """Return True if the TF GraphDef Operation is TPU-compatible (its
    TFLite equivalent is in the Coral allowlist) or transparent under
    TFLite conversion (folds into a neighboring op or is stripped)."""
    return node.type in _TPU_BLUE_TF_OPS


def find_all_blue_nodes(graph):
    return [op for op in graph.get_operations() if is_blue_node(op)]


def _build_cut_record(tpu_ops, all_op_names, op_by_name, consumers):
    """Given a TPU op-name set, compute frontier tensors, CPU ops, and
    has_skip_crossing."""
    frontier = []
    for op_name in tpu_ops:
        op = op_by_name[op_name]
        consumed_externally = any(
            c not in tpu_ops for c in consumers.get(op_name, ())
        )
        if consumed_externally:
            for out in op.outputs:
                frontier.append(out.name)
    frontier_set = set(frontier)

    # Skip-crossing: any edge from a TPU op to a CPU op whose tensor is
    # not in F. By construction, when we cut at op-output boundaries
    # this is always False, but we recompute for safety.
    has_skip_crossing = False
    for op_name in tpu_ops:
        for c in consumers.get(op_name, ()):
            if c in tpu_ops:
                continue
            consumer_op = op_by_name[c]
            for t in consumer_op.inputs:
                if t.op.name == op_name and t.name not in frontier_set:
                    has_skip_crossing = True
                    break
            if has_skip_crossing:
                break
        if has_skip_crossing:
            break

    cpu_ops = all_op_names - tpu_ops
    return {
        "frontier_tensors": sorted(frontier),
        "tpu_ops": tpu_ops,
        "cpu_ops": cpu_ops,
        "num_tpu_ops": len(tpu_ops),
        "num_cpu_ops": len(cpu_ops),
        "has_skip_crossing": has_skip_crossing,
    }


def enumerate_blue_cuts(graph, include_intra_blue_cuts=True):
    """Enumerate every candidate cut in `graph`.

    A candidate cut is a minimal tensor frontier F such that:
      (a) every op upstream of F (i.e. in the TPU partition) is blue, AND
      (b) F is closed under residual-skip detection — no edge crosses F
          without its tensor being in F.

    Returns a list of dicts, one per maximal frontier:
      {
        "frontier_tensors": [tensor_name, ...],
        "tpu_ops": set of op names upstream of F (the TPU partition),
        "cpu_ops": set of op names downstream of F (the CPU partition),
        "num_tpu_ops": int,
        "num_cpu_ops": int,
        "has_skip_crossing": bool,  # always False for a valid cut, but
                                    # tracked because some heuristic
                                    # enumerations may produce candidates
                                    # that fail this check.
      }
    """
    ops = list(graph.get_operations())
    op_by_name = {op.name: op for op in ops}
    all_op_names = set(op_by_name)

    # Build consumer index: op_name -> set of op_names that consume any of
    # its output tensors.
    consumers = {op.name: set() for op in ops}
    for op in ops:
        for tensor in op.inputs:
            producer = tensor.op.name
            if producer in consumers:
                consumers[producer].add(op.name)

    blue = {op.name for op in ops if is_blue_node(op)}

    cuts = []
    seen_tpu_sets = set()

    def emit(tpu_ops):
        if not tpu_ops:
            return
        key = frozenset(tpu_ops)
        if key in seen_tpu_sets:
            return
        seen_tpu_sets.add(key)
        cuts.append(_build_cut_record(
            tpu_ops, all_op_names, op_by_name, consumers))

    # Stage 1: maximal blue partitions bounded by red ops.
    # For each red op R, blue ops reachable upstream form a TPU
    # partition. Group by overlap (union-find) so connected red regions
    # share one partition.
    red_triggers = [op for op in ops if op.name not in blue]

    trigger_ancestors = {}
    for r in red_triggers:
        # walk upward through red op's inputs, but only collect blue ancestors
        anc = set()
        stack = [t.op for t in r.inputs]
        while stack:
            o = stack.pop()
            if o.name in anc or o.name not in blue:
                continue
            anc.add(o.name)
            for t in o.inputs:
                stack.append(t.op)
        if anc:
            trigger_ancestors[r.name] = anc

    parent = {name: name for name in trigger_ancestors}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    blue_to_triggers = {}
    for tname, anc in trigger_ancestors.items():
        for b in anc:
            blue_to_triggers.setdefault(b, []).append(tname)
    for b, tlist in blue_to_triggers.items():
        first = tlist[0]
        for other in tlist[1:]:
            union(first, other)

    groups = {}
    for tname in trigger_ancestors:
        root = find(tname)
        groups.setdefault(root, []).append(tname)

    max_tpu_sets = []
    for tnames in groups.values():
        tpu_ops = set()
        for tname in tnames:
            tpu_ops |= trigger_ancestors[tname]
        max_tpu_sets.append(tpu_ops)
        emit(tpu_ops)

    # Stage 2: intra-blue cuts. The maximal partition is one valid cut;
    # to surface shallower cuts inside the same contiguous blue region
    # (e.g. the existing BiasAdd baseline, which sits inside the
    # backbone+head blue region rather than at its red boundary), we:
    #
    #   (a) emit cut-at-B for every single blue op B in the maximal
    #       partition (TPU partition = upstream blue closure of B).
    #   (b) peel-back enumeration on the frontier antichain: start with
    #       the natural frontier of each maximal partition, then
    #       iteratively replace a frontier blue op with its blue inputs.
    #       Each step yields a strictly smaller TPU partition with a
    #       different frontier (e.g. peeling {Sigmoid, transpose_1, ...}
    #       back to {part_pred/BiasAdd, locref_pred/BiasAdd}).
    closure_cache = {}

    def blue_closure_of(name):
        if name in closure_cache:
            return closure_cache[name]
        seen = set()
        stack = [op_by_name[name]]
        while stack:
            o = stack.pop()
            if o.name in seen or o.name not in blue:
                continue
            seen.add(o.name)
            for t in o.inputs:
                stack.append(t.op)
        closure_cache[name] = seen
        return seen

    def blue_closure_of_set(names):
        out = set()
        for n in names:
            out |= blue_closure_of(n)
        return out

    if include_intra_blue_cuts and max_tpu_sets:
        biggest = max(max_tpu_sets, key=len)

        # (a) every single-op closure inside the biggest partition
        for b in biggest:
            emit(blue_closure_of(b))

        # (b) topological-layer cuts. Compute blue depth d(op) =
        # longest path through `biggest` from any input-less blue op.
        # For each depth D, the cut TPU = {blue ops at depth ≤ D}.
        # This produces a chain of cuts from input-only up to the
        # entire biggest partition, including all natural multi-op
        # frontiers along the way.
        depth = {}
        # Topologically order biggest by Kahn-like traversal
        # In-degree within `biggest`:
        in_deg = {n: 0 for n in biggest}
        for n in biggest:
            for t in op_by_name[n].inputs:
                if t.op.name in biggest:
                    in_deg[n] += 1
        ready = [n for n, d in in_deg.items() if d == 0]
        order = []
        in_deg_work = dict(in_deg)
        while ready:
            n = ready.pop()
            order.append(n)
            for c in consumers.get(n, ()):
                if c in in_deg_work:
                    in_deg_work[c] -= 1
                    if in_deg_work[c] == 0:
                        ready.append(c)
        for n in order:
            d = 0
            for t in op_by_name[n].inputs:
                if t.op.name in biggest:
                    d = max(d, depth[t.op.name] + 1)
            depth[n] = d

        if depth:
            max_d = max(depth.values())
            for D in range(max_d + 1):
                tpu_ops = {n for n in biggest if depth[n] <= D}
                emit(tpu_ops)

        # (b') For each topological depth D, emit closure of all blue
        # ops at depth D that have at least one non-blue descendant
        # (transitively). This produces antichain cuts whose frontier
        # is exactly the set of "co-deep" blue ops feeding the red
        # region — e.g. {part_pred/BiasAdd, locref_pred/BiasAdd}.
        # Compute has_red_desc(B) = exists non-blue descendant.
        has_red_desc = {}
        # process in reverse topological order
        rev_order = list(reversed(order))
        # First, mark ops whose direct consumers are non-blue (or whose
        # consumers transitively reach a non-blue op).
        for n in rev_order:
            red = False
            for c in consumers.get(n, ()):
                if c not in blue:
                    red = True
                    break
                if has_red_desc.get(c, False):
                    red = True
                    break
            has_red_desc[n] = red
        # Also any op outside `biggest` consumed by an op in `biggest`?
        # consumers list already covers all consumers, so this is fine.

        by_depth = {}
        for n in biggest:
            if has_red_desc.get(n, False):
                by_depth.setdefault(depth[n], []).append(n)
        for D, names in by_depth.items():
            # Closure of the WHOLE set at this depth that feeds red ops
            emit(blue_closure_of_set(names))
            # Also closure of each pair (if few enough) — captures
            # natural multi-output cuts like the BiasAdd pair.
            if 2 <= len(names) <= 8:
                from itertools import combinations
                for sub in combinations(names, 2):
                    emit(blue_closure_of_set(sub))

        # (c) peel-back BFS on the natural frontier. Bound the search.
        MAX_PEEL_STEPS = 5000
        natural_frontier = frozenset(
            op_name for op_name in biggest
            if any(c not in biggest for c in consumers.get(op_name, ()))
        )

        visited_frontiers = {natural_frontier}
        peel_queue = [natural_frontier]
        steps = 0
        while peel_queue and steps < MAX_PEEL_STEPS:
            current = peel_queue.pop(0)  # BFS
            for b in current:
                op = op_by_name[b]
                blue_inputs = [
                    t.op.name for t in op.inputs
                    if t.op.name in biggest
                ]
                if not blue_inputs:
                    continue
                new_frontier = frozenset(
                    (set(current) - {b}) | set(blue_inputs)
                )
                if new_frontier in visited_frontiers:
                    continue
                visited_frontiers.add(new_frontier)
                tpu_ops = blue_closure_of_set(new_frontier)
                emit(tpu_ops)
                peel_queue.append(new_frontier)
                steps += 1
                if steps >= MAX_PEEL_STEPS:
                    break

    # Sort cuts by TPU partition size (largest first) for readability
    cuts.sort(key=lambda c: c["num_tpu_ops"], reverse=True)
    return cuts



# to do: cache the result from call to is_blue_node for speed up
# to do: try to minimze the edges of the cut (the edges between the two subgraphs)
def find_max_blue_subgraph(graph):
    """
    Finds the partition of a directed acyclic graph (DAG) that maximizes the size of the subgraph containing only blue nodes.

    Args:
      graph: A dictionary representing the graph. Keys are nodes, and values are sets of neighbor nodes.

    Returns:
      A tuple containing two sets: the first set represents the nodes in the subgraph with only blue nodes, and the second set represents the remaining nodes.
    """
    # Find blue nodes with no incoming edges (potential roots for BFS)
    blue_roots = find_all_blue_nodes(graph)

    max_size = 0
    optimal_partition = None

    for last_node in blue_roots:
        # first node of the graph since we know for sure that the first node is blue
        # initialize necessary variables for BFS
        visited = set()
        queue = deque([last_node])

        # conduct a modified BFS starting from the last_node
        while queue:
            node = queue.popleft()
            visited.add(node.name)

            # checking if all the neighbours are blue for the current node
            for neighbor in node.inputs:
                if neighbor.name not in visited and is_blue_node(node) == True:
                    queue.append(graph.get_operation_by_name(neighbor.name.split(":")[0]))

        all_operation_names = set()
        for operation in graph.get_operations():
            all_operation_names.add(operation.name)

        # Update optimal partition if current size is larger
        if len(visited) > max_size:
            max_size = len(visited)
            optimal_partition = (visited, all_operation_names - visited)

    return optimal_partition


def traverse_graph(op):
  """
  Recursive function to traverse the graph node by node.

  Args:
    op: A TensorFlow Operation object.
  """
  # Print information about the current node
  """print(f"Node Name: {op.name}")
  print(f"Node Type: {op.type}")
  print(f"Node inputs: {op.inputs}")"""

  # Iterate through the input tensors of the current operation
  for input_tensor in op.inputs:
    # Get the operation that produces this input tensor
    producer_op = input_tensor.op

    # Recursively call the function on the producer operation
    traverse_graph(producer_op)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="snapshot-1000.pb")
    parser.add_argument("--baseline-tensors",
        default="pose/part_pred/block4/BiasAdd:0,pose/locref_pred/block4/BiasAdd:0",
        help="Comma-separated tensor names expected to appear as a "
             "frontier in the enumeration output.")
    args = parser.parse_args()

    tfv1 = tf.compat.v1
    gdef = tfv1.GraphDef()
    with tfv1.io.gfile.GFile(args.graph, "rb") as f:
        gdef.ParseFromString(f.read())

    g = tf.Graph()
    with g.as_default():
        tf.graph_util.import_graph_def(gdef, name='')

    cuts = enumerate_blue_cuts(g)
    print(f"enumerate_blue_cuts: {len(cuts)} cuts found in {args.graph}")
    if cuts:
        sizes = [c["num_tpu_ops"] for c in cuts]
        print(f"  smallest TPU region: {min(sizes)} ops")
        print(f"  largest  TPU region: {max(sizes)} ops")

    # Look for cuts that contain the baseline tensors in their
    # frontier. The "natural" cut at BiasAdd has additional frontier
    # tensors because the ResNet block4 dilated-conv pattern produces
    # several SpaceToBatchND/BatchToSpaceND non-blue ops that force
    # extra Relu tensors to also appear in the frontier — so we check
    # subset rather than equality.
    baseline = set(t.strip() for t in args.baseline_tensors.split(",") if t.strip())
    exact = [c for c in cuts if set(c["frontier_tensors"]) == baseline]
    subset = [c for c in cuts if baseline.issubset(set(c["frontier_tensors"]))]
    if exact:
        c = exact[0]
        print(f"  baseline frontier {sorted(baseline)} -> "
              f"EXACT match (tpu={c['num_tpu_ops']} cpu={c['num_cpu_ops']})")
    elif subset:
        c = min(subset, key=lambda c: len(c["frontier_tensors"]))
        print(f"  baseline frontier {sorted(baseline)} -> "
              f"appears as a subset of cut with "
              f"tpu={c['num_tpu_ops']} cpu={c['num_cpu_ops']}, "
              f"full frontier size {len(c['frontier_tensors'])}")
    print(f"  baseline match: {bool(exact or subset)}")

    #print(g.get_operation_by_name(g.get_operations()[len(g.get_operations())-1].name).inputs)
    #print(g.get_operation_by_name(g.get_operations()[len(g.get_operations())-1].name))



if __name__ == '__main__':
    main()