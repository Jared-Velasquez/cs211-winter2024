# CS211 Project: Accuracy-Aware Model Partitioning for Coral Edge TPU

This repository keeps the original script-first layout from the previous team, but removes the most brittle hardcoded paths so we can build a cleaner Student A baseline around it.

The current baseline now covers:

- **Task A**: DeepLabCut animal pose estimation
- **Task B**: SSD MobileNet V2 object detection on COCO val2017
- **Task C**: DeepLab V3 semantic segmentation on Pascal VOC 2012 val

The split / TPU-prep path is only verified end to end for Task A right now. Tasks B and C are currently set up through the float32 baseline and metric-evaluation stage.

The model download/export step stays **outside** this repo. For DLC, that workflow lives in the sibling `model_export/` directory.

## Current Layout

### Top-level scripts

- [`tensorflow_run.py`](./tensorflow_run.py): float32 full-graph baseline runner
- [`run_baseline.py`](./run_baseline.py): float32 baseline runner; saves prediction baselines for all tasks and accuracy metrics where enabled
- [`gen_tflite.py`](./gen_tflite.py): exports the TPU prefix as a SavedModel for an arbitrary boundary
- [`convert.py`](./convert.py): converts the SavedModel prefix to TFLite
- [`split.py`](./split.py): exports split artifacts and metadata for a chosen boundary
- [`updated_edgetpu_test.py`](./updated_edgetpu_test.py): reusable `HybridRunner` validation path; currently used in CPU-only partitioned mode
- [`auto_partition.py`](./auto_partition.py): generic graph candidate enumerator plus TPU-compatibility/BFS scaffold for Student B to extend
- [`import_pb.py`](./import_pb.py): graph visualization helper for TensorBoard

### Minimal shared helpers

These scripts share a small helper layer under [`src/`](./src):

- `config_utils.py`: load concise JSON task configs
- `graph_utils.py`: load/import frozen graphs, extract prefix/suffix graph defs
- `data_loaders.py`: AP-10K pose images, DLC video frames, COCO val images, and Pascal VOC val image+label loading
- `evaluation.py`: labeled accuracy metrics for tasks that enable them in config
- `io_utils.py`: save `.npz` outputs and JSON summaries

### Configs

- [`configs/task_a_dlc.json`](./configs/task_a_dlc.json): current working Task A config
- [`configs/task_b_detection.json`](./configs/task_b_detection.json): current working Task B config
- [`configs/task_c_segmentation.json`](./configs/task_c_segmentation.json): current working Task C config

## How The Config Drives The Pipeline

Each top-level script reads a single task config and uses it as the source of truth for:

- which frozen graph to load
- which dataset or video input loader to use
- which input and output tensors to run
- which boundary tensors define the split
- where generated artifacts should be written

The most important config fields are:

- `task_name`: human-readable task identifier used in summaries
- `task_type`: `pose_estimation`, `object_detection`, or `semantic_segmentation`
- `data_loader`: which loader in `src/data_loaders.py` to use
- `model_path`: the full float32 frozen graph `.pb`
- `images_dir` or `video_path`: the task input source
- `annotations_path` / `annotations_dir`: label source when a task uses labeled evaluation
- `input_tensor`: graph input tensor name
- `output_tensors`: final output tensor names for full-graph inference
- `boundary_tensors`: the manual split point used by the partitioning scripts
- `artifacts_dir`: directory where outputs for that config are saved
- `compute_accuracy`: whether `run_baseline.py` should compute labeled metrics

In other words:

```text
task config
-> tells loaders where the inputs live
-> tells TensorFlow which graph/tensors to run
-> tells split scripts where to cut the graph
-> tells every script where to save its outputs
```

Right now, the split behavior is still **manual and config-defined**. `auto_partition.py` only enumerates candidate graph points; it does not automatically choose and rewrite `boundary_tensors`.

## Data And Model Layout

The repo now expects task-local artifacts under:

```text
data/
  task_a/
    models/
      snapshot-700000.pb
    data/
      ap-10k/
        annotations/
        data/
  task_b/
    models/
      frozen_inference_graph.pb
      ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite
    data/
      val2017/
      annotations/
  task_c/
    models/
      frozen_inference_graph.pb
      deeplabv3_mnv2_pascal_quant_edgetpu.tflite
    data/
      pascal-voc-2012-DatasetNinja/
        val/
          img/
          ann/
artifacts/
  task_a/
    dlc/
  task_b/
    detection/
  task_c/
    segmentation/
```

For Task A, the current working config points at:

- model: `data/task_a/models/snapshot-700000.pb`
- AP-10K images: `data/task_a/data/ap-10k/data`
- AP-10K annotations: `data/task_a/data/ap-10k/annotations/ap10k-val-split1.json`

These inputs are intentionally treated as local runtime assets, not as tracked source files.

For Task C, only the `val/` split is kept locally in the current repo layout because that is the subset Student A needs for baseline and evaluation work.

## Environment

For the baseline repo itself, you only need TensorFlow + NumPy + OpenCV. A minimal dependency list is in [`requirements.txt`](./requirements.txt), and the repo-local conda environment spec is in [`environment.yml`](./environment.yml).

Create the repo-local environment with:

```bash
cd cs211-winter2024
./setup_env.sh
./run_in_env.sh python <script>.py ...
```

This keeps DeepLabCut and model export concerns outside this repo while still giving the baseline scripts a known-good TensorFlow environment.

## Pipeline At A Glance

The current repo supports two related but distinct flows:

1. **Full float32 baseline flow**
   Run the original frozen graph on CPU and save the reference predictions.

2. **Partition / TPU-prep flow**
   Cut the graph at a config-defined boundary, export the prefix, convert it to TFLite, and validate the split in CPU-only mode.

The second flow is the scaffold that later Edge TPU execution will build on.

## Script Inputs And Outputs

This table is the quickest way to understand how the files interact with the config and with one another:

| Script | Reads From Config | Main Purpose | Main Outputs |
|---|---|---|---|
| `tensorflow_run.py` | `model_path`, loader fields, `input_tensor`, `output_tensors`, `artifacts_dir` | Run the full frozen graph on CPU | `full_graph_outputs.npz`, `full_graph_summary.json` |
| `run_baseline.py` | same as `tensorflow_run.py`, plus `compute_accuracy` | Run full CPU baseline and optionally compute task metrics | `full_graph_outputs.npz`, `full_graph_summary.json`, `baseline_results.json` |
| `auto_partition.py` | `model_path` | Enumerate candidate graph points and TPU-compatible traversal scaffolding | `partition_candidates.json` |
| `gen_tflite.py` | `model_path`, `input_tensor`, `boundary_tensors`, `artifacts_dir` | Extract the prefix subgraph at the chosen boundary | `prefix_saved_model/`, `prefix_saved_model_metadata.json` |
| `convert.py` | prefix SavedModel path, task config, optional representative data settings | Convert the exported prefix into TFLite | `output.tflite` |
| `split.py` | `model_path`, `input_tensor`, `boundary_tensors`, `output_tensors`, `artifacts_dir` | Export the split artifacts in one pass | `prefix_saved_model/`, `suffix_graph.pb`, `split_metadata.json` |
| `updated_edgetpu_test.py` | `model_path`, loader fields, `boundary_tensors`, `output_tensors`, `artifacts_dir` | Validate the split path, currently in CPU-only mode | `partitioned_cpu_outputs.npz`, `boundary_outputs.npz`, `partitioned_cpu_summary.json` |
| `import_pb.py` | `model_path` | Visualize the graph in TensorBoard | TensorBoard log directory |

### Important artifact details

- The **base model** is the original frozen TensorFlow graph:
  - format: `.pb`
- The **prefix export** is currently saved as a **SavedModel**, not as a raw prefix `.pb`:
  - directory: `prefix_saved_model/`
  - this is what `convert.py` consumes
- The **suffix export** is currently saved as:
  - `suffix_graph.pb`
- The **TFLite prefix artifact** is:
  - `output.tflite`

So the current split artifact chain looks like:

```text
full frozen graph (.pb)
-> prefix SavedModel
-> prefix TFLite (.tflite)

full frozen graph (.pb)
-> suffix graph (.pb)
```

## Task A Split Workflow

### 1. Float32 baseline

Run the full frozen graph entirely on CPU:

```bash
cd cs211-winter2024
./run_in_env.sh python tensorflow_run.py --config configs/task_a_dlc.json --frame-limit 2
```

This writes:

- `artifacts/task_a/dlc/full_graph_outputs.npz`
- `artifacts/task_a/dlc/full_graph_summary.json`

Important: this is the canonical Student A baseline. It is **not** represented as a fake `0-100` partition. It is simply the full float32 graph run on CPU.

### 2. Enumerate candidate partition points

```bash
./run_in_env.sh python auto_partition.py --config configs/task_a_dlc.json
```

This writes:

- `artifacts/task_a/dlc/partition_candidates.json`

This first pass is intentionally generic. It does not pick a “best” split and it no longer depends on DLC-specific regex rules in the config. Instead it:

- lists candidate tensors from graph ops
- reports a simple “potentially TPU-compatible” op count
- preserves a BFS-style backward-closure helper as starting scaffolding for Student B

### 3. Export the prefix subgraph

```bash
./run_in_env.sh python gen_tflite.py --config configs/task_a_dlc.json --force
```

This writes:

- `artifacts/task_a/dlc/prefix_saved_model/`
- `artifacts/task_a/dlc/prefix_saved_model_metadata.json`

This step uses `boundary_tensors` from the config. Those tensors are the manual definition of the split point.

### 4. Convert the prefix to TFLite

```bash
./run_in_env.sh python convert.py \
  --config configs/task_a_dlc.json \
  --model artifacts/task_a/dlc/prefix_saved_model \
  --output artifacts/task_a/dlc/output.tflite
```

This gives the TFLite prefix artifact that later TPU compilation will consume.

At this stage the model is **TFLite**, but not yet **Edge TPU compiled**.

### 5. Export generalized split metadata

```bash
./run_in_env.sh python split.py --config configs/task_a_dlc.json --force
```

This writes:

- `artifacts/task_a/dlc/prefix_saved_model/`
- `artifacts/task_a/dlc/suffix_graph.pb`
- `artifacts/task_a/dlc/split_metadata.json`

`split_metadata.json` is the main record of what split was used and which artifacts belong to it.

### 6. Validate the split in CPU-only mode

```bash
./run_in_env.sh python updated_edgetpu_test.py --config configs/task_a_dlc.json --frame-limit 2
```

This writes:

- `artifacts/task_a/dlc/partitioned_cpu_outputs.npz`
- `artifacts/task_a/dlc/boundary_outputs.npz`
- `artifacts/task_a/dlc/partitioned_cpu_summary.json`

The current working DLC split reproduces the full float32 outputs exactly in CPU-only validation.

This script is the current stand-in for the future TPU path:

- it runs the prefix on CPU
- captures the boundary tensors
- feeds those tensors into the suffix on CPU
- compares the result against the full baseline

It verifies the **split logic**, but it does **not** currently run a compiled Edge TPU model.

## Float32 Baselines

Use [`run_baseline.py`](./run_baseline.py) when you want a clean float32 baseline record saved. It always writes raw predictions and inference summaries; it only computes labeled accuracy metrics when the task config enables `compute_accuracy`.

Examples:

```bash
./run_in_env.sh python run_baseline.py --config configs/task_a_dlc.json --frame-limit 100
./run_in_env.sh python run_baseline.py --config configs/task_b_detection.json --frame-limit 100
./run_in_env.sh python run_baseline.py --config configs/task_c_segmentation.json --frame-limit 100
```

This writes, per task:

- `full_graph_outputs.npz`
- `full_graph_summary.json`
- `baseline_results.json`

### Task A notes

- Task A now uses AP-10K validation images as an unlabeled input set for the fidelity baseline.
- The current DLC model predicts 39 landmarks, while AP-10K labels 17 landmarks.
- Because Student A is using Task A for relative-degradation experiments, the official baseline skips labeled accuracy and simply stores float32 predictions for later split-vs-baseline comparisons.

### Task B / Task C notes

Run the float32 baselines with:

```bash
./run_in_env.sh python tensorflow_run.py --config configs/task_b_detection.json --frame-limit 2
./run_in_env.sh python tensorflow_run.py --config configs/task_c_segmentation.json --frame-limit 2
```

These write:

- `artifacts/task_b/detection/full_graph_outputs.npz`
- `artifacts/task_b/detection/full_graph_summary.json`
- `artifacts/task_c/segmentation/full_graph_outputs.npz`
- `artifacts/task_c/segmentation/full_graph_summary.json`
- `artifacts/task_b/detection/baseline_results.json`
- `artifacts/task_c/segmentation/baseline_results.json`

Notes:

- Tasks B and C keep both kinds of baseline artifacts:
  - raw float32 predictions for future drift comparisons
  - labeled accuracy metrics for the original float32 models
- Task B outputs stack cleanly into dense arrays such as detection boxes, scores, classes, and `num_detections`.
- Task C outputs keep native image sizes, so the saved `SemanticPredictions` array is stored as an object array when sample shapes differ.

## Edge TPU Sanity Checks

The downloaded precompiled `.tflite` files for Tasks B and C are real Edge TPU binaries. Without TPU hardware they cannot be executed in the standard CPU TFLite interpreter; local allocation fails with `edgetpu-custom-op`, which is expected.

That means the current no-hardware sanity check is:

- verify the files exist in the expected paths
- verify TensorFlow can run the frozen `.pb` baselines
- verify the Edge TPU `.tflite` files are recognized as Edge TPU builds by their unresolved `edgetpu-custom-op`

## What Is Still Missing For Real Edge TPU Support

The current repo stops just before the actual Coral runtime step.

What is already present:

- config-defined manual split points
- prefix extraction as a SavedModel
- TFLite conversion for the prefix
- suffix graph export
- CPU-only split validation through `updated_edgetpu_test.py`

What still needs to be added for actual TPU execution:

1. **Edge TPU compilation step**
   After `convert.py`, the prefix `.tflite` still needs to be compiled with `edgetpu_compiler`.

2. **Compiled-model runtime loading**
   `updated_edgetpu_test.py` needs a real Edge TPU execution path that:
   - loads the compiled TFLite with the Coral runtime
   - runs the prefix on the TPU
   - reads back the prefix outputs

3. **Boundary tensor handoff from TPU to suffix**
   The TPU-produced prefix outputs need to be reshaped and fed into the suffix graph exactly the same way the current CPU-only validation path does.

4. **Device-side timing and logging**
   The TPU path should save:
   - TPU inference time
   - host-side suffix time
   - boundary tensor sizes / transfer info
   - final predictions

5. **Job / artifact handoff**
   If Student C runs experiments on the Raspberry Pi + Coral device, the likely next addition is a small job manifest / executor layer rather than changing the existing task configs.

The key point is:

```text
the config system does not need to be replaced for TPU support
```

Instead, TPU support should plug into the current structure by consuming the same config-defined:

- `model_path`
- `boundary_tensors`
- `output_tensors`
- `artifacts_dir`

and by extending `updated_edgetpu_test.py` from:

- `full_cpu`
- `partitioned_cpu`

to eventually include a true:

- `tpu_cpu`

## Graph Visualization

`import_pb.py` still works as the original TensorBoard helper. Example:

```bash
./run_in_env.sh python import_pb.py \
  --graph=data/task_a/models/snapshot-700000.pb \
  --log_dir=./tb_logs

./run_in_env.sh python -m tensorboard.main --logdir=tb_logs --port=6006 --host=localhost
```

## Current Scope

This first pass still intentionally stops short of:

- TPU hardware execution
- heuristic partition ranking
- boundary proxy metric computation

Those pieces build on top of the current baseline rather than replacing it.
