# CS211 Project: Accuracy-Aware Model Partitioning for Coral Edge TPU

This repository keeps the original script-first layout from the previous team, but removes the most brittle hardcoded paths so we can build a cleaner Student A baseline around it.

The current baseline now covers:

- **Task A**: DeepLabCut animal pose estimation
- **Task B**: SSD MobileNet V2 object detection on COCO val2017
- **Task C**: DeepLab V3 semantic segmentation on Pascal VOC 2012 val

The split / TPU-prep path is only verified end to end for Task A right now. Tasks B and C are currently set up through the float32 baseline and metric-evaluation stage.

The model download/export step stays **outside** this repo. For DLC, that workflow lives in [model_export](/Users/jef/Desktop/219-project/model_export).

## Current Layout

### Top-level scripts

- [tensorflow_run.py](/Users/jef/Desktop/219-project/cs211-winter2024/tensorflow_run.py): float32 full-graph baseline runner
- [run_baseline.py](/Users/jef/Desktop/219-project/cs211-winter2024/run_baseline.py): float32 baseline runner; saves prediction baselines for all tasks and accuracy metrics where enabled
- [gen_tflite.py](/Users/jef/Desktop/219-project/cs211-winter2024/gen_tflite.py): exports the TPU prefix as a SavedModel for an arbitrary boundary
- [convert.py](/Users/jef/Desktop/219-project/cs211-winter2024/convert.py): converts the SavedModel prefix to TFLite
- [split.py](/Users/jef/Desktop/219-project/cs211-winter2024/split.py): exports split artifacts and metadata for a chosen boundary
- [updated_edgetpu_test.py](/Users/jef/Desktop/219-project/cs211-winter2024/updated_edgetpu_test.py): reusable `HybridRunner` validation path; currently used in CPU-only partitioned mode
- [auto_partition.py](/Users/jef/Desktop/219-project/cs211-winter2024/auto_partition.py): generic graph candidate enumerator plus TPU-compatibility/BFS scaffold for Student B to extend
- [import_pb.py](/Users/jef/Desktop/219-project/cs211-winter2024/import_pb.py): graph visualization helper for TensorBoard

### Minimal shared helpers

These scripts share a small helper layer under [src](/Users/jef/Desktop/219-project/cs211-winter2024/src):

- `config_utils.py`: load concise JSON task configs
- `graph_utils.py`: load/import frozen graphs, extract prefix/suffix graph defs
- `data_loaders.py`: AP-10K pose images, DLC video frames, COCO val images, and Pascal VOC val image+label loading
- `evaluation.py`: labeled accuracy metrics for tasks that enable them in config
- `io_utils.py`: save `.npz` outputs and JSON summaries

### Configs

- [configs/task_a_dlc.json](/Users/jef/Desktop/219-project/cs211-winter2024/configs/task_a_dlc.json): current working Task A config
- [configs/task_b_detection.json](/Users/jef/Desktop/219-project/cs211-winter2024/configs/task_b_detection.json): current working Task B config
- [configs/task_c_segmentation.json](/Users/jef/Desktop/219-project/cs211-winter2024/configs/task_c_segmentation.json): current working Task C config

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

For the baseline repo itself, you only need TensorFlow + NumPy + OpenCV. A minimal dependency list is in [requirements.txt](/Users/jef/Desktop/219-project/cs211-winter2024/requirements.txt).

For now, the easiest working environment is the same local conda env already used by the DLC export workflow:

```bash
cd /Users/jef/Desktop/219-project/cs211-winter2024
./run_in_env.sh python <script>.py ...
```

That keeps DeepLabCut and model export concerns outside this repo while still giving the baseline scripts a known-good TensorFlow environment.

## Task A Split Workflow

### 1. Float32 baseline

Run the full frozen graph entirely on CPU:

```bash
cd /Users/jef/Desktop/219-project/cs211-winter2024
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

### 4. Convert the prefix to TFLite

```bash
./run_in_env.sh python convert.py \
  --config configs/task_a_dlc.json \
  --model artifacts/task_a/dlc/prefix_saved_model \
  --output artifacts/task_a/dlc/output.tflite
```

This gives the TFLite prefix artifact that later TPU compilation will consume.

### 5. Export generalized split metadata

```bash
./run_in_env.sh python split.py --config configs/task_a_dlc.json --force
```

This writes:

- `artifacts/task_a/dlc/prefix_saved_model/`
- `artifacts/task_a/dlc/suffix_graph.pb`
- `artifacts/task_a/dlc/split_metadata.json`

### 6. Validate the split in CPU-only mode

```bash
./run_in_env.sh python updated_edgetpu_test.py --config configs/task_a_dlc.json --frame-limit 2
```

This writes:

- `artifacts/task_a/dlc/partitioned_cpu_outputs.npz`
- `artifacts/task_a/dlc/boundary_outputs.npz`
- `artifacts/task_a/dlc/partitioned_cpu_summary.json`

The current working DLC split reproduces the full float32 outputs exactly in CPU-only validation.

## Float32 Baselines

Use [run_baseline.py](/Users/jef/Desktop/219-project/cs211-winter2024/run_baseline.py) when you want a clean float32 baseline record saved. It always writes raw predictions and inference summaries; it only computes labeled accuracy metrics when the task config enables `compute_accuracy`.

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

## Graph Visualization

`import_pb.py` still works as the original TensorBoard helper. Example:

```bash
./run_in_env.sh python import_pb.py \
  --graph=data/task_a/models/snapshot-700000.pb \
  --log_dir=./tb_logs

../model_export/.conda-export/bin/tensorboard --logdir=tb_logs --port=6006 --host=localhost
```

## Current Scope

This first pass still intentionally stops short of:

- TPU hardware execution
- heuristic partition ranking
- boundary proxy metric computation

Those pieces build on top of the current baseline rather than replacing it.
