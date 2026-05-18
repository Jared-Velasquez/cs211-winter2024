# CS211 Project: Accuracy-Aware Model Partitioning for Coral Edge TPU

This repo keeps the original script-first layout, but makes it configurable enough to support the three Student A tasks:

- **Task A**: DeepLabCut pose estimation
- **Task B**: SSD MobileNet V2 object detection on COCO
- **Task C**: DeepLab V3 semantic segmentation on Pascal VOC

The model download/export step stays outside this repo. This repo assumes the models and datasets already exist under `data/`.

## Setup

Create the repo-local environment:

```bash
cd cs211-winter2024
./setup_env.sh
```

Run any script with:

```bash
./run_in_env.sh python <script>.py ...
```

Important files:

- [`environment.yml`](./environment.yml): conda environment spec
- [`requirements.txt`](./requirements.txt): minimal Python dependency list
- [`setup_env.sh`](./setup_env.sh): creates `.conda-baseline/`
- [`run_in_env.sh`](./run_in_env.sh): runs Python inside the repo-local env

## Config-Driven Flow

Each top-level script reads one config from [`configs/`](./configs):

- [`configs/task_a_dlc.json`](./configs/task_a_dlc.json)
- [`configs/task_b_detection.json`](./configs/task_b_detection.json)
- [`configs/task_c_segmentation.json`](./configs/task_c_segmentation.json)

The config controls:

- which frozen graph to load: `model_path`
- which inputs to load: `data_loader`, `images_dir` or `video_path`
- which tensors to run: `input_tensor`, `output_tensors`
- where to split the graph: `boundary_tensors`
- where outputs go: `artifacts_dir`
- whether labeled metrics are computed: `compute_accuracy`

Current split behavior is still **manual**: the partition boundary comes directly from `boundary_tensors` in the config.

Task A's TPU-prep config uses a static input contract:

- `resize`: `[640, 480]` (`[width, height]`)
- `fixed_input_shape`: `[1, 480, 640, 3]` (`[batch, height, width, channels]`)

This keeps the generated prefix TFLite model compatible with `edgetpu_compiler`, which rejects dynamic tensor shapes.

## Main Scripts

- [`tensorflow_run.py`](./tensorflow_run.py): run the full float32 frozen graph on CPU
  - writes `full_graph_outputs.npz` and `full_graph_summary.json`
- [`run_baseline.py`](./run_baseline.py): run the full baseline and save the baseline record
  - writes the same outputs as `tensorflow_run.py`, plus `baseline_results.json`
  - Task A is fidelity-only; Tasks B and C also compute labeled accuracy
- [`auto_partition.py`](./auto_partition.py): enumerate candidate graph points
  - writes `partition_candidates.json`
  - does not choose a final split yet
- [`gen_tflite.py`](./gen_tflite.py): extract the prefix subgraph at the config-defined boundary
  - writes `prefix_saved_model/` and `prefix_saved_model_metadata.json`
- [`convert.py`](./convert.py): convert the prefix SavedModel to TFLite
  - writes `output.tflite`
- [`split.py`](./split.py): export the split artifacts in one step
  - writes `prefix_saved_model/`, `suffix_graph.pb`, and `split_metadata.json`
- [`updated_edgetpu_test.py`](./updated_edgetpu_test.py): validate the split path
  - currently CPU-only
  - writes `partitioned_cpu_outputs.npz`, `boundary_outputs.npz`, and `partitioned_cpu_summary.json`
- [`run_hybrid.py`](./run_hybrid.py): run the generalized hybrid path
  - CPU-only mode validates the shared split runner without PyCoral
  - TPU mode loads a compiled Edge TPU prefix, dequantizes boundary tensors, and runs the CPU suffix
  - writes `hybrid_tpu_outputs.npz`, `hybrid_tpu_boundary_dequantized.npz`, `hybrid_float_boundary_outputs.npz`, and `hybrid_tpu_summary.json`
- [`import_pb.py`](./import_pb.py): load a frozen graph into TensorBoard logs

Shared helpers live under [`src/`](./src):

- `config_utils.py`
- `graph_utils.py`
- `data_loaders.py`
- `evaluation.py`
- `io_utils.py`

## Artifact Chain

The base model is always the original frozen TensorFlow graph:

- full graph: `.pb`

The current split/export chain is:

```text
full graph (.pb)
-> prefix SavedModel
-> prefix TFLite (output.tflite)

full graph (.pb)
-> suffix graph (.pb)
```

Important detail:

- the **prefix** is currently exported as a **SavedModel**, not as a raw prefix `.pb`
- the **suffix** is exported as `suffix_graph.pb`

## Typical Usage

### Full float32 baseline

```bash
./run_in_env.sh python run_baseline.py --config configs/task_a_dlc.json --frame-limit 100
./run_in_env.sh python run_baseline.py --config configs/task_b_detection.json --frame-limit 100
./run_in_env.sh python run_baseline.py --config configs/task_c_segmentation.json --frame-limit 100
```

This produces:

- `full_graph_outputs.npz`
- `full_graph_summary.json`
- `baseline_results.json`

Interpretation:

- **Task A**: saves float32 predictions only for later drift comparison
- **Task B/C**: saves float32 predictions plus labeled accuracy metrics

### Split / TPU-prep flow

```bash
./run_in_env.sh python auto_partition.py --config configs/task_a_dlc.json
./run_in_env.sh python split.py --config configs/task_a_dlc.json --force
./run_in_env.sh python convert.py --config configs/task_a_dlc.json --model artifacts/task_a/dlc/prefix_saved_model --output artifacts/task_a/dlc/output.tflite
./run_in_env.sh python updated_edgetpu_test.py --config configs/task_a_dlc.json --frame-limit 2
./run_in_env.sh python run_hybrid.py --config configs/task_a_dlc.json --cpu-only --frame-limit 2
```

What this means:

- `auto_partition.py` suggests possible graph points
- `split.py` cuts the graph at the chosen boundary and records prefix/suffix artifacts
- `convert.py` turns the prefix into TFLite
- `updated_edgetpu_test.py` checks that the split logic still reproduces the full output
- `run_hybrid.py --cpu-only` checks the new shared runner before TPU hardware is involved

After externally compiling `artifacts/task_a/dlc/output.tflite` with `edgetpu_compiler`, run the hybrid TPU path on a Coral machine:

```bash
./run_in_env.sh python run_hybrid.py \
  --config configs/task_a_dlc.json \
  --compiled-tflite artifacts/task_a/dlc/output_edgetpu.tflite \
  --frame-limit 2
```

If `--compiled-tflite` is omitted, the default is `output_edgetpu.tflite` under the task artifact directory.

If you change `resize`, `fixed_input_shape`, or `boundary_tensors`, regenerate artifacts in order:

```bash
./run_in_env.sh python split.py --config configs/task_a_dlc.json --force
./run_in_env.sh python convert.py --config configs/task_a_dlc.json --model artifacts/task_a/dlc/prefix_saved_model --output artifacts/task_a/dlc/output.tflite
edgetpu_compiler artifacts/task_a/dlc/output.tflite -o artifacts/task_a/dlc
```

`convert.py` validates static input/output shapes by default. Use `--allow-dynamic` only for non-TPU debugging.

For int8 calibration on an incomplete AP-10K download, skip missing images explicitly:

```bash
./run_in_env.sh python convert.py --config configs/task_a_dlc.json --model artifacts/task_a/dlc/prefix_saved_model --output artifacts/task_a/dlc/output_int8.tflite --opt int_fallback --frame-limit 100 --skip-missing-images
```

This is useful for representative calibration/debugging with partial data. A complete dataset is still preferable for final baseline or accuracy work.

## Current TPU Status

The repo currently stops **before** real Edge TPU execution.

What already exists:

- manual config-defined split points
- prefix extraction
- TFLite conversion
- suffix export
- CPU-only split validation
- PyCoral-backed hybrid runner for compiled Edge TPU prefixes

What is still needed outside this repo:

1. compile `output.tflite` with `edgetpu_compiler`
2. install the Coral runtime/PyCoral on the TPU host
3. run the hybrid command against the compiled model

The Edge TPU compiler is x86-64 only. ARM64 TPU machines can run the compiled `_edgetpu.tflite`, but should compile on an x86-64 Linux machine or in an x86-64 cloud/Colab environment and copy the compiled file back.

So:

- `updated_edgetpu_test.py` currently verifies **split correctness**
- `run_hybrid.py` runs the compiled Edge TPU prefix when PyCoral and hardware are available

PyCoral is intentionally imported lazily, so CPU-only workflows do not require the Coral runtime. Install PyCoral and the Edge TPU runtime on the Linux/Coral host that will execute `run_hybrid.py` in TPU mode.

## Data Layout

Expected layout:

```text
data/
  task_a/
    models/
    data/
  task_b/
    models/
    data/
  task_c/
    models/
    data/
artifacts/
  task_a/
  task_b/
  task_c/
```

The datasets and exported models are treated as local assets, not tracked repo content.

## Scope

This branch currently gives you:

- working config-driven baselines for Tasks A, B, and C
- saved float32 prediction baselines for all tasks
- labeled accuracy baselines for Tasks B and C
- a manual split/export pipeline
- CPU-only validation of the split path
- hybrid TPU-prefix / CPU-suffix execution for compiled Edge TPU prefix models

It does **not** yet give you:

- automatic partition selection
- boundary proxy metrics
