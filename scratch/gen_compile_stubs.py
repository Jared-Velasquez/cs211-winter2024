"""Generate per-model compile-results stub docs from metadata.json files."""
import glob
import json
import os

REPO = "/Users/jaredvelasquez/projects/cs211-winter2024"

MODEL_CONFIG = {
    "dlc_resnet50": {
        "prefix": "dlc_",
        "input_size_arg": "-w 320 -t 320",
        "calibration": "rhino_5.mp4 (>=200 frames; coordinate with Student A)",
        "out_doc": f"{REPO}/docs/dlc_compile_results.md",
        "title": "DLC Compile Results",
        "model_path_hint": "snapshot-1000.pb",
    },
    "ssd_mobilenet_v2": {
        "prefix": "ssd_",
        "input_size_arg": "-w 300 -t 300",
        "calibration": "COCO val2017 images (coordinate set with Student A)",
        "out_doc": f"{REPO}/docs/ssd_compile_results.md",
        "title": "SSD MobileNet V2 Compile Results",
        "model_path_hint": "artifacts/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb",
    },
    "deeplab_v3_mnv2": {
        "prefix": "deeplab_",
        "input_size_arg": "-w 513 -t 513",
        "calibration": "Pascal VOC 2012 val images (coordinate set with Student A)",
        "out_doc": f"{REPO}/docs/deeplab_compile_results.md",
        "title": "DeepLab V3 (MobileNet V2 backbone) Compile Results",
        "model_path_hint": "artifacts/deeplabv3_mnv2_pascal_trainval/frozen_inference_graph.pb",
    },
}


def load_all_metadata():
    out = {}
    for path in glob.glob(f"{REPO}/artifacts/*/metadata.json"):
        with open(path) as f:
            meta = json.load(f)
        model = meta["model"]
        out.setdefault(model, []).append(meta)
    for model in out:
        out[model].sort(key=lambda m: m["num_tpu_ops"])
    return out


HEADER = """# {title}

**Status:** NOT YET COMPILED — `edgetpu_compiler` and `pycoral` are not installed on the development machine.
Run the commands below on a host with the Coral toolchain (see [Coral docs](https://coral.ai/docs/edgetpu/compiler/)).

For each candidate:
1. Run `convert.py` to quantize the TPU SavedModel into a `.tflite` (this is the slow step — needs the calibration set).
2. Run `edgetpu_compiler` to map the int8 `.tflite` to TPU ops.
3. Pair with Student C for a one-shot runtime sanity check before declaring the candidate **valid**.

If `int8_pure` fails, fall back to `int_fallback`. Always try `int8_pure` first; record both results.

**Calibration source:** {calibration}

---
"""

CANDIDATE_TEMPLATE = """## `{pid}`

- **TPU SavedModel:** `artifacts/{prefix}{pid}/tpu_savedmodel/`
- **CPU SavedModel:** `artifacts/{prefix}{pid}/cpu_savedmodel/`
- **Boundary:** `{boundary_tensors}`
- **Boundary shape(s):** `{shapes}`  (bandwidth ≈ {bandwidth_kib:.1f} KiB at int8)
- **Static op counts:** TPU = {num_tpu_ops}, CPU = {num_cpu_ops}
- **`has_skip_crossing`:** {has_skip_crossing} *(conservative — see partition_points_*.md notes)*

**Quantize (pure int8):**
```
venv/bin/python convert.py -m artifacts/{prefix}{pid}/tpu_savedmodel \\
  -O int8_pure -r <CALIBRATION_SRC> {input_size_arg} \\
  -o artifacts/{prefix}{pid}/tpu_int8.tflite
```

**Quantize (int8 with float fallback):**
```
venv/bin/python convert.py -m artifacts/{prefix}{pid}/tpu_savedmodel \\
  -O int_fallback -r <CALIBRATION_SRC> {input_size_arg} \\
  -o artifacts/{prefix}{pid}/tpu_int_fallback.tflite
```

**Compile:**
```
edgetpu_compiler -o artifacts/{prefix}{pid}/ artifacts/{prefix}{pid}/tpu_int8.tflite
# (or tpu_int_fallback.tflite if pure-int8 failed)
```

**Result:** ☐ int8_pure  ☐ int_fallback  ☐ both failed
**Ops mapped to Edge TPU:** _(fill in from compiler stdout)_
**Rejection reason (if any):** _(paste compiler log line)_
**Runtime sanity (Student C):** ☐ produces output on one sample

---
"""


def main():
    metadata = load_all_metadata()
    for model, cfg in MODEL_CONFIG.items():
        cands = metadata.get(model, [])
        out = [HEADER.format(title=cfg["title"], calibration=cfg["calibration"])]
        for m in cands:
            out.append(CANDIDATE_TEMPLATE.format(
                pid=m["partition_id"],
                prefix=cfg["prefix"],
                boundary_tensors=", ".join(m["tpu_output_tensors"]),
                shapes=", ".join(str(s) for s in m["boundary_tensor_shapes"]),
                bandwidth_kib=m["boundary_bandwidth_bytes"] / 1024,
                num_tpu_ops=m["num_tpu_ops"],
                num_cpu_ops=m["num_cpu_ops"],
                has_skip_crossing=m["has_skip_crossing"],
                input_size_arg=cfg["input_size_arg"],
            ))
        with open(cfg["out_doc"], "w") as f:
            f.write("".join(out))
        print(f"Wrote {cfg['out_doc']} ({len(cands)} candidates)")


if __name__ == "__main__":
    main()
