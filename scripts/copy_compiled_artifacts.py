#!/usr/bin/env python3
"""Copy compiled _edgetpu.tflite files from compiled_artifacts/ into artifacts/.

For each of the 16 compiled candidates, copies:
  compiled_artifacts/<id>/tpu_int8_pure_edgetpu.tflite
  → artifacts/<id>/tpu_int8_pure_edgetpu.tflite

Does NOT move or delete the originals in compiled_artifacts/.
The 4 DeepLab failures (empty dirs) are skipped silently.
"""
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPILED_DIR = os.path.join(ROOT, "compiled_artifacts")
ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")

TFLITE_FILENAME = "tpu_int8_pure_edgetpu.tflite"

copied = []
skipped = []
errors = []

for candidate_id in sorted(os.listdir(COMPILED_DIR)):
    src_dir = os.path.join(COMPILED_DIR, candidate_id)
    if not os.path.isdir(src_dir):
        continue

    src_file = os.path.join(src_dir, TFLITE_FILENAME)
    if not os.path.exists(src_file):
        skipped.append(candidate_id)
        continue

    dst_dir = os.path.join(ARTIFACTS_DIR, candidate_id)
    if not os.path.isdir(dst_dir):
        errors.append(f"{candidate_id}: destination dir not found in artifacts/")
        continue

    dst_file = os.path.join(dst_dir, TFLITE_FILENAME)
    shutil.copy2(src_file, dst_file)
    src_size = os.path.getsize(src_file)
    dst_size = os.path.getsize(dst_file)
    if src_size != dst_size:
        errors.append(f"{candidate_id}: size mismatch src={src_size} dst={dst_size}")
    else:
        copied.append((candidate_id, dst_size))

print(f"\n{'='*60}")
print(f"COPIED ({len(copied)}):")
for cid, sz in copied:
    print(f"  ✅ {cid}  ({sz:,} bytes)")

print(f"\nSKIPPED — no edgetpu.tflite in compiled_artifacts/ ({len(skipped)}):")
for cid in skipped:
    print(f"  ⚠  {cid}")

if errors:
    print(f"\nERRORS ({len(errors)}):")
    for e in errors:
        print(f"  ❌ {e}")
    sys.exit(1)

print(f"\nDone. {len(copied)} files copied, {len(skipped)} skipped (expected 4 DeepLab failures).")
sys.exit(0)
