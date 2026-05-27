#!/usr/bin/env python3
"""Update the three compile-results docs with per-operator breakdown tables.

For each compiled candidate, replaces the existing "Ops mapped to Edge TPU" line
with the real X/Y number plus a markdown per-operator table.
For the 4 DeepLab failures, verifies their rejection-reason block is correct.

Usage: python3 scripts/update_compile_docs.py
"""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")
DOCS_DIR = os.path.join(ROOT, "docs")

# Map each candidate to its doc file
DOC_MAP = {
    "dlc": os.path.join(DOCS_DIR, "dlc_compile_results.md"),
    "ssd": os.path.join(DOCS_DIR, "ssd_compile_results.md"),
    "deeplab": os.path.join(DOCS_DIR, "deeplab_compile_results.md"),
}

FAILED_DEEPLAB = {
    "deeplab_split_after_expanded_conv_16",
    "deeplab_split_after_aspp",
    "deeplab_split_after_logits",
    "deeplab_split_after_resize",
}


def status_icon(mapped: bool, status_str: str = "") -> str:
    if mapped:
        return "✅ Mapped"
    # Determine failure type from ops_breakdown context
    # We'll infer from the mapping flag; specific wording handled inline
    return "❌ Not mapped"


def build_breakdown_block(ops_breakdown: list[dict], ops_mapped_str: str) -> str:
    """Build the markdown ops section to replace existing lines."""
    lines = [f"**Ops mapped to Edge TPU:** {ops_mapped_str}"]
    lines.append("| Operator | Count | Status |")
    lines.append("|---|---|---|")
    for entry in ops_breakdown:
        op = entry["op"]
        count = entry["count"]
        mapped = entry["mapped"]
        if mapped:
            icon = "✅ Mapped"
        else:
            icon = "⚠ Not mapped"
        lines.append(f"| {op} | {count} | {icon} |")
    lines.append("")  # blank line after table for correct Markdown rendering
    return "\n".join(lines)


def load_metadata(candidate_id: str) -> dict:
    meta_path = os.path.join(ARTIFACTS_DIR, candidate_id, "metadata.json")
    with open(meta_path) as f:
        return json.load(f)


def update_doc(doc_path: str, candidates: list[str]) -> int:
    """Update a single doc file for the given list of candidates.
    Returns number of candidates updated."""
    with open(doc_path) as f:
        content = f.read()

    updated = 0
    for cid in candidates:
        if cid in FAILED_DEEPLAB:
            # Verify the doc already has the correct failure content — no replacement needed
            partition_id = cid.replace("deeplab_", "").replace("_", "-")
            # Actually partition ids use underscores in the docs too
            # Just check it has the rejection reason
            if "Compilation failed due to large activation tensors" not in content:
                print(f"  ⚠  {cid}: doc missing failure reason — check manually")
            else:
                print(f"  ✓  {cid}: failure reason already correct in doc")
            continue

        meta = load_metadata(cid)
        ops_mapped_str = meta.get("tpu_ops_mapped_edgetpu", "?/?")
        ops_breakdown = meta.get("edgetpu_ops_breakdown", [])

        if not ops_breakdown:
            print(f"  ⚠  {cid}: no breakdown in metadata, skipping doc update")
            continue

        # Build the new block
        new_block = build_breakdown_block(ops_breakdown, ops_mapped_str)

        # Pattern to find existing "Ops mapped to Edge TPU" line for this candidate
        # The line may be:
        #   **Ops mapped to Edge TPU:** <anything>
        # followed optionally by an existing table.
        # We replace from **Ops mapped** through the next blank line or next ** section.
        #
        # Strategy: within the candidate's section (between its ## header and the next ---),
        # find and replace the ops line + optional table block.

        # Find the candidate's section block
        # Sections are delimited by "## `<partition_id>`" headers and "---" lines
        partition_id = meta["partition_id"]
        # The section header looks like: ## `split_after_block1`
        section_pattern = re.compile(
            r'(## `' + re.escape(partition_id) + r'`.*?)'  # section header + content
            r'(?=\n---|\Z)',  # up to next --- or end of file
            re.DOTALL
        )
        section_m = section_pattern.search(content)
        if not section_m:
            print(f"  ⚠  {cid}: section '## `{partition_id}`' not found in {doc_path}")
            continue

        section_text = section_m.group(1)
        section_start = section_m.start(1)
        section_end = section_m.end(1)

        # Within the section, replace the "**Ops mapped to Edge TPU:**" line
        # and any immediately following table (lines starting with |)
        ops_pattern = re.compile(
            r'\*\*Ops mapped to Edge TPU:\*\*[^\n]*'   # the ops mapped line
            r'(?:\n\| [^\n]*)*'                         # optional table rows
            r'(?:\n\|---[^\n]*)*'                       # optional separator rows
            r'(?:\n\| [^\n]*)*',                        # optional more table rows
            re.MULTILINE
        )
        new_section = ops_pattern.sub(new_block, section_text, count=1)
        if new_section == section_text:
            print(f"  ⚠  {cid}: 'Ops mapped' line not found in section, skipping")
            continue

        content = content[:section_start] + new_section + content[section_end:]
        updated += 1
        print(f"  ✅ {cid}: updated with {ops_mapped_str} + {len(ops_breakdown)}-op table")

    with open(doc_path, "w") as f:
        f.write(content)

    return updated


def main():
    # Collect candidates by model
    dlc_candidates = sorted([
        d for d in os.listdir(ARTIFACTS_DIR)
        if d.startswith("dlc_") and os.path.isdir(os.path.join(ARTIFACTS_DIR, d))
    ])
    ssd_candidates = sorted([
        d for d in os.listdir(ARTIFACTS_DIR)
        if d.startswith("ssd_") and os.path.isdir(os.path.join(ARTIFACTS_DIR, d))
    ])
    deeplab_candidates = sorted([
        d for d in os.listdir(ARTIFACTS_DIR)
        if d.startswith("deeplab_") and os.path.isdir(os.path.join(ARTIFACTS_DIR, d))
    ])

    print("=== Updating DLC compile results ===")
    n = update_doc(DOC_MAP["dlc"], dlc_candidates)
    print(f"  → {n} sections updated\n")

    print("=== Updating SSD compile results ===")
    n = update_doc(DOC_MAP["ssd"], ssd_candidates)
    print(f"  → {n} sections updated\n")

    print("=== Updating DeepLab compile results ===")
    n = update_doc(DOC_MAP["deeplab"], deeplab_candidates)
    print(f"  → {n} sections updated\n")

    print("Done.")


if __name__ == "__main__":
    main()
