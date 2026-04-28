#!/usr/bin/env python3
"""Strip non-functional custom calls from whole_computation HLO files.

Removes xla_ffi_python_gpu_callback (JAX profiling markers) and
Sharding custom calls that prevent standalone run_hlo_module execution.

Usage:
    python3 strip_custom_calls.py ~/hlo_datasets/*/whole_computation/*.hlo
    python3 strip_custom_calls.py --hlo-dir ~/hlo_datasets --validate \
        --xla-tool ~/xla/bazel-bin/xla/tools/run_hlo_module
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
from pathlib import Path


def strip_custom_calls(hlo_text: str) -> tuple[str, int]:
    """Remove non-functional custom call instructions from HLO text.

    Strips:
    1. xla_ffi_python_gpu_callback lines (mark_start__, mark_end__)
    2. Sharding custom calls (sharding_constraint)

    Returns (cleaned_text, num_removed).
    """
    lines = hlo_text.split('\n')
    cleaned = []
    removed = 0

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip xla_ffi_python_gpu_callback lines (may span multiple lines)
        if 'xla_ffi_python_gpu_callback' in stripped:
            # Consume continuation lines (no '=' at start means continuation)
            removed += 1
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith(('%', 'ROOT', '}')):
                # Check if this is a continuation of the custom call
                if '=' in lines[i] or lines[i].strip().startswith(('%', 'ROOT', '}')):
                    break
                i += 1
            continue

        # Skip Sharding custom calls
        if 'custom_call_target="Sharding"' in stripped:
            # This instruction assigns to a variable. We need to replace
            # usages of this variable with its input operand.
            removed += 1
            i += 1
            continue

        cleaned.append(line)
        i += 1

    return '\n'.join(cleaned), removed


def fix_sharding_references(hlo_text: str) -> str:
    """Replace references to removed sharding_constraint with its input.

    For lines like:
        sharding_constraint.1 = f32[...] custom-call(dot_general.21), custom_call_target="Sharding"
    The output (sharding_constraint.1) should be replaced with the input (dot_general.21)
    everywhere in the file.
    """
    # Find sharding constraint definitions
    pattern = r'^\s+(\S+)\s*=\s*\S+\s+custom-call\((\S+)\),\s*custom_call_target="Sharding"'
    replacements = {}
    for match in re.finditer(pattern, hlo_text, re.MULTILINE):
        output_name = match.group(1)
        input_name = match.group(2).rstrip(',)')
        replacements[output_name] = input_name

    # Apply replacements (but not in the definition line itself)
    result = hlo_text
    for old, new in replacements.items():
        # Replace references but not definitions
        # Use word boundary to avoid partial matches
        result = re.sub(
            r'(?<![=%\w])' + re.escape(old) + r'(?![.\w])',
            new,
            result,
        )

    return result


def process_hlo_file(input_path: str, output_path: str = None) -> int:
    """Process a single HLO file, stripping custom calls.

    Args:
        input_path: Path to original HLO file.
        output_path: Path for cleaned file. If None, overwrites input.

    Returns:
        Number of instructions removed.
    """
    if output_path is None:
        output_path = input_path

    text = Path(input_path).read_text()

    # First fix sharding references (before removing the lines)
    text = fix_sharding_references(text)

    # Then strip the custom call lines
    cleaned, num_removed = strip_custom_calls(text)

    Path(output_path).write_text(cleaned)
    return num_removed


def validate_hlo(xla_tool: str, hlo_path: str) -> bool:
    """Test that a cleaned HLO compiles and runs via run_hlo_module."""
    try:
        result = subprocess.run(
            [xla_tool, "--platform=CUDA", "--reference_platform=",
             "--input_format=hlo", hlo_path],
            capture_output=True, text=True, timeout=120,
        )
        if "compiled and ran" in result.stderr:
            return True
        if result.returncode == 0:
            return True
        print(f"  [FAIL] {result.stderr[-300:]}")
        return False
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT]")
        return False
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Strip custom calls from whole_computation HLO files"
    )
    parser.add_argument("files", nargs="*", help="HLO files to process")
    parser.add_argument("--hlo-dir",
                        help="HLO dataset directory (process all whole_computation)")
    parser.add_argument("--output-suffix", default="_stripped",
                        help="Suffix for output files (default: _stripped)")
    parser.add_argument("--in-place", action="store_true",
                        help="Overwrite original files")
    parser.add_argument("--validate", action="store_true",
                        help="Validate cleaned files with run_hlo_module")
    parser.add_argument("--xla-tool",
                        help="Path to run_hlo_module (required for --validate)")
    args = parser.parse_args()

    files = list(args.files) if args.files else []

    if args.hlo_dir:
        for model_dir in sorted(glob.glob(os.path.join(args.hlo_dir, "*"))):
            wc_dir = os.path.join(model_dir, "whole_computation")
            if os.path.isdir(wc_dir):
                for hlo in glob.glob(os.path.join(wc_dir, "*.hlo")):
                    files.append(hlo)

    if not files:
        parser.error("No HLO files specified. Use positional args or --hlo-dir")

    if args.validate and not args.xla_tool:
        parser.error("--xla-tool required for --validate")

    print(f"Processing {len(files)} HLO files")

    for hlo_path in files:
        model_name = os.path.basename(os.path.dirname(os.path.dirname(hlo_path)))
        basename = os.path.splitext(os.path.basename(hlo_path))[0]

        if args.in_place:
            output_path = hlo_path
        else:
            output_path = os.path.join(
                os.path.dirname(hlo_path),
                f"{basename}{args.output_suffix}.hlo",
            )

        num_removed = process_hlo_file(hlo_path, output_path)
        print(f"  {model_name}/whole_computation: removed {num_removed} custom calls → {os.path.basename(output_path)}")

        if args.validate:
            print(f"    Validating...", end=" ", flush=True)
            if validate_hlo(args.xla_tool, output_path):
                print("OK")
            else:
                print("FAILED")


if __name__ == "__main__":
    main()
