#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
reference_runner="$script_dir/run_upstream_scheduler_reference.sh"

cd "$repo_root"
reference=$(mktemp)
trap 'rm -f "$reference"' EXIT HUP INT TERM

# Compile/run the pinned upstream harness once for the complete corpus. The Rust
# comparator is already case-keyed and reports the first structural field mismatch,
# so recompiling the same C/C++ extractor once per case only adds validation latency.
"$reference_runner" > "$reference"
case_count=$(wc -l < "$reference" | tr -d ' ')
cargo run --quiet --example scheduler_report -- --check "$reference" > /dev/null
printf 'upstream scheduler differential: %d/%d cases matched\n' "$case_count" "$case_count"
