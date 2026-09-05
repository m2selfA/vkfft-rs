#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
reference_runner="$script_dir/run_upstream_scheduler_reference.sh"

cd "$repo_root"
cases=$(mktemp)
reference=$(mktemp)
success_names=$(mktemp)
success_cases=$(mktemp)
classification_reference=$(mktemp)
classification_names=$(mktemp)
classification_cases=$(mktemp)
classification_candidate=$(mktemp)
physical_inputs=$(mktemp)
physical_reference=$(mktemp)
physical_names=$(mktemp)
physical_cases=$(mktemp)
trap 'rm -f "$cases" "$reference" "$success_names" "$success_cases" "$classification_reference" "$classification_names" "$classification_cases" "$classification_candidate" "$physical_inputs" "$physical_reference" "$physical_names" "$physical_cases"' EXIT HUP INT TERM

# TSV schema (18-field extended form; the readers also retain legacy 15/17-field support):
# case profile precision sequence shared pow2 max_threads max_workgroup
# strided fastest_axis bandwidth batch grouped_batch zero_padding min_direct max_direct min_fft max_fft
#
# A small 21-field axis2 witness set appends: upstream_axis_id middle_axis_len
# axis1_grouped_batch before the four tuning fields. This keeps groupedBatch[1]'s
# literal upstream gate independent from groupedBatch[axis_id] without multiplying
# the whole parameter grid by another synthetic dimension.
#
# The grid intentionally describes only scheduler inputs. The pinned C harness derives
# the actual Rader containers from VkFFTScheduler, while the Rust candidate independently
# classifies the same axis through FftPlan before invoking its upload splitter.
profiles="nv-vk amd-vk intel-cl intel-vk intel-l0"
precisions="f32 f16 f64f32 dd"
sequences="1088 1984 1922 33728 8789 9367 102272 139264 278528 1114112"
resources="tiny mid native"
axes="c s0-1 s0-8 s0-64 s2-1 s2-8 s2-64"
batch_modes="b1a b1g1 b5g1 b5g3 b2g3"
padding_modes="z0 z1"

for profile in $profiles; do
    for precision in $precisions; do
        min_direct=17
        max_direct=89
        min_fft=17
        max_fft=16384
        case "$profile" in
            intel-cl|intel-vk|intel-l0) max_direct=17 ;;
        esac
        case "$precision" in
            dd)
                min_direct=11
                max_direct=29
                case "$profile" in
                    amd-vk) min_fft=19 ;;
                    *) min_fft=17 ;;
                esac
                ;;
            f64f32)
                case "$profile" in
                    amd-vk) min_fft=29 ;;
                    *) min_fft=17 ;;
                esac
                ;;
        esac

        for resource in $resources; do
            case "$resource" in
                tiny)
                    shared=2048; pow2=2048; max_threads=64; max_workgroup=64
                    ;;
                mid)
                    shared=8192; pow2=8192; max_threads=256; max_workgroup=256
                    ;;
                native)
                    case "$profile" in
                        nv-vk)
                            shared=49152; pow2=32768; max_threads=1024; max_workgroup=1024
                            ;;
                        amd-vk)
                            shared=65536; pow2=65536; max_threads=1024; max_workgroup=1024
                            ;;
                        intel-cl)
                            shared=32768; pow2=32768; max_threads=256; max_workgroup=256
                            ;;
                        intel-vk)
                            shared=32768; pow2=32768; max_threads=1024; max_workgroup=1024
                            ;;
                        intel-l0)
                            shared=65536; pow2=65536; max_threads=1024; max_workgroup=1024
                            ;;
                    esac
                    ;;
            esac

            for sequence in $sequences; do
                for axis in $axes; do
                    case "$axis" in
                        c) strided=0; fastest=0; bandwidth=0 ;;
                        s0-1) strided=1; fastest=1; bandwidth=0 ;;
                        s0-8) strided=1; fastest=8; bandwidth=0 ;;
                        s0-64) strided=1; fastest=64; bandwidth=0 ;;
                        s2-1) strided=1; fastest=1; bandwidth=2 ;;
                        s2-8) strided=1; fastest=8; bandwidth=2 ;;
                        s2-64) strided=1; fastest=64; bandwidth=2 ;;
                    esac
                    for batch_mode in $batch_modes; do
                        case "$batch_mode" in
                            b1a) batch=1; grouped=0 ;;
                            b1g1) batch=1; grouped=1 ;;
                            b5g1) batch=5; grouped=1 ;;
                            b5g3) batch=5; grouped=3 ;;
                            b2g3) batch=2; grouped=3 ;;
                        esac
                        for padding_mode in $padding_modes; do
                            case "$padding_mode" in
                                z0) zero_padding=0 ;;
                                z1) zero_padding=1 ;;
                            esac
                            case_name="sweep-${profile}-${precision}-n${sequence}-${resource}-${axis}-${batch_mode}-${padding_mode}"
                            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                                "$case_name" "$profile" "$precision" "$sequence" \
                                "$shared" "$pow2" "$max_threads" "$max_workgroup" \
                                "$strided" "$fastest" "$bandwidth" "$batch" "$grouped" "$zero_padding" \
                                "$min_direct" "$max_direct" "$min_fft" "$max_fft" >> "$cases"
                        done
                    done
                done
            done
        done
    done
done

# Directed axis-id=2 witnesses. The target groupedBatch[2] stays fixed while
# groupedBatch[1] changes independently, covering ordinary/DD, NVIDIA/AMD,
# and padded/unpadded physical paths without mechanically multiplying the grid.
cat >> "$cases" <<'EOF'
axis2-amd-dd-g3-g0	amd-vk	dd	1088	8192	8192	256	256	1	8	0	1	3	0	2	4	0	11	29	19	16384
axis2-amd-dd-g3-g8	amd-vk	dd	1088	8192	8192	256	256	1	8	0	1	3	0	2	4	8	11	29	19	16384
axis2-amd-f16-g3-g0	amd-vk	f16	8789	65536	65536	1024	1024	1	64	0	5	3	0	2	8	0	17	89	17	16384
axis2-amd-f16-g3-g8	amd-vk	f16	8789	65536	65536	1024	1024	1	64	0	5	3	0	2	8	8	17	89	17	16384
axis2-nv-f32-g3-g0	nv-vk	f32	102272	49152	32768	1024	1024	1	64	0	5	3	0	2	8	0	17	89	17	16384
axis2-nv-f32-g3-g16	nv-vk	f32	102272	49152	32768	1024	1024	1	64	0	5	3	0	2	8	16	17	89	17	16384
axis2-amd-dd-pad-g3-g0	amd-vk	dd	1088	8192	8192	256	256	1	8	0	2	3	1	2	4	0	11	29	19	16384
axis2-amd-dd-pad-g3-g8	amd-vk	dd	1088	8192	8192	256	256	1	8	0	2	3	1	2	4	8	11	29	19	16384
axis2-intelvk-f32-n278528-g8-b0-z0	intel-vk	f32	278528	32768	32768	1024	1024	1	8	0	5	3	0	2	4	8	17	17	17	16384
axis2-intelvk-f32-n278528-g8-b2-z0	intel-vk	f32	278528	32768	32768	1024	1024	1	8	2	5	3	0	2	4	8	17	17	17	16384
axis2-intelvk-f32-n278528-g8-b0-z1	intel-vk	f32	278528	32768	32768	1024	1024	1	8	0	5	3	1	2	4	8	17	17	17	16384
axis2-intelvk-f32-n278528-g8-b2-z1	intel-vk	f32	278528	32768	32768	1024	1024	1	8	2	5	3	1	2	4	8	17	17	17	16384
axis2-amd-dd-n1114112-g8-b0-z0	amd-vk	dd	1114112	65536	65536	1024	1024	1	8	0	5	3	0	2	4	8	11	29	19	16384
axis2-amd-dd-n1114112-g8-b2-z0	amd-vk	dd	1114112	65536	65536	1024	1024	1	8	2	5	3	0	2	4	8	11	29	19	16384
axis2-amd-dd-n1114112-g8-b0-z1	amd-vk	dd	1114112	65536	65536	1024	1024	1	8	0	5	3	1	2	4	8	11	29	19	16384
axis2-amd-dd-n1114112-g8-b2-z1	amd-vk	dd	1114112	65536	65536	1024	1024	1	8	2	5	3	1	2	4	8	11	29	19	16384
EOF

"$reference_runner" --rader-upload-file "$cases" > "$reference"
sed -n 's/.*"case":"\([^"]*\)".*/\1/p' "$reference" > "$success_names"
awk -F '\t' 'NR==FNR { ok[$1] = 1; next } ($1 in ok)' "$success_names" "$cases" > "$success_cases"
case_count=$(wc -l < "$reference" | tr -d ' ')
requested_count=$(wc -l < "$cases" | tr -d ' ')
skipped_count=$((requested_count - case_count))
if [ "$case_count" -eq 0 ]; then
    echo "parameterized upstream sweep produced no supported cases" >&2
    exit 2
fi
cargo run --quiet --example scheduler_report -- \
    --rader-upload-file "$success_cases" --check "$reference" > /dev/null
printf 'parameterized upstream Rader-upload differential: %d/%d supported cases matched (%d upstream unsupported skipped)\n' \
    "$case_count" "$case_count" "$skipped_count"

"$reference_runner" --classification-file "$cases" > "$classification_reference"
cut -f 1 "$classification_reference" > "$classification_names"
awk -F '\t' 'NR==FNR { ok[$1] = 1; next } ($1 in ok)' "$classification_names" "$cases" > "$classification_cases"
classification_count=$(wc -l < "$classification_reference" | tr -d ' ')
classification_skipped=$((requested_count - classification_count))
if [ "$classification_count" -eq 0 ]; then
    echo "parameterized upstream classification sweep produced no supported cases" >&2
    exit 2
fi
cargo run --quiet --example scheduler_report -- \
    --classification-file "$classification_cases" > "$classification_candidate"
if ! diff -u "$classification_reference" "$classification_candidate"; then
    echo "parameterized upstream algorithm classification differs from Rust candidate" >&2
    exit 1
fi
printf 'parameterized upstream algorithm classification: %d/%d supported cases matched (%d upstream unsupported skipped)\n' \
    "$classification_count" "$classification_count" "$classification_skipped"

# Physical AxisBlock parity covers both contiguous and strided batch1 upstream-Rader
# multi-upload cases. The upstream harness executes VkFFTSplitAxisBlock per upload; Rust
# reads the corresponding physical blocks from its materialized recursive IR, with strided
# probes retagged through the production higher-axis path using fastest_axis_len.
cat "$cases" > "$physical_inputs"
"$reference_runner" --axis-block-file "$physical_inputs" > "$physical_reference"
sed -n 's/.*"case":"\([^"]*\)-u[0-9][0-9]*".*/\1/p' "$physical_reference" | sort -u > "$physical_names"
awk -F '\t' 'NR==FNR { ok[$1] = 1; next } ($1 in ok)' "$physical_names" "$physical_inputs" > "$physical_cases"
physical_record_count=$(wc -l < "$physical_reference" | tr -d ' ')
physical_case_count=$(wc -l < "$physical_names" | tr -d ' ')
physical_requested_count=$(wc -l < "$physical_inputs" | tr -d ' ')
physical_skipped=$((physical_requested_count - physical_case_count))
if [ "$physical_record_count" -eq 0 ]; then
    echo "parameterized upstream physical AxisBlock sweep produced no records" >&2
    exit 2
fi
cargo run --quiet --example scheduler_report -- \
    --axis-block-file "$physical_cases" --check "$physical_reference" > /dev/null
printf 'parameterized upstream physical AxisBlock: %d/%d records matched across %d cases (%d upstream single/unsupported/Bluestein skipped)\n' \
    "$physical_record_count" "$physical_record_count" "$physical_case_count" "$physical_skipped"
