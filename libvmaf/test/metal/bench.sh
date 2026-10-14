#!/bin/sh
#
# Benchmark Metal feature extraction against the CPU baseline.
#
# usage: test/metal/bench.sh <feature[=options]|model> [frames]
#
#   feature: CPU "--feature X --no_prediction" vs Metal "--feature X_metal",
#            options as in "--feature X=key=value:..."
#   model:   $MODEL (default: the default model); CPU build vs Metal build
#            (Metal extractors are preferred, unported features fall back
#            to the CPU)
#
# Environment (paths relative to libvmaf/):
#   CPU_VMAF    CPU baseline binary      (default: build/tools/vmaf)
#   METAL_VMAF  Metal-enabled binary     (default: build-metal/tools/vmaf)
#   REF, DIST   inputs                   (default: build/*.y4m test clips)
#   THREADS     multi-thread count       (default: performance cores)
#   THREAD_LIST thread counts to run     (default: "1 $THREADS"; 0 runs
#               the extractors on the calling thread, the CLI default)
#   RUNS        runs per config, best is reported (default: 1)
#   MODEL       model for "model", e.g. "version=vmaf_v1.0.16_3d0h"
#
# Prints a markdown table.

set -e
cd "$(dirname "$0")/../.."

MODE=${1:?usage: $0 <feature|model> [frames]}
FRAMES=${2:-}
CPU_VMAF=${CPU_VMAF:-build/tools/vmaf}
METAL_VMAF=${METAL_VMAF:-build-metal/tools/vmaf}
REF=${REF:-build/pedestrian_area_1080p25.y4m}
DIST=${DIST:-build/av1.y4m}
THREADS=${THREADS:-$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo 8)}
RUNS=${RUNS:-1}

frame_args=""
[ -n "$FRAMES" ] && frame_args="--frame_cnt $FRAMES"

# warm the page cache
cat "$REF" "$DIST" > /dev/null

# run <binary> <threads> <extra args...>: prints best wall time in seconds
run() {
    bin=$1; threads=$2; shift 2
    best=""
    i=0
    while [ $i -lt "$RUNS" ]; do
        t=$( { /usr/bin/time -p "$bin" -r "$REF" -d "$DIST" $frame_args \
                 --threads "$threads" -q -o /dev/null "$@" > /dev/null; } 2>&1 |
             awk '/^real/ { print $2 }')
        if [ -z "$best" ] || awk "BEGIN { exit !($t < $best) }"; then
            best=$t
        fi
        i=$((i + 1))
    done
    echo "$best"
}

n_frames=$FRAMES
if [ -z "$n_frames" ]; then
    size=$(stat -f %z "$REF")
    header=$(head -1 "$REF")
    w=$(echo "$header" | sed -E 's/.* W([0-9]+).*/\1/')
    h=$(echo "$header" | sed -E 's/.* H([0-9]+).*/\1/')
    n_frames=$(( size / (w * h * 3 / 2 + 6) ))
fi

if [ "$MODE" = model ]; then
    cpu_args=${MODEL:+--model $MODEL}
    metal_args=$cpu_args
else
    name=${MODE%%=*}
    opts=""
    [ "$name" != "$MODE" ] && opts="=${MODE#*=}"
    cpu_args="--no_prediction --feature $MODE"
    metal_args="--no_prediction --feature ${name}_metal$opts"
fi

echo "| config | threads | time (s) | fps | speedup vs CPU |"
echo "|---|---:|---:|---:|---:|"
for t in ${THREAD_LIST:-1 $THREADS}; do
    cpu=$(run "$CPU_VMAF" "$t" $cpu_args)
    exact=$(run "$METAL_VMAF" "$t" $metal_args)
    fast=$(run "$METAL_VMAF" "$t" $metal_args --metal_fast)
    for row in "CPU:$cpu" "Metal exact:$exact" "Metal fast:$fast"; do
        name=${row%%:*}
        time=${row#*:}
        fps=$(awk "BEGIN { printf \"%.1f\", $n_frames / $time }")
        speedup=$(awk "BEGIN { printf \"%.2f\", $cpu / $time }")
        echo "| $name | $t | $time | $fps | ${speedup}x |"
    done
done
