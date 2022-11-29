#!/bin/bash
set -o xtrace

cd $(mktemp -d tmp-sycl-XXXX)

icpx -fsycl -std=c++17 ../bench_sycl.cpp ../main.cpp -o sycl

rm -f sycl.log
export PrintDebugSettings=1

LCOMMANDS=("C C" "C M2D" "C D2M" "M2D D2M" "H2D D2H")

for envs in "ZE_AFFINITY_MASK=0.0" \
            "ZE_AFFINITY_MASK=0" \
            "ZE_AFFINITY_MASK=0.0 SYCL_PI_LEVEL_ZERO_BATCH_SIZE=40" \
           "ZE_AFFINITY_MASK=0.0 CFESingleSliceDispatchCCSMode=1" \
           "ZE_AFFINITY_MASK=0.0 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 EnableFlushTaskSubmission=1"
do
    (
    export $envs
    for mode in "out_of_order" "in_order"
    do
     	./sycl "$mode" ${LCOMMANDS[@]/#/--commands }
        #./sycl "$mode" ${COMMANDS[@]/#/--commands } --enable_profiling
    done
    ) |& tee -a sycl.log
done

../parse.py sycl.log
