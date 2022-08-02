#!/bin/bash
set -o xtrace

icpx -fiopenmp -fopenmp-targets=spir64 -std=c++17 bench_omp.cpp main.cpp -DHOST_THREADS -o omp_host_threads
icpx -fiopenmp -fopenmp-targets=spir64 -std=c++17 bench_omp.cpp main.cpp -DNOWAIT -o omp_nowait

rm -f omp.log
export PrintDebugSettings=1

for envs in "ZE_AFFINITY_MASK=0.0" \
            "ZE_AFFINITY_MASK=0" \
            "ZE_AFFINITY_MASK=0.0 LIBOMPTARGET_LEVEL_ZERO_USE_IMMEDIATE_COMMAND_LIST=1 EnableFlushTaskSubmission=1" \
            "ZE_AFFINITY_MASK=0.0 LIBOMPTARGET_LEVEL_ZERO_USE_IMMEDIATE_COMMAND_LIST=1 EnableFlushTaskSubmission=1 LIBOMPTARGET_LEVEL0_USE_COPY_ENGINE=all"
do
    (
    export $envs
    for commands in "C C" "C M2D" "C D2M" "M2D D2M" "H2D D2H"
    do
        for mode in "host_threads" "nowait"
        do
            ./omp_$mode "$mode" $commands
        done
    done
   ) |& tee -a omp.log
done

./parse.py omp.log
