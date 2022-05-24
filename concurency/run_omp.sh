# Dpcpp

set -o xtrace
icpx -fiopenmp -fopenmp-targets=spir64 -std=c++17 omp_con.cpp -DHOST_THREADS -o host_threads
icpx -fiopenmp -fopenmp-targets=spir64 -std=c++17 omp_con.cpp -DNOWAIT -o nowait

rm -f omp.log omp_summary.log
export ZE_AFFINITY_MASK=0.0
export EnableFlushTaskSubmission=1
export PrintDebugSettings=1

for env in "ZE_AFFINITY_MASK=0.0" "ZE_AFFINITY_MASK=1.0" "LIBOMPTARGET_LEVEL_ZERO_COMMAND_BATCH=copy" "LIBOMPTARGET_LEVEL0_USE_COPY_ENGINE=all" "CFESingleSliceDispatchCCSMode"
do
    (
    export $env
    for commands in "C C" "C M2D" "C D2M" "M2D D2M" "H2D D2H"
    do
        for mode in "host_threads" "nowait"
        do

            ./$mode "$mode" $commands
            if [ $? == 0 ]; then
                ./$mode "$mode" $commands --enable_profiling
            fi
        done
    done
    ) |& tee -a omp.log
done

echo "#SUMMARY"
grep  -E 'export|./host_threads|./nowait|SUCCESS|FAILURE' omp.log |& tee omp_summary.log
./parse.py omp_summary.log
