# Dpcpp

set -o xtrace
dpcpp sycl_con.cpp 
rm -f sycl.log sycl_summary.log
export ZE_AFFINITY_MASK=0.0
export PrintDebugSettings=1
export EnableFlushTaskSubmission=1

for env in "ZE_AFFINITY_MASK=0.0" "ZE_AFFINITY_MASK=1.0" "SYCL_PI_LEVEL_ZERO_BATCH_SIZE=40" "CFESingleSliceDispatchCCSMode=1"
do
    (
    export $env
    for commands in "C C" "C M2D" "C D2M" "M2D D2M" "H2D D2H"
    do
        for mode in "out_of_order" "in_order"
        do
            ./a.out "$mode" $commands
            if [ $? == 0 ]; then
                ./a.out "$mode" $commands --enable_profiling
            fi
        done
    done
    ) |& tee -a sycl.log
done

echo "#SUMMARY"
grep  -E 'export|./a.out|SUCCESS|FAILURE' sycl.log |& tee sycl_summary.log
./parse.py sycl_summary.log
