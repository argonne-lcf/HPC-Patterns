set -o xtrace

dpcpp sycl_con.cpp

kernelTripcount=100000

( 
for commands in "C C" "C MD" "C DM" "MD DM"
do
  for mode in "out_of_order" "in_order"
  do
    ./a.out $mode $commands --kernel_tripcount $kernelTripcount
    ./a.out $mode $commands --kernel_tripcount $kernelTripcount --enable_profiling
  done
done
) |& tee run.log

echo "#SUMMARY"
egrep  -E '+./a.out|SUCCESS|FAILURE' run.log

