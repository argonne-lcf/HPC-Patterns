set -o xtrace

dpcpp sycl_con.cpp


( 
for commands in "C C" "C MD" "C DM" "MD DM"
do
  for mode in "out_of_order" "in_order"
  do
    ./a.out $mode $commands
    ./a.out $mode $commands --enable_profiling
  done
done
) |& tee run.log

echo "#SUMMARY"
egrep  -E '+./a.out|SUCCESS|FAILURE' run.log

