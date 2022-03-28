set -o xtrace
dpcpp sycl_con.cpp

(
for commands in "C C" "C MD" "C DM" "MD DM"
do
  for mode in "out_of_order" "in_order"
  do
    ./a.out "$mode" $commands
    if [ $? == 0 ]; then
        ./a.out "$mode" $commands --enable_profiling
    fi
  done
done
) |& tee run.log

echo "#SUMMARY"
grep  -E '+./a.out|SUCCESS|FAILURE' run.log
