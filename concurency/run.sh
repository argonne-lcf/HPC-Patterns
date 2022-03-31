set -o xtrace
dpcpp sycl_con.cpp

(
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
) |& tee run.log

echo "#SUMMARY"
grep  -E '+./a.out|SUCCESS|FAILURE' run.log
