set -o xtrace

for c in "C C" "C MD" "C DM" "MD DM"
do
  for p in "disable_profiling" "enable_profiling"
  do
    ./a.out 800000 $p out_of_order 1 $c
    ./a.out 800000 $p in_order 2 $c
  done
done
