#!/bin/bash
set -o xtrace

cd $(dirname $0)

mpicxx -cxx=icpx -fsycl -std=c++17 -DUSE_CCL -I$CCL_ROOT/include \
       bench_ccl.cpp main.cpp -o ccl_con -L$CCL_ROOT/lib -lccl

rm -f ccl.log

LCOMMANDS=("C P" "P C" "P P")

for mode in "in_order" "out_of_order"
do
    mpirun -n 2 ./ccl_env.sh ./ccl_con "$mode" ${LCOMMANDS[@]/#/--commands }
done |& tee -a ccl.log

../../parse.py ccl.log 2>/dev/null || true
