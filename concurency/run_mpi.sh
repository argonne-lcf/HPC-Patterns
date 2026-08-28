#!/bin/bash
set -o errexit
set -o pipefail
set -o xtrace

cd "$(dirname "$0")"

mpicxx -cxx=icpx -fsycl -std=c++17 -DUSE_MPI bench_mpi.cpp main.cpp -o mpi_con

rm -f mpi.log
export ZE_FLAT_DEVICE_HIERARCHY=FLAT

LCOMMANDS=("C P" "P C" "P P")

for mode in "in_order" "out_of_order"
do
    mpirun -n 2 ./mpi_con "$mode" ${LCOMMANDS[@]/#/--commands }
done |& tee -a mpi.log

./parse.py mpi.log 2>/dev/null || true
