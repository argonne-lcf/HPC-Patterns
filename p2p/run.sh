#!/bin/bash -x

icpx -lze_loader topology.cpp -o topology
mpicxx -fsycl peer2pear.cpp -o peer2pear_i
mpicxx -fsycl -DUSE_WIN peer2pear.cpp -o peer2pear_w

export MPIR_CVAR_CH4_IPC_GPU_ENGINE_TYPE=copy_high_bandwidth

for mode in compact spread compact_plan
do
  for affinity_metchanism in ZAM ODS 
  do
    for bin in peer2pear_i peer2pear_w
    do
      for n in 2 12
      do 
         mpirun -n $n -- ./tile_mapping.sh $mode $affinity_metchanism ./$bin "$bin $n $mode $affinity_metchanism"
      done
    done
  done
done
