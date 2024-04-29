#!/usr/bin/env bash

num_gpu=$(/usr/bin/udevadm info /sys/module/i915/drivers/pci:i915/* |& grep -v Unknown | grep -c "P: /devices")
num_tile=2
num_socket=2

_MPI_RANKID=$PALS_LOCAL_RANKID

if [[ $1 == "compact" ]]; then
  gpu_id=$((_MPI_RANKID / num_tile))
  tile_id=$((_MPI_RANKID % num_tile))
  mask=$gpu_id.$tile_id
elif [[ $1 == "spread" ]]; then
  gpu_id=$((_MPI_RANKID % num_gpu))
  tile_id=$((_MPI_RANKID / num_gpu))
  mask=$gpu_id.$tile_id
elif [[ $1 == "compact_plan" ]]; then
  export ZES_ENABLE_SYSMAN=1
  mask=$(./topology $_MPI_RANKID)
fi
shift;

if [[ $1 == "ZAM" ]]; then
  export ZE_AFFINITY_MASK=$mask
elif [[ $1 == "ODS" ]]; then
  export ONEAPI_DEVICE_SELECTOR=level_zero:$mask
else
  exit "WRONG AFFINITY MECHANISM EITHER ZAM OR ODS"
fi
shift;

# Node 1, 3 are HBM
#num_gpu_per_socket=$((num_gpu / num_socket))
#numa_id=$((1+ gpu_id / num_gpu_per_socket))
#numactl -p $numa_id "$@"
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
"$@"
