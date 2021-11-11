#pragma once

#include <CL/sycl.hpp>

using Devices = std::vector<sycl::device>;
inline Devices get_devices(const char *target) {
  for (auto &p : sycl::platform::get_platforms()) {
    if (p.get_info<sycl::info::platform::name>().rfind(target) == 0)
      return p.get_devices();
  }

  return Devices();
}

/** Create a list of devices for the applications managed by a MPI processor
 *
 * This should be handled by MPI runtime.
 * @param mpi_rank
 * @param mpi_size
 * @param device_fission if true, tiles will be used as a device.
 */
inline Devices get_devices(int mpi_rank, int mpi_size,
                           bool device_fission = true) {
#if defined(__INTEL_LLVM_COMPILER)
  // only detect GPUs of a Level-zero platform (cannot use more than 1)
  auto gpus = get_devices("Intel(R) OpenCL");
  Devices all_devices;
  if (device_fission) {
    for (auto &g : gpus) {
      try {
        Devices tiles = g.create_sub_devices<
            sycl::info::partition_property::partition_by_affinity_domain>(
            sycl::info::partition_affinity_domain::numa);
        all_devices.insert(all_devices.end(), tiles.begin(), tiles.end());
      } catch (sycl::feature_not_supported) {
        all_devices.push_back(g);
      }
    }
  } else {
    all_devices = gpus;
  }

  if (all_devices.empty())
    return all_devices;

  if (all_devices.size() < mpi_size) {
    return Devices{all_devices[mpi_rank % all_devices.size()]};
  } else {
    int ndev_rank = all_devices.size() / mpi_size;
    // a subset of devices (compact distribution)
    return Devices{all_devices.begin() + mpi_rank * ndev_rank,
                   all_devices.begin() + (mpi_rank + 1) * ndev_rank};
  }
#else
  // cannot create context with multiple devices
  auto gpus = get_devices("CUDA");
  return Devices{gpus[mpi_rank % gpus.size()]};
#endif
}
