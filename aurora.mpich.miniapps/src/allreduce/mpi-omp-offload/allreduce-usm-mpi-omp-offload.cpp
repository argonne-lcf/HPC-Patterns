#include <cassert>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include <omp.h>

#include <getopt.h>

#include "mpi.h"

#include "mpi_datatype.hpp"

#ifndef ALIGNMENT
#define ALIGNMENT (2 * 1024 * 1024) // 2MB
#endif

#define TARGET_SIMD simd

template <typename T>
inline void Accumulate(const T *restrict VA, T *restrict VC,
                       const size_t array_size, int dev_id) {

#pragma omp target is_device_ptr(VA, VC)
#pragma omp teams distribute parallel for TARGET_SIMD
  for (int i = 0; i < array_size; i++) {
    VC[i] += VA[i];
  }
}

template <typename T>
inline void SendRecvRing(T *src, T *dest, int dev_id, int host_id, int mpi_rank,
                         int right, int left, size_t array_size) {

  mpi::status mpi_status;
  const auto mpi_data_type = mpi::get_datatype(T{});

  if (mpi_rank % 2) {
    MPI_Send(src, array_size, mpi_data_type, right, 0, MPI_COMM_WORLD);
    MPI_Recv(dest, array_size, mpi_data_type, left, 1, MPI_COMM_WORLD,
             &mpi_status);
  } else {
    MPI_Recv(dest, array_size, mpi_data_type, left, 0, MPI_COMM_WORLD,
             &mpi_status);
    MPI_Send(src, array_size, mpi_data_type, right, 1, MPI_COMM_WORLD);
  }
}

template <typename T>
inline void AllreduceColl(T *restrict src, T *restrict dest,
                          size_t array_size) {
  const auto mpi_data_type = mpi::get_datatype(T{});

  MPI_Allreduce(src, dest, array_size, mpi_data_type, MPI_SUM, MPI_COMM_WORLD);
}

void print_help() {
  std::cout << "Usage: \n";
  std::cout << "options:                                     " << '\n';
  std::cout << " -p 2^p elements                  default: 25" << '\n';
  std::cout << " -H                   omp_target_alloc_host  " << '\n';
  std::cout << " -D                   omp_target_alloc_device" << '\n';
  std::cout << " -S                   omp_target_alloc_shared" << '\n';
  std::cout << "Default allocator:    omp_target_alloc       " << '\n';
}

void error(std::string message, bool rank_zero_only = false) {
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  if (!rank_zero_only || mpi_rank == 0)
    std::cerr << "Error: " << message << "\n";
  MPI_Finalize();
  exit(1);
}

int main(int argc, char **argv) {
  int mpi_rank{0}, mpi_size{1};

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  size_t array_size = 1 << 25;

  int nsteps = 10;
  int nblocks = 1;
  int opt;
  bool use_allreduce = false;

  enum { alloc_target = 0, alloc_host, alloc_shared, alloc_device };
  // default: omp_alloc_target
  int allockind = alloc_target;

  while (optind < argc) {
    if ((opt = getopt(argc, argv, "haHDSp:")) != -1) {
      switch (opt) {
      case 'h':
        print_help();
        return 1;
      case 'a':
        use_allreduce = true;
        break;
      case 'H':
        allockind = alloc_host;
        break;
      case 'D':
        allockind = alloc_device;
        break;
      case 'S':
        allockind = alloc_shared;
        break;
      case 'p': // 2^p
        int x = atoi(optarg);
        array_size = 1 << x;
        break;
      }
    }
  }

  bool use_subdevices = false;
  if (const char *env_p = std::getenv("LIBOMPTARGET_DEVICES")) {
    use_subdevices = !strcmp(env_p, "subdevice");
  }

  using value_t = APP_DATA_TYPE;

  int host_id = omp_get_initial_device();

  int num_devices = omp_get_num_devices();

  if (num_devices == 0) {
    error("No devices", true);
  }

  // order devices in round-robin way
  int dev_id = mpi_rank % num_devices;

  omp_set_default_device(dev_id);

  value_t *VA, *VB, *VC;
  size_t bytes = sizeof(value_t) * array_size;

#if defined(__INTEL_CLANG_COMPILER)
  if (allockind == alloc_host) {
    VA = static_cast<value_t *>(omp_target_alloc_host(bytes, dev_id));
    VB = static_cast<value_t *>(omp_target_alloc_host(bytes, dev_id));
    VC = static_cast<value_t *>(omp_target_alloc_host(bytes, dev_id));
  } else if (allockind == alloc_shared) {
    VA = static_cast<value_t *>(omp_target_alloc_shared(bytes, dev_id));
    VB = static_cast<value_t *>(omp_target_alloc_shared(bytes, dev_id));
    VC = static_cast<value_t *>(omp_target_alloc_shared(bytes, dev_id));
  } else if (allockind == alloc_device) {
    VA = static_cast<value_t *>(omp_target_alloc_device(bytes, dev_id));
    VB = static_cast<value_t *>(omp_target_alloc_device(bytes, dev_id));
    VC = static_cast<value_t *>(omp_target_alloc_device(bytes, dev_id));
  } else
#endif
  {
    VA = static_cast<value_t *>(omp_target_alloc(bytes, dev_id));
    VB = static_cast<value_t *>(omp_target_alloc(bytes, dev_id));
    VC = static_cast<value_t *>(omp_target_alloc(bytes, dev_id));
  }

  if (VA == nullptr || VB == nullptr || VC == nullptr)
    error("Alloc failed");

  {
    const value_t initA = value_t(mpi_rank);
    const value_t initC = value_t(0); // result

#pragma omp target is_device_ptr(VA, VC)
#pragma omp teams distribute parallel for TARGET_SIMD
    for (int i = 0; i < array_size; i++) {
      VA[i] = initA;
      VB[i] = initA;
      VC[i] = initC;
    }
  }

  int right_rank = (mpi_rank + 1 + mpi_size) % mpi_size;
  int left_rank = (mpi_rank - 1 + mpi_size) % mpi_size;

  std::chrono::high_resolution_clock::time_point t1, t2;
  t1 = std::chrono::high_resolution_clock::now();

  if (use_allreduce) {
    AllreduceColl(VA, VC, array_size);
  } else {
    Accumulate(VA, VC, array_size, dev_id);
    for (int s = 1; s < mpi_size; ++s) {
      SendRecvRing(VA, VB, dev_id, host_id, mpi_rank, right_rank, left_rank,
                   array_size);
      std::swap(VA, VB);
      Accumulate(VA, VC, array_size, dev_id);
    }
  }
  t2 = std::chrono::high_resolution_clock::now();

  value_t *bufferA = (value_t *)malloc(array_size * sizeof(value_t));
  omp_target_memcpy(bufferA, VC, array_size * sizeof(value_t), 0, 0, host_id,
                    dev_id);

  double t_loc =
      std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
          .count() /
      (nsteps * nblocks);
  double dt{};
  MPI_Allreduce(&t_loc, &dt, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  value_t result = ((mpi_size - 1) * mpi_size) / 2;

  for (int i = 0; i < array_size; ++i)
    assert(std::abs(result - bufferA[i]) < 1e-6);

  std::cout << "Passed " << mpi_rank << std::endl;

  omp_target_free(VC, dev_id);
  omp_target_free(VB, dev_id);
  omp_target_free(VA, dev_id);

  MPI_Finalize();

  return 0;
}
