#include <cassert>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>

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
                       const size_t array_size, int device_id) {

#pragma omp target teams distribute parallel for TARGET_SIMD
  for (int i = 0; i < array_size; i++) {
    VC[i] += VA[i];
  }
}

template <typename T>
inline void SendRecvRing(T *restrict src, T *restrict dest, int mpi_rank,
                         int right, int left, size_t array_size) {

  mpi::status mpi_status;
  const auto mpi_data_type = mpi::get_datatype(T{});

#pragma omp target data use_device_ptr(src, dest)
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

// Allreduce assuming data need to be moved back-and-forth
template <typename T>
inline void AllreduceBase(T *restrict src, T *restrict dest,
                          size_t array_size) {
  const auto mpi_data_type = mpi::get_datatype(T{});

  // Use GPU buffer directly
#pragma omp target data use_device_ptr(src, dest)
  MPI_Allreduce(src, dest, array_size, mpi_data_type, MPI_SUM, MPI_COMM_WORLD);
}

void print_help() {
  std::cout << "Usage: \n";
  std::cout << "options:                                " << '\n';
  std::cout << " -p 2^p elements         default: 25    " << '\n';
}

int main(int argc, char **argv) {
  int mpi_rank{0}, mpi_size{1};

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  size_t array_size = 1 << 25;

  int nsteps = 10;
  int nblocks = 1;
  bool use_allreduce = false;
  int opt;
  while (optind < argc) {
    if ((opt = getopt(argc, argv, "hap:")) != -1) {
      switch (opt) {
      case 'h':
        print_help();
        return 1;
      case 'a': // # of blocks
        use_allreduce = true;
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

  int num_devices = omp_get_num_devices();
  // order devices in round-robin way
  int device_id = mpi_rank % num_devices;

  omp_set_default_device(device_id);

  value_t *VA = static_cast<value_t *>(malloc(sizeof(value_t) * array_size));
  value_t *VB = static_cast<value_t *>(malloc(sizeof(value_t) * array_size));
  value_t *VC = static_cast<value_t *>(malloc(sizeof(value_t) * array_size));

#pragma omp target enter data map(alloc                                        \
                                  : VA [0:array_size], VB [0:array_size],      \
                                    VC [0:array_size])

  {
    const value_t initA = value_t(mpi_rank);
    const value_t initB = value_t(mpi_rank);
    const value_t initC = value_t(0); // result

#pragma omp target teams distribute parallel for TARGET_SIMD
    for (int i = 0; i < array_size; i++) {
      VA[i] = initA;
      VB[i] = initB;
      VC[i] = initC;
    }
  }

  int right_rank = (mpi_rank + 1 + mpi_size) % mpi_size;
  int left_rank = (mpi_rank - 1 + mpi_size) % mpi_size;

  std::chrono::high_resolution_clock::time_point t1, t2;
  t1 = std::chrono::high_resolution_clock::now();

  if (use_allreduce) {
    AllreduceBase(VA, VC, array_size);
  } else {
    Accumulate(VA, VC, array_size, device_id);
    for (int s = 1; s < mpi_size; ++s) {
      SendRecvRing(VA, VB, mpi_rank, right_rank, left_rank, array_size);
      std::swap(VA, VB); // swap src <-> dest
      Accumulate(VA, VC, array_size, device_id);
    }
  }

  t2 = std::chrono::high_resolution_clock::now();

  double t_loc =
      std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
          .count() /
      (nsteps * nblocks);
  double dt{};
  MPI_Allreduce(&t_loc, &dt, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  value_t result = ((mpi_size - 1) * mpi_size) / 2;

// copy to host for checking
#pragma omp target update from(VC [0:array_size])

  std::cout << mpi_rank << " " << VC[0] << std::endl;

  for (int i = 0; i < array_size; ++i)
    assert(std::abs(result - VC[i]) < 1e-6);

  std::cout << "Passed " << mpi_rank << std::endl;

#pragma omp target exit data map(delete                                        \
                                 : VA [0:array_size], VB [0:array_size],       \
                                   VC [0:array_size])

  free(VA);
  free(VB);
  free(VC);

  MPI_Finalize();

  return 0;
}
