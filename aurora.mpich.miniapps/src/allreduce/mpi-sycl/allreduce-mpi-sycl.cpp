/** triad + send/recv between the pairs
 *
 * Even ranks send to the right rank
 */
#include <chrono>
#include <complex>
#include <utility>
#include <vector>

#include <CL/sycl.hpp>

#include <getopt.h>

#include <mpi.h>

#include "devices.hpp"
#include "mpi_datatype.hpp"

#ifndef ALIGNMENT
#define ALIGNMENT (128) // Bigger will fail on CPU device?
#endif
#ifndef APP_DATA_TYPE
#define APP_DATA_TYPE float
#endif

template <typename T>
inline sycl::event Accumulate(const T *restrict VA, T *restrict VC,
                              size_t array_size, sycl::queue &aq) {
  return aq.parallel_for({array_size},
                         [=](sycl::id<1> wiID) { VC[wiID] += VA[wiID]; });
}

template <typename T>
inline void Initialize(T *VA, T *VB, T *VC, size_t array_size, T a, T b, T c,
                       sycl::queue &aq) {
  aq.parallel_for({array_size}, [=](sycl::id<1> wiID) {
      VA[wiID] = a;
      VB[wiID] = b;
      VC[wiID] = c;
    }).wait();
}

template <typename T>
inline void SendRecvRing(T *restrict src, T *restrict dest, int mpi_rank,
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
  std::cout << " -H                   sycl::usm::alloc::host  " << '\n';
  std::cout << " -D                   sycl::usm::alloc::device" << '\n';
  std::cout << " -S                   sycl::usm::alloc::shared (default)"
            << '\n';
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

  if (mpi_size < 4 || (mpi_size % 2) != 0) {
    error("Set MPI ranks to an even integer >= 4", true);
  }

  size_t array_size = 1 << 25;
  int nsteps = 10;
  int nblocks = 10;
  int nqueues = 1;
  bool use_allreduce = false;
  sycl::usm::alloc allockind = sycl::usm::alloc::shared;

  int opt;
  while (optind < argc) {
    if ((opt = getopt(argc, argv, "haHDSp:")) != -1) {
      switch (opt) {
      case 'h':
        print_help();
        return 1;
      case 'H':
        allockind = sycl::usm::alloc::host;
        break;
      case 'D':
        allockind = sycl::usm::alloc::device;
        break;
      case 'S':
        allockind = sycl::usm::alloc::shared;
        break;
      case 'a':
        use_allreduce = true;
        break;
      case 'p': // 2^p
        int x = atoi(optarg);
        array_size = 1 << x;
        break;
      }
    }
  }

  using value_t = APP_DATA_TYPE;

  auto devices = get_devices(mpi_rank, mpi_size, true);

  if (devices.empty()) {
    std::cerr << "No devices\n";
    MPI_Finalize();
    exit(1);
  }

  int num_devices = devices.size();
  int device_id = 0;

  //distribute ranks to devices in round-robin way
  device_id = mpi_rank % num_devices;
  // create a context containing all the devices
  sycl::context mContext(devices);
  // create a queue
  // sycl::queue mQueue{mContext, devices[device_id]};
  sycl::queue mQueue{devices[device_id]};

  value_t *VA =
      sycl::aligned_alloc<value_t>(ALIGNMENT, array_size, mQueue, allockind);
  value_t *VB =
      sycl::aligned_alloc<value_t>(ALIGNMENT, array_size, mQueue, allockind);
  value_t *VC =
      sycl::aligned_alloc<value_t>(ALIGNMENT, array_size, mQueue, allockind);

  if (VA == nullptr || VB == nullptr || VC == nullptr)
    error("Alloc failed");

  Initialize(VA, VB, VC, array_size, value_t(mpi_rank), value_t(mpi_rank),
             value_t(0), mQueue);

  int right_rank = (mpi_rank + 1 + mpi_size) % mpi_size;
  int left_rank = (mpi_rank - 1 + mpi_size) % mpi_size;

  std::chrono::high_resolution_clock::time_point t1, t2;
  t1 = std::chrono::high_resolution_clock::now();

  if (use_allreduce) {
    AllreduceColl(VA, VC, array_size);
  } else {
    Accumulate(VA, VC, array_size, mQueue).wait();
    for (int s = 1; s < mpi_size; ++s) {
      SendRecvRing(VA, VB, mpi_rank, right_rank, left_rank, array_size);
      std::swap(VA, VB); // swap src <-> dest
      Accumulate(VA, VC, array_size, mQueue).wait();
    }
  }
  t2 = std::chrono::high_resolution_clock::now();

  auto dt = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
                .count();

  double t_max = 0.0;
  MPI_Allreduce(&dt, &t_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  dt = t_max;

  value_t result = ((mpi_size - 1) * mpi_size) / 2;

  if (allockind == sycl::usm::alloc::device) {
    value_t *temp = sycl::aligned_alloc<value_t>(ALIGNMENT, array_size, mQueue,
                                                 sycl::usm::alloc::shared);
    mQueue.memcpy(temp, VC, array_size * sizeof(value_t)).wait();
    for (int i = 0; i < array_size; ++i)
      assert(std::abs(result - temp[i]) < 1e-6);
    free(temp, mQueue.get_context());
  } else {
    for (int i = 0; i < array_size; ++i)
      assert(std::abs(result - VC[i]) < 1e-6);
  }

  std::cout << "Passed " << mpi_rank << std::endl;

  MPI_Finalize();

  free(VC, mQueue.get_context());
  free(VB, mQueue.get_context());
  free(VA, mQueue.get_context());

  return 0;
}
