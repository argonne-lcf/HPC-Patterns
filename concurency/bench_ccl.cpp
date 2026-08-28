#include "bench.hpp"

#include <chrono>
#include <iostream>
#include <mpi.h>
#include <numeric>
#include <oneapi/ccl.hpp>
#include <string>
#include <sycl/sycl.hpp>
#include <unordered_map>
#include <vector>

const std::string alowed_modes = "(in_order | out_of_order | serial)";

void validate_mode(std::string binname, std::string &mode) {
  if ((mode != "in_order") && (mode != "out_of_order") && (mode != "serial"))
    print_help_and_exit(binname, "Need to specify: " + alowed_modes);
}

// Rank 0 publishes the KVS address, MPI carries it to the others.
static ccl::shared_ptr_class<ccl::kvs> bootstrap_kvs(int rank) {
  ccl::kvs::address_type addr;
  if (rank == 0) {
    auto kvs = ccl::create_main_kvs();
    addr = kvs->get_address();
    MPI_Bcast(addr.data(), addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    return kvs;
  }
  MPI_Bcast(addr.data(), addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
  return ccl::create_kvs(addr);
}

// The communicator outlives every bench() call and must be torn down before
// MPI_Finalize.
static ccl::communicator *comm_ptr = nullptr;
void ccl_teardown() {
  delete comm_ptr;
  comm_ptr = nullptr;
}

static unsigned long now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::high_resolution_clock::now().time_since_epoch())
      .count();
}

template <class T>
std::pair<long, std::vector<long>>
bench(std::string mode, std::vector<std::string> &commands,
      std::unordered_map<std::string, size_t> &commands_parameters, bool enable_profiling,
      int n_queues, int n_repetitions, bool verbose) {

  //   ___
  //    |  ._  o _|_
  //   _|_ | | |  |_
  //
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (n_queues == -1)
    n_queues = (mode == "in_order") ? commands.size() : 1;

  if (verbose && rank == 0)
    std::cout << "#n_queues used: " << n_queues << std::endl;

  MPI_Comm node;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
  int local_rank;
  MPI_Comm_rank(node, &local_rank);
  MPI_Comm_free(&node);

  // The device, context and communicator live for the whole run: bench() is
  // called many times, and both the KVS server and the CCL context cannot be
  // rebuilt per call.
  const auto gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
  static const sycl::device D = gpus.at(local_rank % gpus.size());
  static const sycl::context C(D);

  // By default SYCL queue are out-of-order
  sycl::property_list pl;
  if ((mode == "in_order") && enable_profiling)
    pl = sycl::property_list{sycl::property::queue::in_order{},
                             sycl::property::queue::enable_profiling{}};
  else if (mode == "in_order")
    pl = sycl::property_list{sycl::property::queue::in_order{}};
  else if (enable_profiling)
    pl = sycl::property_list{sycl::property::queue::enable_profiling{}};

  // List of queues!
  // One shoud not use 'Qs(n_queues, sycl::queue(C, D, pl))'!
  // This copy queue and hence sharing the native objects.
  std::vector<sycl::queue> Qs;
  for (size_t i = 0; i < n_queues; i++)
    Qs.push_back(sycl::queue(C, D, pl));

  // oneCCL rides on the same queues
  std::vector<ccl::stream> streams;
  for (auto &Q : Qs)
    streams.push_back(ccl::create_stream(Q));

  static auto ccl_device = ccl::create_device(D);
  static auto ccl_context = ccl::create_context(C);
  if (comm_ptr == nullptr)
    comm_ptr = new ccl::communicator(
        ccl::create_communicator(size, rank, ccl_device, ccl_context, bootstrap_kvs(rank)));
  ccl::communicator &comm = *comm_ptr;

  // Initialize buffers according to the commands
  std::vector<std::vector<T *>> buffers;
  for (auto &command : commands) {
    const auto N = commands_parameters["globalsize_" + command];
    std::vector<T *> buffer;
    for (auto c : command) {
      if (c == 'M')
        buffer.push_back(static_cast<T *>(calloc(N, sizeof(T))));
      else if (c == 'H')
        buffer.push_back(sycl::malloc_host<T>(N, C));
      else if (c == 'S')
        buffer.push_back(sycl::malloc_shared<T>(N, D, C));
      else
        buffer.push_back(sycl::malloc_device<T>(N, D, C));
    }
    buffers.push_back(buffer);
  }

  long total_time = std::numeric_limits<long>::max();
  std::vector<long> commands_times;
  if (mode == "serial")
    std::fill_n(std::back_inserter(commands_times), commands.size(),
                std::numeric_limits<long>::max());

  //    _
  //   |_)  _  ._   _ |_
  //   |_) (/_ | | (_ | |
  //
  for (int r = 0; r < n_repetitions; r++) {
    MPI_Barrier(MPI_COMM_WORLD);
    // Save start and end
    const unsigned long l_start0 = now_ns();
    // Run all commands
    for (int i = 0; i < commands.size(); i++) {
      const unsigned long l_start = now_ns();
      sycl::queue Q = Qs[i % n_queues];
      const auto N = commands_parameters["globalsize_" + commands[i]];

      if (commands[i] == "C") {
        T *ptr = buffers[i][0];
        const auto kernel_tripcount = commands_parameters["tripcount_C"];
        Q.parallel_for(sycl::range{N}, [ptr, kernel_tripcount](sycl::id<1> j) {
          ptr[j] = busy_wait(kernel_tripcount, (T)j);
        });
      } else if (commands[i] == "P") {
        // Even ranks send, odd ranks receive
        T *ptr = buffers[i][0];
        const int peer = (rank % 2 == 0) ? rank + 1 : rank - 1;
        if (rank % 2 == 0)
          ccl::send(ptr, N, ccl::datatype::float32, peer, comm, streams[i % n_queues]);
        else
          ccl::recv(ptr, N, ccl::datatype::float32, peer, comm, streams[i % n_queues]);
      } else {
        // Copy is src -> dest
        Q.copy(buffers[i][0], buffers[i][1], N);
      }

      if (mode == "serial") {
        Q.wait();
        const unsigned long l_end = now_ns();

        unsigned long start, end;
        MPI_Allreduce(&l_start, &start, 1, MPI_UNSIGNED_LONG, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&l_end, &end, 1, MPI_UNSIGNED_LONG, MPI_MAX, MPI_COMM_WORLD);

        const long curent_kernel_time = (end - start) / 1000;
        commands_times[i] = std::min(commands_times[i], curent_kernel_time);
      }
    }
    // Sync all queues
    for (auto &Q : Qs)
      Q.wait();
    // Save time
    const unsigned long l_end0 = now_ns();

    unsigned long start0, end0;
    MPI_Allreduce(&l_start0, &start0, 1, MPI_UNSIGNED_LONG, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&l_end0, &end0, 1, MPI_UNSIGNED_LONG, MPI_MAX, MPI_COMM_WORLD);

    const long curent_total_time = (end0 - start0) / 1000;
    if (verbose && rank == 0)
      std::cout << "#repetition " << r << ": " << curent_total_time << " us" << std::endl;
    total_time = std::min(total_time, curent_total_time);
  }

  // Assume the "best theoritical" serial
  if (mode == "serial")
    total_time =
        std::min(total_time, std::accumulate(commands_times.begin(), commands_times.end(), 0L));

  //    _
  //   /  |  _   _. ._      ._
  //   \_ | (/_ (_| | | |_| |_)
  //                        |
  for (const auto &buffer : buffers)
    for (const auto &ptr : buffer)
      // Shorter than to remember commands types...
      (sycl::get_pointer_type(ptr, C) != sycl::usm::alloc::unknown) ? sycl::free(ptr, C)
                                                                    : free(ptr);

  return {total_time, commands_times};
}

template std::pair<long, std::vector<long>>
bench<float>(std::string mode, std::vector<std::string> &commands,
             std::unordered_map<std::string, size_t> &commands_parameters, bool enable_profiling,
             int n_queues, int n_repetitions, bool verbose);
