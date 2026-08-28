#include "bench.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <mpi.h>
#include <numeric>
#include <string>
#include <sycl/sycl.hpp>
#include <unordered_map>
#include <vector>

const std::string alowed_modes = "(in_order | out_of_order | serial)";

void validate_mode(std::string binname, std::string &mode) {
  if ((mode != "in_order") && (mode != "out_of_order") && (mode != "serial"))
    print_help_and_exit(binname, "Need to specify: " + alowed_modes);
}

[[noreturn]] static void abort_benchmark(int rank, const std::string &message) {
  if (rank == 0)
    std::cerr << "ERROR: " << message << std::endl;
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  std::abort();
}

template <class T>
std::pair<long, std::vector<long>>
bench(std::string mode, std::vector<std::string> &commands,
      std::unordered_map<std::string, size_t> &commands_parameters, bool enable_profiling,
      int n_queues, int n_repetitions, bool verbose) {

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Point-to-point commands pair ranks 0-1, 2-3, and so on.
  const auto point_to_point_count = std::count(commands.begin(), commands.end(), "P");
  if (point_to_point_count > 0 && (size < 2 || size % 2 != 0))
    abort_benchmark(rank, "the MPI benchmark needs an even number of ranks (at least two)");

  if (n_queues == -1)
    n_queues = (mode == "in_order") ? commands.size() : 1;
  if (n_queues <= 0)
    abort_benchmark(rank, "--queues must be greater than zero");

  if (verbose && rank == 0)
    std::cout << "#n_queues used: " << n_queues << std::endl;

  MPI_Comm node;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
  int local_rank;
  MPI_Comm_rank(node, &local_rank);
  MPI_Comm_free(&node);

  const auto gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
  if (gpus.empty())
    abort_benchmark(rank, "no SYCL GPU device is available");

  const sycl::device D = gpus.at(local_rank % gpus.size());
  const sycl::context C(D);

  // SYCL queues are out-of-order unless the in_order property is requested.
  sycl::property_list properties;
  if ((mode == "in_order") && enable_profiling)
    properties = sycl::property_list{sycl::property::queue::in_order{},
                                     sycl::property::queue::enable_profiling{}};
  else if (mode == "in_order")
    properties = sycl::property_list{sycl::property::queue::in_order{}};
  else if (enable_profiling)
    properties = sycl::property_list{sycl::property::queue::enable_profiling{}};

  // Construct each queue separately so it owns a distinct native queue.
  std::vector<sycl::queue> queues;
  queues.reserve(n_queues);
  for (int i = 0; i < n_queues; ++i)
    queues.emplace_back(C, D, properties);

  // Initialize one independent set of buffers for every command.
  std::vector<std::vector<T *>> buffers;
  for (const auto &command : commands) {
    const auto N = commands_parameters["globalsize_" + command];
    if (command == "P" && N > static_cast<size_t>(std::numeric_limits<int>::max()))
      abort_benchmark(rank, "globalsize_P exceeds the MPI count limit");

    std::vector<T *> command_buffers;
    for (const auto memory_type : command) {
      if (memory_type == 'M')
        command_buffers.push_back(static_cast<T *>(calloc(N, sizeof(T))));
      else if (memory_type == 'H')
        command_buffers.push_back(sycl::malloc_host<T>(N, C));
      else if (memory_type == 'S')
        command_buffers.push_back(sycl::malloc_shared<T>(N, D, C));
      else
        command_buffers.push_back(sycl::malloc_device<T>(N, D, C));
    }
    buffers.push_back(std::move(command_buffers));
  }

  long total_time = std::numeric_limits<long>::max();
  std::vector<long> commands_times;
  if (mode == "serial")
    commands_times.assign(commands.size(), std::numeric_limits<long>::max());

  using clock = std::chrono::steady_clock;
  std::vector<MPI_Request> requests;
  requests.reserve(point_to_point_count);
  for (int repetition = 0; repetition < n_repetitions; ++repetition) {
    requests.clear();
    MPI_Barrier(MPI_COMM_WORLD);
    const auto total_start = clock::now();

    for (size_t i = 0; i < commands.size(); ++i) {
      const auto command_start = clock::now();
      auto &queue = queues[i % queues.size()];
      const auto N = commands_parameters["globalsize_" + commands[i]];

      if (commands[i] == "C") {
        T *ptr = buffers[i][0];
        const auto kernel_tripcount = commands_parameters["tripcount_C"];
        queue.parallel_for(sycl::range{N}, [ptr, kernel_tripcount](sycl::id<1> j) {
          ptr[j] = busy_wait(kernel_tripcount, static_cast<T>(j));
        });
      } else if (commands[i] == "P") {
        MPI_Request request = MPI_REQUEST_NULL;
        T *ptr = buffers[i][0];
        const int peer = (rank % 2 == 0) ? rank + 1 : rank - 1;
        const int tag = static_cast<int>(i);
        if (rank % 2 == 0)
          MPI_Isend(ptr, static_cast<int>(N), MPI_FLOAT, peer, tag, MPI_COMM_WORLD, &request);
        else
          MPI_Irecv(ptr, static_cast<int>(N), MPI_FLOAT, peer, tag, MPI_COMM_WORLD, &request);

        if (mode == "serial")
          MPI_Wait(&request, MPI_STATUS_IGNORE);
        else
          requests.push_back(request);
      } else {
        // Copy is source -> destination.
        queue.copy(buffers[i][0], buffers[i][1], N);
      }

      if (mode == "serial") {
        queue.wait();
        const long local_time = std::max(
            1L, std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - command_start)
                    .count());
        long command_time;
        MPI_Allreduce(&local_time, &command_time, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
        commands_times[i] = std::min(commands_times[i], command_time);
      }
    }

    // Enter MPI first so the host can drive communication progress while
    // independent kernels or copies are still running on the GPU.
    if (!requests.empty())
      MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
    for (auto &queue : queues)
      queue.wait();

    const long local_total_time = std::max(
        1L,
        std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - total_start).count());
    long current_total_time;
    MPI_Allreduce(&local_total_time, &current_total_time, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);

    if (verbose && rank == 0)
      std::cout << "#repetition " << repetition << ": " << current_total_time << " us"
                << std::endl;
    total_time = std::min(total_time, current_total_time);
  }

  // Use the sum of the best individual command times as the serial reference.
  if (mode == "serial")
    total_time =
        std::min(total_time, std::accumulate(commands_times.begin(), commands_times.end(), 0L));

  for (const auto &command_buffers : buffers)
    for (const auto ptr : command_buffers)
      (sycl::get_pointer_type(ptr, C) != sycl::usm::alloc::unknown) ? sycl::free(ptr, C)
                                                                    : free(ptr);

  return {total_time, commands_times};
}

template std::pair<long, std::vector<long>>
bench<float>(std::string mode, std::vector<std::string> &commands,
             std::unordered_map<std::string, size_t> &commands_parameters, bool enable_profiling,
             int n_queues, int n_repetitions, bool verbose);
