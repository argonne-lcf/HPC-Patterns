#include "bench.hpp"

#include <cassert>
#include <chrono>
#include <iostream>
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

template <class T>
std::pair<long, std::vector<long>>
bench(std::string mode, std::vector<std::string> &commands,
      std::unordered_map<std::string, size_t> &commands_parameters, bool enable_profiling,
      int n_queues, int n_repetitions, bool verbose) {

  //   ___
  //    |  ._  o _|_
  //   _|_ | | |  |_
  //
  if (n_queues == -1)
    n_queues = (mode == "in_order") ? commands.size() : 1;

  if (verbose)
    std::cout << "#n_queues used: " << n_queues << std::endl;

  const sycl::device D{sycl::gpu_selector_v};
  const sycl::context C(D);
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

  // Initialize buffers according to the commands
  std::vector<std::vector<T *>> buffers;
  for (auto &command : commands) {
    const auto N = commands_parameters["globalsize_" + command];
    std::vector<T *> buffer;
    for (auto c : command) {
      if (c == 'C')
        buffer.push_back(sycl::malloc_device<T>(N, D, C));
      else if (c == 'M')
        buffer.push_back(static_cast<T *>(calloc(N, sizeof(T))));
      else if (c == 'D')
        buffer.push_back(sycl::malloc_device<T>(N, D, C));
      else if (c == 'H')
        buffer.push_back(sycl::malloc_host<T>(N, C));
      else if (c == 'S')
        buffer.push_back(sycl::malloc_shared<T>(N, D, C));
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
    auto s0 = std::chrono::high_resolution_clock::now();
    // Run all commands
    for (int i = 0; i < commands.size(); i++) {
      const auto s = std::chrono::high_resolution_clock::now();
      sycl::queue Q = Qs[i % n_queues];
      const auto N = commands_parameters["globalsize_" + commands[i]];

      if (commands[i] == "C") {
        T *ptr = buffers[i][0];
        const auto kernel_tripcount = commands_parameters["tripcount_C"];
        Q.parallel_for(sycl::range{N}, [ptr, kernel_tripcount](sycl::id<1> j) {
          ptr[j] = busy_wait(kernel_tripcount, (T)j);
        });
      } else {
        // Copy is src -> dest
        Q.copy(buffers[i][0], buffers[i][1], N);
      }

      if (mode == "serial") {
        Q.wait();
        const auto e = std::chrono::high_resolution_clock::now();
        const auto curent_kernel_time =
            std::chrono::duration_cast<std::chrono::microseconds>(e - s).count();
        commands_times[i] = std::min(commands_times[i], curent_kernel_time);
      }
    }
    // Sync all queues
    for (auto &Q : Qs)
      Q.wait();
    // Save time
    const auto e0 = std::chrono::high_resolution_clock::now();
    const auto curent_total_time =
        std::chrono::duration_cast<std::chrono::microseconds>(e0 - s0).count();
    if (verbose)
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
