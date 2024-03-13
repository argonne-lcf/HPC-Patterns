#include "bench.hpp"

#include <cassert>
#include <chrono>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <string>
#include <unordered_map>
#include <vector>

const std::string alowed_modes = "(nowait | host_threads | serial)";

void validate_mode(std::string binname, std::string &mode) {
  if ((mode != "nowait") && (mode != "host_threads") && (mode != "serial"))
    print_help_and_exit(binname, "Need to specify: " + alowed_modes);
}

// No metadirective in most of the compiler so...
//  UGLY PRAGMA to the rescue!
template <class T, bool isHostThreads>
std::pair<long, std::vector<long>>
bench2(std::string mode, std::vector<std::string> &commands,
       std::unordered_map<std::string, size_t> &commands_parameters, bool enable_profiling,
       int n_queues, int n_repetitions, bool verbose) {

  //   ___
  //    |  ._  o _|_
  //   _|_ | | |  |_
  //
  // Initialize buffers according to the commands
  if (n_queues == -1)
    n_queues = (mode == "host_threads") ? commands.size() : 1;

  if (verbose)
    std::cout << "#n_host_threads used: " << n_queues << std::endl;

  std::vector<T *> buffers;
  for (auto &command : commands) {
    const auto N = commands_parameters["globalsize_" + command];
    T *ptr;
    if (command.find("H") != std::string::npos) {
      ptr = static_cast<T *>(omp_target_alloc_host(N * sizeof(T), omp_get_default_device()));
    } else {
      ptr = static_cast<T *>(calloc(N, sizeof(T)));
    }
    assert(ptr && "Wrong Allocation");
#pragma omp target enter data map(alloc : ptr[:N])
    buffers.push_back(ptr);
  }

  long total_time = std::numeric_limits<long>::max();
  std::vector<long> commands_times;
  if (mode == "serial") {
    std::fill_n(std::back_inserter(commands_times), commands.size(),
                std::numeric_limits<long>::max());
    omp_set_num_threads(1);
  } else if (mode == "host_threads")
    omp_set_num_threads(n_queues);

  //    _
  //   |_)  _  ._   _ |_
  //   |_) (/_ | | (_ | |
  //
  for (int r = 0; r < n_repetitions; r++) {
    auto s0 = std::chrono::high_resolution_clock::now();
#pragma omp metadirective when(user = {condition(isHostThreads)} : parallel for) otherwize()
    for (int i = 0; i < commands.size(); i++) {
      const auto s = std::chrono::high_resolution_clock::now();
      const auto N = commands_parameters["globalsize_" + commands[i]];
      T *ptr = buffers[i];
      if (commands[i] == "C") {
        const auto kernel_tripcount = commands_parameters["tripcount_C"];
#pragma omp metadirective when(                                                                    \
        user = {condition(isHostThreads)} : target teams distribute parallel for)                  \
        otherwize(target teams distribute parallel for nowait)
        for (int j = 0; j < N; j++)
          ptr[j] = busy_wait(kernel_tripcount, (T)j);
      } else if (commands[i] == "DM" or commands[i] == "DH") {
#pragma omp metadirective when(user = {condition(isHostThreads)} : target update from(ptr[:N]))  \
    otherwize(target update from(ptr[:N]) nowait)
      } else if (commands[i] == "MD" or commands[i] == "HD") {
#pragma omp metadirective when(user = {condition(isHostThreads)} : target update to(ptr[:N]))    \
    otherwize(target update to(ptr[:N]) nowait)
      }

      if (mode == "serial") {
#pragma omp metadirective when(user = {condition(!isHostThreads)} : taskwait) otherwize()
        const auto e = std::chrono::high_resolution_clock::now();
        const auto curent_kernel_time =
            std::chrono::duration_cast<std::chrono::microseconds>(e - s).count();
        commands_times[i] = std::min(commands_times[i], curent_kernel_time);
      }
    }
#pragma omp metadirective when(user = {condition(!isHostThreads)} : taskwait) otherwize()
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
  for (int i = 0; i < buffers.size(); i++) {
    const auto N = commands_parameters["globalsize_" + commands[i]];
    auto *ptr = buffers[i];
#pragma omp target exit data map(delete : ptr[:N])
    if (commands[i].find("H") != std::string::npos)
      omp_target_free(ptr, omp_get_default_device());
    else
      free(ptr);
  }
  return {total_time, commands_times};
}

template <class T>
std::pair<long, std::vector<long>>
bench(std::string mode, std::vector<std::string> &commands,
      std::unordered_map<std::string, size_t> &commands_parameters, bool enable_profiling,
      int n_queues, int n_repetitions, bool verbose) {

  if (mode == "host_threads")
    return bench2<T, true>(mode, commands, commands_parameters, enable_profiling, n_queues,
                           n_repetitions, verbose);
  else
    return bench2<T, false>(mode, commands, commands_parameters, enable_profiling, n_queues,
                            n_repetitions, verbose);
}

template std::pair<long, std::vector<long>>
bench<float>(std::string mode, std::vector<std::string> &commands,
             std::unordered_map<std::string, size_t> &commands_parameters, bool enable_profiling,
             int n_queues, int n_repetitions, bool verbose);
