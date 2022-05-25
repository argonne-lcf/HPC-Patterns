#define MAD_4(x, y)                                                                                                                                                                                    \
  x = y * x + y;                                                                                                                                                                                       \
  y = x * y + x;                                                                                                                                                                                       \
  x = y * x + y;                                                                                                                                                                                       \
  y = x * y + x;
#define MAD_16(x, y)                                                                                                                                                                                   \
  MAD_4(x, y);                                                                                                                                                                                         \
  MAD_4(x, y);                                                                                                                                                                                         \
  MAD_4(x, y);                                                                                                                                                                                         \
  MAD_4(x, y);
#define MAD_64(x, y)                                                                                                                                                                                   \
  MAD_16(x, y);                                                                                                                                                                                        \
  MAD_16(x, y);                                                                                                                                                                                        \
  MAD_16(x, y);                                                                                                                                                                                        \
  MAD_16(x, y);

#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>
#include <algorithm>
#include <cassert>

#include <omp.h>
template <class T> T busy_wait(size_t N, T i) {
  T x = 1.3f;
  T y = i;
  for (size_t j = 0; j < N; j++) {
    MAD_64(x, y);
  }
  return y;
}

std::string sanitize_command(std::string command) {
  std::string command_sanitized(command);
  command_sanitized.erase(std::remove(command_sanitized.begin(), command_sanitized.end(), '2'), command_sanitized.end());
  return command_sanitized;
}

// No metadirective in most of the compiler so...
//  UGLY PRAGMA to the rescue!
template <class T>
std::pair<long, std::vector<long>> bench(std::string mode, std::vector<std::string> &commands, std::unordered_map<std::string, size_t> &commands_parameters, bool enable_profiling, int n_queues,
                                         int n_repetitions) {

  //   ___
  //    |  ._  o _|_
  //   _|_ | | |  |_
  //
  // Initialize buffers according to the commands
  std::vector<T *> buffers;
  for (auto &command : commands) {
    const auto N = commands_parameters["globalsize_" + command];
    T* ptr;
    if (command.find("H") != std::string::npos) {
        ptr = static_cast<T *>(omp_target_alloc_host(N*sizeof(T), omp_get_default_device() ));
    } else {
        ptr = static_cast<T *>(calloc(N, sizeof(T)));
    }
    assert(ptr && "Wrong Allocation");
    #pragma omp target enter data map(alloc: ptr[:N])
    buffers.push_back(ptr);
  }

  long total_time = std::numeric_limits<long>::max();
  std::vector<long> commands_times;
  if (mode == "serial") {
    std::fill_n(std::back_inserter(commands_times), commands.size(), std::numeric_limits<long>::max());
    omp_set_num_threads(1);
  } else if (mode == "host_threads")
    omp_set_num_threads(n_queues);


  //    _
  //   |_)  _  ._   _ |_
  //   |_) (/_ | | (_ | |
  //
  for (int r = 0; r < n_repetitions; r++) {
    auto s0 = std::chrono::high_resolution_clock::now();
#ifdef HOST_THREADS
    #pragma omp parallel for
#endif
    for (int i = 0; i < commands.size(); i++) {
      const auto s = std::chrono::high_resolution_clock::now();
      const auto N = commands_parameters["globalsize_" + commands[i]];
      T *ptr = buffers[i];
      if (commands[i] == "C") {
        const auto kernel_tripcount = commands_parameters["tripcount_C"];
#ifdef NOWAIT
    #pragma omp target teams distribute parallel for nowait
#else
    #pragma omp target teams distribute parallel for
#endif
        for (int j=0; j < N; j++)
            ptr[j] = busy_wait(kernel_tripcount, (T)j);
      } else if (commands[i] == "D2M" or commands[i] == "D2H") {
#ifdef NOWAIT
    #pragma omp target update from(ptr[:N]) nowait
#else
    #pragma omp target update from(ptr[:N])
#endif
      } else if (commands[i] == "M2D" or commands[i] == "H2D") {
#ifdef NOWAIT
    #pragma omp target update to(ptr[:N]) nowait
#else
    #pragma omp target update to(ptr[:N])
#endif
     }

     if (mode == "serial") {
#ifdef NOWAIT
    #pragma omp taskwait
#endif
        const auto e = std::chrono::high_resolution_clock::now();
        const auto curent_kernel_time = std::chrono::duration_cast<std::chrono::microseconds>(e - s).count();
        commands_times[i] = std::min(commands_times[i], curent_kernel_time);
      }
    }
#ifdef NOWAIT
    #pragma omp taskwait
#endif
    // Save time
    const auto e0 = std::chrono::high_resolution_clock::now();
    const auto curent_total_time = std::chrono::duration_cast<std::chrono::microseconds>(e0 - s0).count();
#ifdef LOG_VERBOSE
    if (mode != "serial")
        std::cout << "#repetition "  << r << ": " << curent_total_time << " us" << std::endl;
#endif
    total_time = std::min(total_time, curent_total_time);
  }
  // Assume the "best theoritical" serial
  if (mode == "serial")
    total_time = std::min(total_time, std::accumulate(commands_times.begin(), commands_times.end(), 0L));


  //    _
  //   /  |  _   _. ._      ._
  //   \_ | (/_ (_| | | |_| |_)
  //                        |
  //for (auto &ptr : buffers) {
  for (int i=0; i < buffers.size(); i++) {
    const auto N = commands_parameters["globalsize_" + commands[i]];
    auto* ptr = buffers[i];
    #pragma omp target exit data map(delete: ptr[:N])
    if (commands[i].find("H") != std::string::npos)
        omp_target_free(ptr, omp_get_default_device());
    else
        free(ptr);
  }
  return std::make_pair(total_time, commands_times);
}

void print_help_and_exit(std::string binname, std::string msg) {
  if (!msg.empty())
    std::cout << "ERROR: " << msg << std::endl;
  std::string help = "Usage: " + binname +
                     " (host_threads | nowait | serial)\n"
                     "                [--tripcount_C <tripcount>]\n"
                     "                [--globalsize_{C,A2B} <global_size>]\n"
                     "                [--queues <n_queues>]\n"
                     "                [--repetitions <n_repetions>]\n"
                     "                COMMAND...\n"
                     "\n"
                     "Options:\n"
                     "--tripcount_C               [default: -1]. Each kernel work-item will perform 64*C_tripcount FMA\n"
                     "                              '-1' will auto-tune this parameter so each commands take similar time\n"
                     "--globalsize_{C,A2B}        [default: -1]. Work-group size of the commands\n"
                     "                             '-1' will auto-tune this parameter so each commands take similar time\n"
                     "--globalsize_default_memory [default: -1].  Size of the memory buffer before auto-tuning \n"
                     "                             '-1' mean maximun possible size\n"
                     "--queues                    [default: -1]. Number of queues used to run COMMANDS\n"
                     "                              '-1' mean automatic selection:\n"
                     "                                - if `host_threads`, one queue per COMMAND\n"
                     "                                - else one queue\n"
                     "--repetitions               [default: 10]. Number of repetions for each measuremnts\n"
                     "COMMAND                     [possible values: C, A2B]\n"
                     "                              C:  Compute kernel\n"
                     "                              A2B: Memcopy from A to B\n"
                     "                              Where A,B can be:\n"
                     "                                M: Malloc allocated memory\n"
                     "                                D: sycl::device allocated memory\n"
                     "                                H: sycl::host allocated memory\n"
                     "                                S: sycl::shared allocated memory\n";
  std::cout << help << std::endl;
  std::exit(1);
}

size_t get_default_command_parameter(std::string command, size_t num_command, std::unordered_map<std::string, long> &commands) {
  if (command.rfind("globalsize_C", 0) == 0)
    return 1;
  if (command.rfind("tripcount_C", 0) == 0)
    return 40000;
  if (command.rfind("globalsize_", 0) == 0) {
    if (commands["globalsize_default_memory"] != -1)
      return commands["globalsize_default_memory"];
    const auto max_mem_alloc_command = 1e9; //~One gigabyte
    return max_mem_alloc_command / sizeof(float);
  }
  return 0;
}

std::string commands_to_parameters_tunned(std::string command) {
  if (command == "C")
    return "tripcount_C";
  return "globalsize_" + command;
}

int main(int argc, char *argv[]) {
  //    _                       _
  //   |_) _. ._ _ o ._   _    /  |     /\  ._ _      ._ _   _  ._ _|_  _
  //   |  (_| | _> | | | (_|   \_ |_   /--\ | (_| |_| | | | (/_ | | |_ _>
  //                      _|                   _|
  //
  std::unordered_map<std::string, long> commands_parameters_cli = {{"globalsize_C", -1}, {"tripcount_C", -1}, {"globalsize_default_memory", -1}};
  bool enable_profiling = false;
  int n_queues = -1;
  int n_repetitions = 10;

  std::vector<std::string> argl(argv + 1, argv + argc);
  if (argl.empty())
    print_help_and_exit(argv[0], "");

  std::string mode{argl[0]};
  if ((mode != "nowait") && (mode != "host_threads") && (mode != "serial"))
    print_help_and_exit(argv[0], "Need to specify 'host_threads', 'nowait', 'serial', option");

  std::vector<std::string> commands;
  // I'm just an old C programmer trying to do some C++
  for (int i = 1; i < argl.size(); i++) {
    std::string s{argl[i]};
    if (s == "--queues") {
      i++;
      if (i < argl.size()) {
        n_queues = std::stoi(argl[i]);
      } else {
        print_help_and_exit(argv[0], "Need to specify an value for '--queues'");
      }
    } else if (s == "--repetitionss") {
      i++;
      if (i < argl.size()) {
        n_repetitions = std::stoi(argl[i]);
      } else {
        print_help_and_exit(argv[0], "Need to specify an value for '--queues'");
      }
    } else if ((s.rfind("--tripcount_") == 0) || (s.rfind("--globalsize_", 0) == 0)) {
      i++;
      if (i < argl.size()) {
        commands_parameters_cli[s.substr(2)] = std::stol(argl[i]);
      } else {
        print_help_and_exit(argv[0], "Need to specify an value for " + s);
      }
    } else if (s.rfind("-", 0) == 0) {
      print_help_and_exit(argv[0], "Unsupported option: '" + s + "'");
    } else {
      static std::vector<std::string> command_supported = {"C", "M", "D","H"};
      for (auto c : sanitize_command(s)) {
        if (std::find(command_supported.begin(), command_supported.end(), std::string{c}) == command_supported.end())
          print_help_and_exit(argv[0], "Unsupported value for COMMAND");
      }
      commands.push_back(s);
    }
  }
  if (n_queues == -1)
    n_queues = (mode == "host_threads") ? commands.size() : 1;

  if (commands.empty())
    print_help_and_exit(argv[0], "Need to specify COMMANDS (C,M2D,D2M,H2D,D2H)");

  //    _       _                 _
  //   | \  _ _|_ _.     | _|_   |_) _. ._ _. ._ _   _ _|_  _  ._   \  / _. |      _   _
  //   |_/ (/_ | (_| |_| |  |_   |  (_| | (_| | | | (/_ |_ (/_ |     \/ (_| | |_| (/_ _>
  //
  // Add missing global_size
  for (const auto &command : commands)
    commands_parameters_cli.try_emplace("globalsize_" + command, -1);
  std::unordered_map<std::string, size_t> commands_parameters;
  for (const auto & [ k, v ] : commands_parameters_cli)
    commands_parameters[k] = (v == -1) ? get_default_command_parameter(k, commands.size(), commands_parameters_cli) : v;

  std::set<std::string> commands_uniq(commands.begin(), commands.end());
  //                                     __
  //    /\     _|_  _ _|_     ._   _    (_   _  ._ o  _. |
  //   /--\ |_| |_ (_) |_ |_| | | (/_   __) (/_ |  | (_| |
  //
  // We want each command to take the same time. We have only one parameter (kernel_tripcount)
  // In first approximation all our commands are linear in time
  bool need_auto_tunne = false;
  for (const auto k : commands_uniq) {
    const auto name_parameter = commands_to_parameters_tunned(k);
    need_auto_tunne |= (commands_parameters_cli[name_parameter] == -1);
  }
  if (need_auto_tunne && (commands_uniq.size() != 1)) {
    std::cout << "Performing Autotuning" << std::endl;
    // Get the baseline. We assume everything is linear, run the max value
    std::vector<std::string> commands_uniq_vec(commands_uniq.begin(), commands_uniq.end());
    auto[_, serial_commands_times] = bench<float>("serial", commands_uniq_vec, commands_parameters, enable_profiling, n_queues, n_repetitions);

    // Take the mintime of the max value
    long min_time = std::numeric_limits<long>::max();
    for (int i = 0; i < commands_uniq_vec.size(); i++) {
      if (commands_uniq_vec[i] == "C")
        continue;
      min_time = std::min(serial_commands_times[i], min_time);
    }
    // Just need to apply the regression now
    for (int i = 0; i < commands_uniq_vec.size(); i++) {
      const auto name_command = commands_uniq_vec[i];
      const auto name_parameter = commands_to_parameters_tunned(name_command);
      if (commands_parameters_cli[name_parameter] == -1) {
        // Todo check if new_parameter >= max possible values
        long new_parameter = (1. * min_time) / serial_commands_times[i] * commands_parameters[name_parameter];
        commands_parameters[name_parameter] = new_parameter;
      }
    }
  }

  std::cout << "Parameters used:" << std::endl;
  for (const auto k : commands_uniq) {
    const auto name_parameter = commands_to_parameters_tunned(k);
    std::cout << "  " << name_parameter << ": " << commands_parameters[name_parameter] << std::endl;
    if (k == "C")
        std::cout << "  " << "globalsize_C"<< ": " << commands_parameters["globalsize_C"] << std::endl;

  }

  //    _                             __                   _       _
  //   /   _  ._ _  ._     _|_  _    (_   _  ._ o  _. |   |_)  _ _|_ _  ._ _  ._   _  _
  //   \_ (_) | | | |_) |_| |_ (/_   __) (/_ |  | (_| |   | \ (/_ | (/_ | (/_ | | (_ (/_
  //                |
  const auto & [ serial_total_time, serial_commands_times ] = bench<float>("serial", commands, commands_parameters, enable_profiling, n_queues, n_repetitions);
  std::cout << "Best Total Time Serial: " << serial_total_time << "us" << std::endl;
  for (size_t i = 0; i < commands.size(); i++)
    std::cout << "  Best Time Command " << i << " (" << std::setw(3) << commands[i] << "): " << serial_commands_times[i] << "us" << std::endl;

  const double max_speedup = (1. * serial_total_time) / *std::max_element(serial_commands_times.begin(), serial_commands_times.end());
  std::cout << "Maximum Theoretical Speedup: " << max_speedup << "x" << std::endl;

  if (commands.size() >= 1 && max_speedup <= 1.50)
    std::cerr << "  WARNING: Large Unbalance Between Commands" << std::endl;

  const auto & [ concurent_total_time, _ ] = bench<float>(mode, commands, commands_parameters, enable_profiling, n_queues, n_repetitions);
  std::cout << "Best Total Time //: " << concurent_total_time << "us" << std::endl;
  const double speedup = (1. * serial_total_time) / concurent_total_time;
  std::cout << "Speedup Relative to Serial: " << speedup << "x" << std::endl;

  if (max_speedup >= 1.3 * speedup) {
    std::cout << "FAILURE: Far from Theoretical Speedup" << std::endl;
    return 1;
  }

  std::cout << "SUCCESS: Close to Theoretical Speedup" << std::endl;
  return 0;
}
