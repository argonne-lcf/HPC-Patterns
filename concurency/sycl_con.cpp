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
#include <unordered_map>
#include <utility>
#include <vector>
#include <set>

#include <sycl/sycl.hpp>
template <class T> T busy_wait(size_t N, T i) {
  T x = 1.3f;
  T y = i;
  for (size_t j = 0; j < N; j++) {
    MAD_64(x, y);
  }
  return y;
}

template <class T>
std::pair<long, std::vector<long>> bench(std::string mode, std::vector<std::string>& commands, std::unordered_map<std::string, size_t>& commands_parameters,
          bool enable_profiling, int n_queues, int n_repetitions) {

  //   ___
  //    |  ._  o _|_
  //   _|_ | | |  |_
  //
  const sycl::device D{sycl::gpu_selector()};
  const sycl::context C(D);
  // By default SYCL queue are out-of-order
  sycl::property_list pl;
  if ((mode == "in_order") && enable_profiling)
    pl = sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}};
  else if (mode == "in_order")
    pl = sycl::property_list{sycl::property::queue::in_order{}};
  else if (enable_profiling)
    pl = sycl::property_list{sycl::property::queue::enable_profiling{}};

  // List of queues!
  // One shoud not use 'Qs(n_queues, sycl::queue(C, D, pl))'!
  // This copy queue and hence sharing the native objects.
  std::vector<sycl::queue> Qs;
  for (size_t i=0; i < n_queues; i++)
    Qs.push_back(sycl::queue(C, D, pl));

  // Initialize buffers according to the commands
  std::vector<std::vector<T *>> buffers;
  for (auto &command : commands) {
    const auto N = commands_parameters["globalsize_" + command];
    std::vector<T *> buffer;
    std::string command_sanitized(command);
    command_sanitized.erase(std::remove(command_sanitized.begin(), command_sanitized.end(), '2'), command_sanitized.end()); 
    for (auto c: command_sanitized) {
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
    std::fill_n(std::back_inserter(commands_times), commands.size(), std::numeric_limits<long>::max());

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
        Q.parallel_for(sycl::range{N}, [ptr, kernel_tripcount](sycl::id<1> j) { ptr[j] = busy_wait(kernel_tripcount, (T)j); });
      } else {
        //Copy is src -> dest
        Q.copy(buffers[i][0], buffers[i][1], N);
      }

      if (mode == "serial") {
        Q.wait();
        const auto e = std::chrono::high_resolution_clock::now();
        const auto curent_kernel_time = std::chrono::duration_cast<std::chrono::microseconds>(e - s).count();
        commands_times[i] = std::min(commands_times[i], curent_kernel_time);
      }
    }
    // Sync all queues
    for (auto &Q : Qs)
      Q.wait();
    // Save time
    const auto e0 = std::chrono::high_resolution_clock::now();
    const auto curent_total_time = std::chrono::duration_cast<std::chrono::microseconds>(e0 - s0).count();
    total_time = std::min(total_time, curent_total_time);
  }

  // Assume the "best theoritical" serial
  if (mode == "serial")
    total_time = std::min(total_time, std::accumulate(commands_times.begin(), commands_times.end(), 0L));

  //    _
  //   /  |  _   _. ._      ._
  //   \_ | (/_ (_| | | |_| |_)
  //                        |
  for (const auto &buffer : buffers)
    for (const auto &ptr : buffer)
      // Shorter than to remember commands types...
      (sycl::get_pointer_type(ptr, C) != sycl::usm::alloc::unknown) ? sycl::free(ptr, C) : free(ptr);

  return std::make_pair(total_time, commands_times);
}

void print_help_and_exit(std::string binname, std::string msg) {
  if (!msg.empty())
    std::cout << "ERROR: " << msg << std::endl;
  std::string help = "Usage: " + binname +
                     " (in_order | out_of_order | serial)\n"
                     "                [--enable_profiling]\n"
                     "                [--tripcount_C <tripcount>]\n"
                     "                [--globalsize_{C,M2D,D2M} <global_size>]\n"
                     "                [--queues <n_queues>]\n"
                     "                [--repetitions <n_repetions>]\n"
                     "                COMMAND...\n"
                     "\n"
                     "Options:\n"
                     "--tripcount_C             [default: -1]. Each kernel work-item will perform 64*C_tripcount FMA\n"
                     "                            '-1' will auto-tune this parameter so each commands take similar time\n"
                     "--globalsize_{C,M2D,D2M}    [default: -1]. Work-group size of the commands\n"
                     "                            '-1' will auto-tune this parameter so each commands take similar time\n"
                     "--queues                  [default: -1]. Number of queues used to run COMMANDS\n"
                     "                            '-1' mean automatic selection:\n"
                     "                              - if `in_order`, one queue per COMMAND\n"
                     "                              - else one queue\n"
                     "--repetitions             [default: 10]. Number of repetions for each measuremnts\n"
                     "COMMAND                   [possible values: C,M2D,D2M]\n"
                     "                             C:  Compute kernel\n"
                     "                             M2D: Malloc allocated memory to Device memory memcopy\n"
                     "                             D2M: Device Memory to Malloc allocated memory memcopy\n";
  std::cout << help << std::endl;
  std::exit(1);
}

int main(int argc, char *argv[]) {
  //    _                       _
  //   |_) _. ._ _ o ._   _    /  |     /\  ._ _      ._ _   _  ._ _|_  _
  //   |  (_| | _> | | | (_|   \_ |_   /--\ | (_| |_| | | | (/_ | | |_ _>
  //                      _|                   _|
  //
  std::unordered_map<std::string, long> commands_parameters_cli = {{"globalsize_M2D", -1}, {"globalsize_D2M", -1}, {"globalsize_C", -1}, {"tripcount_C", -1}};
  bool enable_profiling = false;
  int n_queues = -1;
  int n_repetitions=10;

  std::vector<std::string> argl(argv + 1, argv + argc);
  if (argl.empty())
    print_help_and_exit(argv[0], "");

  std::string mode{argl[0]};
  if ((mode != "out_of_order") && (mode != "in_order") && (mode != "serial"))
    print_help_and_exit(argv[0], "Need to specify 'in_order', 'out_of_order', 'serial', option");

  std::vector<std::string> commands;
  // I'm just an old C programmer trying to do some C++
  for (int i = 1; i < argl.size(); i++) {
    std::string s{argl[i]};
    if (s == "--enable_profiling") {
      enable_profiling = true;
    } else if (s == "--queues") {
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
      static std::vector<std::string> command_supported = {"C", "M2D", "D2M"};
      if (std::find(command_supported.begin(), command_supported.end(), s) == command_supported.end())
        print_help_and_exit(argv[0], "Unsupported value for COMMAND");
      commands.push_back(s);
    }
  }
  if (n_queues == -1)
    n_queues = (mode == "in_order") ? commands.size() : 1;

  if (commands.empty())
    print_help_and_exit(argv[0], "Need to specify COMMANDS (C,M2D,D2M)");

  //    _       _                 _
  //   | \  _ _|_ _.     | _|_   |_) _. ._ _. ._ _   _ _|_  _  ._   \  / _. |      _   _
  //   |_/ (/_ | (_| |_| |  |_   |  (_| | (_| | | | (/_ |_ (/_ |     \/ (_| | |_| (/_ _>
  //
  const sycl::device D{sycl::gpu_selector()};
  const auto max_mem_alloc_command = std::min(D.get_info<sycl::info::device::global_mem_size>() / commands.size(), D.get_info<sycl::info::device::max_mem_alloc_size>());
  std::unordered_map<std::string, size_t> commands_parameters_default = {
                                                                       {"globalsize_M2D", max_mem_alloc_command / sizeof(float)},
                                                                       {"globalsize_D2M", max_mem_alloc_command / sizeof(float)},
                                                                       {"globalsize_C", D.get_info<sycl::info::device::sub_group_sizes>()[0]},
                                                                       {"tripcount_C", 40000}};

  std::unordered_map<std::string,size_t> commands_parameters;
  for (const auto &[k,v]: commands_parameters_cli)
    commands_parameters[k] = (v == -1) ? commands_parameters_default[k] : v;

  std::set<std::string> commands_uniq(commands.begin(), commands.end());
  //                                     __
  //    /\     _|_  _ _|_     ._   _    (_   _  ._ o  _. |
  //   /--\ |_| |_ (_) |_ |_| | | (/_   __) (/_ |  | (_| |
  //

   std::unordered_map<std::string,std::string> commands_parameters_tunned{ { "C", "tripcount_C"},
                                                                           { "M2D", "globalsize_M2D"},
                                                                           { "D2M", "globalsize_D2M"}  };


  // We want each command to take the same time. We have only one parameter (kernel_tripcount)
  // In first approximation all our commands are linear in time
  bool need_auto_tunne = false;
  for (const auto k: commands_uniq) {
    const auto name_parameter = commands_parameters_tunned[k];
    need_auto_tunne |= (commands_parameters_cli[name_parameter] == -1);
  }

  if (need_auto_tunne && (commands_uniq.size() != 1)) {
    std::cout << "Performing Autotuning" << std::endl;
    // Get the baseline. We assume everything is linear, run the max value
    std::vector<std::string> commands_uniq_vec(commands_uniq.begin(), commands_uniq.end());
    auto [_, serial_commands_times ] = bench<float>("serial", commands_uniq_vec, commands_parameters, enable_profiling, n_queues, n_repetitions);
    // Take the mintime of the max value 
    long min_time =  std::numeric_limits<long>::max();
    for (int i=0; i < commands_uniq_vec.size(); i++ ) {
      if (commands_uniq_vec[i] == "C")
        continue;
      min_time = std::min(serial_commands_times[i], min_time);
    }

    // Just need to apply the regression now 
    for (int i=0 ; i <= commands_uniq_vec.size() ; i++) {
        const auto name_command = commands_uniq_vec[i];
        const auto name_parameter = commands_parameters_tunned[name_command];
        if (commands_parameters_cli[name_parameter] == -1) { 
            // Todo check if new_parameter >= max possible values
            long new_parameter = (1. * min_time) / serial_commands_times[i] * commands_parameters[name_parameter];
            commands_parameters[name_parameter] = new_parameter;
        }
    }
  }

  std::cout <<"Parameters used:" << std::endl;
  for (const auto k: commands_uniq) {
    const auto name_parameter = commands_parameters_tunned[k];
    std::cout << "  "<< name_parameter << ": " << commands_parameters[name_parameter] << std::endl;
  }

  //    _                             __                   _       _
  //   /   _  ._ _  ._     _|_  _    (_   _  ._ o  _. |   |_)  _ _|_ _  ._ _  ._   _  _
  //   \_ (_) | | | |_) |_| |_ (/_   __) (/_ |  | (_| |   | \ (/_ | (/_ | (/_ | | (_ (/_
  //                |
  const auto & [ serial_total_time, serial_commands_times ] = bench<float>("serial", commands, commands_parameters, enable_profiling, n_queues,n_repetitions);
  std::cout << "Best Total Time Serial: " << serial_total_time << "us" << std::endl;
  for (size_t i = 0; i < commands.size(); i++)
    std::cout << "  Best Time Command " << i << " (" << std::setw(3) << commands[i] << "): " << serial_commands_times[i] << "us" << std::endl;

  const double max_speedup = (1. * serial_total_time) / *std::max_element(serial_commands_times.begin(), serial_commands_times.end());
  std::cout << "Maximum Theoretical Speedup: " << max_speedup << "x" << std::endl;

  if (commands.size() >= 1 && max_speedup <= 1.50)
    std::cerr << "  WARNING: Large Unbalance Between Commands" << std::endl;

  const auto & [ concurent_total_time, _ ] = bench<float>(mode, commands, commands_parameters, enable_profiling, n_queues,n_repetitions);
  std::cout << "Best Total Time //: " << concurent_total_time << "us" << std::endl;
  const double speedup = (1. * serial_total_time) / concurent_total_time;
  std::cout << "Speedup Relative to Serial: " << speedup << "x" << std::endl;

  if (max_speedup >= 1.3 * speedup) {
    std::cout << "FAILURE: Far from Theoretical Speedup" << std::endl;
    return 1;
  }

  std::cout << "SUCCESS: Close from Theoretical Speedup" << std::endl;
  return 0;
}
