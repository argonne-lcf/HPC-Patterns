#include "bench.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#define TOL_SPEEDUP 0.3

std::string sanitize_command(std::string command) {
  std::string command_sanitized(command);
  command_sanitized.erase(std::remove(command_sanitized.begin(), command_sanitized.end(), '2'),
                          command_sanitized.end());
  return command_sanitized;
}

template <class T>
std::string time_info(std::vector<std::string> commands, long time,
                      std::unordered_map<std::string, size_t> &commands_parameters) {

  unsigned bytes = 0;
  for (const auto &command : commands)
    if (command != "C")
      bytes += commands_parameters["globalsize_" + command] * sizeof(T);

  std::stringstream sout;
  sout << time << "us";
  if (bytes)
    sout << " (" << (1E-3) * bytes / time << " GBytes/s)";

  return sout.str();
}

void print_help_and_exit(std::string binname, std::string msg) {
  if (!msg.empty())
    std::cout << "ERROR: " << msg << std::endl;
  std::string help =
      "Usage: " + binname + " " + alowed_modes +
      "\n"
      "                [--enable_profiling]\n"
      "                [--tripcount_C <tripcount>]\n"
      "                [--globalsize_{C,A2B} <global_size>]\n"
      "                [--queues <n_queues>]\n"
      "                [--repetitions <n_repetions>]\n"
      "                COMMAND...\n"
      "\n"
      "Options:\n"
      "--tripcount_C               [default: -1]. Each kernel work-item will perform "
      "64*C_tripcount FMA\n"
      "                              '-1' will auto-tune this parameter so each commands take "
      "similar time\n"
      "--globalsize_{C,A2B}        [default: -1]. Work-group size of the commands\n"
      "                             '-1' will auto-tune this parameter so each commands take "
      "similar time\n"
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

size_t get_default_command_parameter(std::string command, size_t num_command,
                                     std::unordered_map<std::string, long> &commands) {
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
  std::unordered_map<std::string, long> commands_parameters_cli = {
      {"globalsize_C", -1}, {"tripcount_C", -1}, {"globalsize_default_memory", -1}};
  bool enable_profiling = false;
  bool verbose = false;

  int n_queues = -1;
  int n_repetitions = 10;

  std::vector<std::string> argl(argv + 1, argv + argc);
  if (argl.empty())
    print_help_and_exit(argv[0], "");

  std::string mode{argl[0]};
  validate_mode({argv[0]}, mode);
  std::vector<std::string> commands;
  // I'm just an old C programmer trying to do some C++
  for (int i = 1; i < argl.size(); i++) {
    std::string s{argl[i]};
    if (s == "--enable_profiling") {
      enable_profiling = true;
    }
    else if (s == "--verbose") {
      verbose = true;
    } else if (s == "--queues") {
      i++;
      if (i < argl.size()) {
        n_queues = std::stoi(argl[i]);
      } else {
        print_help_and_exit(argv[0], "Need to specify an value for '--queues'");
      }
    } else if (s == "--repetitions") {
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
      static std::vector<std::string> command_supported = {"C", "M", "D", "H"};
      const auto sc = sanitize_command(s);
      for (auto c : sc) {
        if (std::find(command_supported.begin(), command_supported.end(), std::string{c}) ==
            command_supported.end())
          print_help_and_exit(argv[0], "Unsupported value for COMMAND");
      }
      commands.push_back(sc);
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
  for (const auto &[k, v] : commands_parameters_cli)
    commands_parameters[k] =
        (v == -1) ? get_default_command_parameter(k, commands.size(), commands_parameters_cli) : v;

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
    std::cout << "Performing Autotuning to Balance Commands Times" << std::endl;
    // Get the baseline. We assume everything is linear, run the max value
    std::vector<std::string> commands_uniq_vec(commands_uniq.begin(), commands_uniq.end());
    auto [_, serial_commands_times] =
        bench<float>("serial", commands_uniq_vec, commands_parameters, enable_profiling, n_queues,
                     n_repetitions, verbose);

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
        long new_parameter =
            (1. * min_time) / serial_commands_times[i] * commands_parameters[name_parameter];
        commands_parameters[name_parameter] = new_parameter;
      }
    }
  }

  std::cout << "Parameters used:" << std::endl;
  for (const auto k : commands_uniq) {
    const auto name_parameter = commands_to_parameters_tunned(k);
    std::cout << "  " << name_parameter << ": " << commands_parameters[name_parameter] << std::endl;
    if (k == "C")
      std::cout << "  "
                << "globalsize_C"
                << ": " << commands_parameters["globalsize_C"] << std::endl;
  }

  //    _                             __                   _       _
  //   /   _  ._ _  ._     _|_  _    (_   _  ._ o  _. |   |_)  _ _|_ _  ._ _  ._   _  _
  //   \_ (_) | | | |_) |_| |_ (/_   __) (/_ |  | (_| |   | \ (/_ | (/_ | (/_ | | (_ (/_
  //                |
  const auto &[serial_total_time, serial_commands_times] = bench<float>(
      "serial", commands, commands_parameters, enable_profiling, n_queues, n_repetitions, verbose);
  std::cout << "Best Total Time Serial: " << serial_total_time << "us" << std::endl;
  for (size_t i = 0; i < commands.size(); i++) {
    std::cout << "  Best Time Command " << i << " (" << std::setw(3) << commands[i] << "): "
              << time_info<float>({commands[i]}, serial_commands_times[i], commands_parameters)
              << std::endl;
  }

  const double max_speedup =
      (1. * serial_total_time) /
      *std::max_element(serial_commands_times.begin(), serial_commands_times.end());
  std::cout << "Maximum Theoretical Speedup: " << max_speedup << "x" << std::endl;

  if (commands.size() >= 1 && max_speedup <= 1.50)
    std::cerr << "  WARNING: Large Unbalance Between Commands" << std::endl;

  const auto &[concurent_total_time, _] = bench<float>(
      mode, commands, commands_parameters, enable_profiling, n_queues, n_repetitions, verbose);
  std::cout << "Best Total Time //: "
            << time_info<float>(commands, concurent_total_time, commands_parameters) << std::endl;
  const double speedup = (1. * serial_total_time) / concurent_total_time;
  std::cout << "Speedup Relative to Serial: " << speedup << "x" << std::endl;

  if (max_speedup >= ((1. + TOL_SPEEDUP) * speedup)) {
    std::cout << "FAILURE: Far from Theoretical Speedup" << std::endl;
    return 1;
  }

  std::cout << "SUCCESS: Close from Theoretical Speedup" << std::endl;
  return 0;
}
