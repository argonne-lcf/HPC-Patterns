#define MAD_4(x, y)                                                            \
  x = y * x + y;                                                               \
  y = x * y + x;                                                               \
  x = y * x + y;                                                               \
  y = x * y + x;
#define MAD_16(x, y)                                                           \
  MAD_4(x, y);                                                                 \
  MAD_4(x, y);                                                                 \
  MAD_4(x, y);                                                                 \
  MAD_4(x, y);
#define MAD_64(x, y)                                                           \
  MAD_16(x, y);                                                                \
  MAD_16(x, y);                                                                \
  MAD_16(x, y);                                                                \
  MAD_16(x, y);

#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>
#include <omp.h>
#define NUM_REPETION 2

template <class T> T busy_wait(long N, T i) {
  T x = 1.3f;
  T y = (T)i;
  for (long j = 0; j < N; j++) {
    MAD_64(x, y);
  }
  return y;
}

template <class T>
void bench(std::vector<std::string> commands, long kernel_tripcount,
           std::string mode, int num_threads,
           long *total_cpu_time, long *max_cpu_time_command,
           int *max_index_cpu_time_command) {

  const int globalWIs =  16;
  const int N = 27000000;

  std::vector<T *> buffers;
  for (auto &command : commands) {
    std::vector<T *> buffer;
    char t = command[0];
    T *b;
    if (t == 'M') {
        b = static_cast<T *>(malloc(N * sizeof(T)));
        std::fill(b, b + N, 0);
        #pragma omp target enter data map(alloc: b[:N])
      } else if (t == 'D') {
        b = static_cast<T *>(malloc(N * sizeof(T)));
        #pragma omp target enter data map(alloc: b[:N])
      } else if (t == 'C') {
        b = static_cast<T *>(malloc(globalWIs * sizeof(T)));
        #pragma omp target enter data map(alloc: b[:globalWIs])
      }
   buffers.push_back(b);
  }

  *total_cpu_time = std::numeric_limits<long>::max();
  for (int r = 0; r < NUM_REPETION; r++) {
    std::vector<long> cpu_times;
    auto s0 = std::chrono::high_resolution_clock::now();
    #pragma omp metadirective \
            when( user={condition(mode == "host_threads")}: parallel for) \
            default()
    for (int i = 0; i < commands.size(); i++) {
       std::cout << omp_get_thread_num() << std::endl;
      const auto s = std::chrono::high_resolution_clock::now();
      if (commands[i] == "C") {
        T *ptr = buffers[i];
        #pragma omp metadirective \
                when(user={condition(mode == "nowait")}: target teams distribute parallel for nowait) \
                default(                                 target teams distribute parallel for)
        for (int j=0; j < globalWIs; j++)
          ptr[j] = busy_wait(kernel_tripcount, (T)j);
      } else if (commands[i] == "DM") {
        T *ptr=buffers[i];
        #pragma omp metadirective \
                when(user={condition(mode == "nowait")}: target update from(ptr[:N]) nowait) \
                default(                                 target update from(ptr[:N]))

      } else if (commands[i] == "MD") {
         T *ptr=buffers[i];
        #pragma omp metadirective \
                when(user={condition(mode == "nowait")}: target update to(ptr[:N]) nowait) \
                default(                                 target update to(ptr[:N]))
      }
      if (mode == "serial") {
        const auto e = std::chrono::high_resolution_clock::now();
        cpu_times.push_back(
            std::chrono::duration_cast<std::chrono::microseconds>(e - s)
                .count());
      }
    }
    #pragma omp metadirective \
            when(user={condition(mode == "nowait")}: taskwait) \
            default()
    const auto e0 = std::chrono::high_resolution_clock::now();
    const auto curent_total_cpu_time =
        std::chrono::duration_cast<std::chrono::microseconds>(e0 - s0).count();
    if (curent_total_cpu_time < *total_cpu_time) {
      *total_cpu_time = curent_total_cpu_time;
      if (mode=="serial") {
        *max_index_cpu_time_command =
            std::distance(cpu_times.begin(),
                          std::max_element(cpu_times.begin(), cpu_times.end()));
        *max_cpu_time_command = cpu_times[*max_index_cpu_time_command];
      }
    }
  }


  for (const auto &ptr : buffers) {
    //#pragma omp target exit data 
    free(ptr);
  }
}

void print_help_and_exit(std::string binname, std::string msg) {
  if (!msg.empty())
    std::cout << "ERROR: " << msg << std::endl;
  std::string help = "Usage: " + binname +
                     " (nowait | host_threads | serial)\n"
                     "                [--kernel_tripcount=<tripcount>]\n"
                     "                COMMAND...\n"
                     "\n"
                     "Options:\n"
                     "--kernel_tripcount       [default: 10000]\n"
                     "when out_of_order, one per COMMANDS when in order\n"
                     "COMMAND                  [possible values: C,MD,DM]\n"
                     "                            C:  Compute kernel \n"
                     "                            MD: Malloc allocated memory "
                     "to Device memory memcopy \n"
                     "                            DM: Device Memory to Malloc "
                     "allocated memory memcopy \n";
  std::cout << help << std::endl;
  std::exit(1);
}

int main(int argc, char *argv[]) {
  std::vector<std::string> argl(argv + 1, argv + argc);
  if (argl.empty())
    print_help_and_exit(argv[0], "");

  std::string mode = argl[0];
  if ((mode != "nowait") && (mode != "host_threads") && (mode != "serial") )
    print_help_and_exit(argv[0], "Need to specify 'nowait', 'host_threads', or 'serial' option)");

  int threads_count = -1;
  std::vector<std::string> commands;
  long kernel_tripcount = 10000;

  // I'm just an old C programmer trying to do some C++
  for (int i = 1; i < argl.size(); i++) {
    std::string s{argl[i]};
    if (s == "--threads_count") {
      if (mode != "host_threads")
        std::cout << "Warning  --threads_count only make sense for `host_threads` modes" << std::endl;  
      i++;
      if (i < argl.size()) {
        threads_count = std::stoi(argl[i]);
      } else {
        print_help_and_exit(argv[0], "Need to specify an value for '--threads_count'");
      }
    } else if (s == "--kernel_tripcount") {
      i++;
      if (i < argl.size()) {
        kernel_tripcount = std::stol(argl[i]);
      } else {
        print_help_and_exit(argv[0],
                            "Need to specify an value for '--kernel_tripcount'");
      }
    } else if (s.rfind("-", 0) == 0) {
      print_help_and_exit(argv[0], "Unsuported option: '" + s + "'");
    } else {
      static std::vector<std::string> command_supported = {"C", "MD", "DM"};
      if (std::find(command_supported.begin(), command_supported.end(), s) ==
          command_supported.end())
        print_help_and_exit(argv[0], "Unsuported value for COMMAND");
      commands.push_back(s);
    }
  }
  if (threads_count == -1) {
    threads_count = commands.size();
  }
  if (commands.empty())
    print_help_and_exit(argv[0], "Need to specify somme COMMAND and the order "
                                 "('--out_of_order' or '--in_order')");

  long serial_total_cpu_time;
  int serial_max_cpu_time_index_command;
  long serial_max_cpu_time_command;
  bench<float>(commands, kernel_tripcount, "serial", threads_count, &serial_total_cpu_time, &serial_max_cpu_time_command,
               &serial_max_cpu_time_index_command);
  std::cout << "Total serial (us): " << serial_total_cpu_time
            << " (max commands (us) was "
            << commands[serial_max_cpu_time_index_command] << ": "
            << serial_max_cpu_time_command << ")" << std::endl;
  const double max_speedup =
      1. * serial_total_cpu_time / serial_max_cpu_time_command;
  std::cout << "Maximun Theoritical Speedup (assuming maximun concurency and "
               "negligeable runtime overhead) "
            << max_speedup << "x" << std::endl;
  if (max_speedup <= 1.30)
    std::cerr << "  WARNING: Large unblance between commands. Please play with '--kernel_tripcount' "
              << std::endl;

  long concurent_total_cpu_time;
  bench<float>(commands, kernel_tripcount, mode, threads_count, &concurent_total_cpu_time, NULL, NULL);
  std::cout << "Total // (us):     " << concurent_total_cpu_time << std::endl;
  std::cout << "Got " << 1. * serial_total_cpu_time / concurent_total_cpu_time
            << "x speed-up relative to serial" << std::endl;

  if (concurent_total_cpu_time <= 1.30 * serial_max_cpu_time_command) {
    std::cout << "SUCCESS: Concurent is faster than serial" << std::endl;
    return 0;
  }

  std::cout << "FAILURE: No Concurent Execution" << std::endl;
  return 1;
}
