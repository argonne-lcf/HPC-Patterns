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

#include <chrono>
#include <iostream>
#include <sycl/sycl.hpp>
#define NUM_REPETION 8

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
           bool enable_profiling, bool in_order, int n_queues, bool serial,
           long *total_cpu_time, long *max_cpu_time_command,
           int *max_index_cpu_time_command) {

  const sycl::device D{sycl::gpu_selector()};
  const int globalWIs = D.get_info<sycl::info::device::sub_group_sizes>()[0];
  const int N = D.get_info<sycl::info::device::max_mem_alloc_size>() / sizeof(T);

  const sycl::context C(D);
  sycl::property_list pl;
  if (enable_profiling && in_order)
    pl = sycl::property_list{sycl::property::queue::in_order{},
                             sycl::property::queue::enable_profiling{}};
  else if (enable_profiling)
    pl = sycl::property_list{sycl::property::queue::enable_profiling{}};
  else if (in_order)
    pl = sycl::property_list{sycl::property::queue::in_order{}};

  std::vector<sycl::queue> Qs;
  for (int i = 0; i < n_queues; i++)
    Qs.push_back(sycl::queue(C, D, pl));

  std::vector<std::vector<T *>> buffers;
  for (auto &command : commands) {
    std::vector<T *> buffer;
    for (const char &t : command) {
      T *b;
      if (t == 'M') {
        b = static_cast<T *>(malloc(N * sizeof(T)));
        std::fill(b, b + N, 0);
      } else if (t == 'D') {
        b = sycl::malloc_device<T>(N, D, C);
      } else if (t == 'C') {
        b = sycl::malloc_device<T>(globalWIs, D, C);
      }
      buffer.push_back(b);
    }
    buffers.push_back(buffer);
  }

  *total_cpu_time = std::numeric_limits<long>::max();
  for (int r = 0; r < NUM_REPETION; r++) {
    std::vector<long> cpu_times;
    auto s0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < commands.size(); i++) {
      const auto s = std::chrono::high_resolution_clock::now();
      sycl::queue Q = Qs[i % n_queues];
      sycl::event event;
      if (commands[i] == "C") {
        T *ptr = buffers[i][0];
        Q.parallel_for(globalWIs, [ptr, kernel_tripcount](sycl::item<1> j) {
          ptr[j] = busy_wait(kernel_tripcount, (T)j);
        });
      } else {
        Q.copy(buffers[i][1], buffers[i][0], N);
      }

      if (serial) {
        Q.wait();
        const auto e = std::chrono::high_resolution_clock::now();
        cpu_times.push_back(
            std::chrono::duration_cast<std::chrono::microseconds>(e - s)
                .count());
      }
    }
    for (auto &Q : Qs)
      Q.wait();
    const auto e0 = std::chrono::high_resolution_clock::now();
    const auto curent_total_cpu_time =
        std::chrono::duration_cast<std::chrono::microseconds>(e0 - s0).count();
    if (curent_total_cpu_time < *total_cpu_time) {
      *total_cpu_time = curent_total_cpu_time;
      if (serial) {
        *max_index_cpu_time_command =
            std::distance(cpu_times.begin(),
                          std::max_element(cpu_times.begin(), cpu_times.end()));
        *max_cpu_time_command = cpu_times[*max_index_cpu_time_command];
      }
    }
  }
  for (const auto &buffer : buffers)
    for (const auto &ptr : buffer)
      (sycl::get_pointer_type(ptr, C) != sycl::usm::alloc::unknown)
          ? sycl::free(ptr, C)
          : free(ptr);
}

void print_help_and_exit(std::string msg) {
  if (!msg.empty())
    std::cout << "ERROR: " << msg << std::endl;
  const char *help = "Usage: ./a.out  (in_order | out_of_order)\n"
                     "                [--enable_profiling]\n"
                     "                [--n_queues=<queues>]\n"
                     "                [--kernel_tripcount=<tripcount>]\n"
                     "                COMMAND...\n"
                     "\n"
                     "Options:\n"
                     "--kernel_tripcount       [default: 10000]\n"
                     "--n_queues=<nqueues>     [default: -1]. Number of queues used to run COMMANDS. \n"
                     "                                        If -1: one queue when out_of_order, one per COMMANDS when in order\n"
                     "COMMAND                  [possible values: C,MD,DM]\n"
                     "                            C:  Compute kernel \n"
                     "                            MD: Malloc allocated memory to Device memory memcopy \n"
                     "                            DM: Device Memory to Malloc allocated memory memcopy \n";
  std::cout << help << std::endl;
  std::exit(1);
}

int main(int argc, char *argv[]) {
  std::vector<std::string> argl(argv+1, argv + argc);
  if (argl.empty())
    print_help_and_exit("");

  bool in_order = false;
  if ((argl[0] != "out_of_order") && (argl[0] != "in_order"))
    print_help_and_exit("Need to specify (in_order,out_of_order)");
  else if (argl[0] == "in_order")
    in_order = true;
  // in_order = false; mean out_of_order. I know stupid optimization

  bool enable_profiling = false;
  int n_queues = -1;
  std::vector<std::string> commands;
  long kernel_tripcount = 10000;
  
  // I'm just an old C programmer trying to do some C++
  for (int i=1 ; i < argl.size(); i++) {
    std::string s{argl[i]};
    if (s == "--enable_profiling") {
      enable_profiling = true;
    } else if (s == "--queues") {
      i++;
      if (i < argl.size()) {
        n_queues = std::stoi(argl[i]);
      } else {
        print_help_and_exit("Need to specify an value for --queues");
      }
    } else if (s == "--kernel_tripcount") {
      i++;
      if (i < argl.size()) {
        kernel_tripcount = std::stol(argl[i]);
      } else {
        print_help_and_exit("Need to specify an value for --kernel_tripcount");
      }
    } else if (s.rfind("-", 0) == 0) {
      print_help_and_exit("Unsuported option: '" + s + "'");
    } else {
      static std::vector<std::string> command_supported = {"C", "MD", "DM"};
      if (std::find(command_supported.begin(), command_supported.end(), s) ==
          command_supported.end())
        print_help_and_exit("Unsuported value for COMMAND");
      commands.push_back(s);
    }
  }
  if (n_queues == -1) {
    if (in_order)
      n_queues = 1;
    else
      n_queues = commands.size();
  }
  if (commands.empty())
    print_help_and_exit("Need to specify somme COMMAND and the order "
                        "(--out_of_order or --in_order)");

  long serial_total_cpu_time;
  int serial_max_cpu_time_index_command;
  long serial_max_cpu_time_command;
  bench<float>(commands, kernel_tripcount, enable_profiling, in_order, n_queues,
               true, &serial_total_cpu_time, &serial_max_cpu_time_command,
               &serial_max_cpu_time_index_command);
  std::cout << "Total serial (us): " << serial_total_cpu_time
            << " (max commands (us) was "
            << commands[serial_max_cpu_time_index_command] << ": "
            << serial_max_cpu_time_command << ")" << std::endl;
  const double expected =
      1. * serial_total_cpu_time / serial_max_cpu_time_command;
  std::cout << "Expecting (assuming maximun concurency and negligeable runtime "
               "overhead) "
            << expected << "x speedup" << std::endl;
  if (expected <= 1.30)
    std::cerr << " Warining: Large unblance between the commands... You may "
                 "not want to trust bellow conclusion."
              << std::endl;

  long concurent_total_cpu_time;
  bench<float>(commands, kernel_tripcount, enable_profiling, in_order, n_queues,
               false, &concurent_total_cpu_time, NULL, NULL);
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
