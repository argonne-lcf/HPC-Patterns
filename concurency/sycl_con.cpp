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
#include <utility>
#include <vector>
#include <iomanip>
#include <numeric>

#include <sycl/sycl.hpp>
#define NUM_REPETION 10

template <class T> T busy_wait(long N, T i) {
  T x = 1.3f;
  T y = (T)i;
  for (long j = 0; j < N; j++) {
    MAD_64(x, y);
  }
  return y;
}

template <class T>
std::pair<long, std::vector<long>> bench(std::string mode, std::vector<std::string> commands, 
          bool enable_profiling, int n_queues, long kernel_tripcount =0) {

//   __
//    |  ._  o _|_
//   _|_ | | |  |_
//
  const sycl::device D{sycl::gpu_selector()};
  const int globalWIs = D.get_info<sycl::info::device::sub_group_sizes>()[0];
  const int N_copy = D.get_info<sycl::info::device::max_mem_alloc_size>() / sizeof(T);
  const sycl::context C(D);

  // By default SYCL queue are out-of-order
  sycl::property_list pl;
  if ( (mode == "in_order") && enable_profiling)
    pl = sycl::property_list{sycl::property::queue::in_order{},
                             sycl::property::queue::enable_profiling{}};
  else if (mode == "in_order")
    pl = sycl::property_list{sycl::property::queue::in_order{}};
  else if (enable_profiling)
    pl = sycl::property_list{sycl::property::queue::enable_profiling{}};

  std::vector<sycl::queue> Qs(n_queues, sycl::queue(C, D, pl)) ;
  //Initialize buffers according to the commands
  std::vector<std::vector<T *>> buffers;
  for (auto &command : commands) {
    std::vector<T *> buffer;
    for (const char &t : command) {
      T *b;
      if (t == 'M') {
        b = static_cast<T *>(calloc(N_copy,sizeof(T)));
      } else if (t == 'D') {
        b = sycl::malloc_device<T>(N_copy, D, C);
      } else if (t == 'C') {
        b = sycl::malloc_device<T>(globalWIs, D, C);
      }
      buffer.push_back(b);
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
  for (int r = 0; r < NUM_REPETION; r++) {
    auto s0 = std::chrono::high_resolution_clock::now();
    //Run all commands
    for (int i = 0; i < commands.size(); i++) {
      const auto s = std::chrono::high_resolution_clock::now();
      sycl::queue Q = Qs[i % n_queues];
      if (commands[i] == "C") {
        T *ptr = buffers[i][0];
        Q.parallel_for(globalWIs, [ptr, kernel_tripcount](sycl::item<1> j) {
          ptr[j] = busy_wait(kernel_tripcount, (T)j);
        });
      } else {
        Q.copy(buffers[i][1], buffers[i][0], N_copy);
      }

      if (mode == "serial") {
        Q.wait();
        const auto e = std::chrono::high_resolution_clock::now();
        const auto curent_kernel_time = std::chrono::duration_cast<std::chrono::microseconds>(e - s).count();
        commands_times[i] = std::min(commands_times[i], curent_kernel_time);
      }
    }
    //Sync all queues
    for (auto &Q : Qs)
      Q.wait();
    // Save time
    const auto e0 = std::chrono::high_resolution_clock::now();
    const auto curent_total_time = std::chrono::duration_cast<std::chrono::microseconds>(e0 - s0).count();
    total_time = std::min(total_time, curent_total_time);
  }

//    _
//   /  |  _   _. ._      ._ 
//   \_ | (/_ (_| | | |_| |_)
//                        |
  for (const auto &buffer : buffers)
    for (const auto &ptr : buffer)
      //Shorter than to remember commands types...
      (sycl::get_pointer_type(ptr, C) != sycl::usm::alloc::unknown) ? sycl::free(ptr, C) : free(ptr);

  return std::make_pair(total_time, commands_times);
}

void print_help_and_exit(std::string binname, std::string msg) {
  if (!msg.empty())
    std::cout << "ERROR: " << msg << std::endl;
  std::string help = "Usage: " + binname +
                     " (in_order | out_of_order | serial)\n"
                     "                [--enable_profiling]\n"
                     "                [--n_queues=<queues>]\n"
                     "                [--kernel_tripcount=<tripcount>]\n"
                     "                COMMAND...\n"
                     "\n"
                     "Options:\n"
                     "--kernel_tripcount       [default: -1]. Number of FMA per compute kernel\n"
                     "                             '-1' mean autotuning of this parameter\n" 
                     "--n_queues=<nqueues>     [default: -1]. Number of queues used to run COMMANDS\n"
                     "                            '-1' mean automatic selection:\n"
                     "                              - if `in_order`, one queue per COMMAND\n"
                     "                              - else one queue\n"
                     "COMMAND                  [possible values: C,MD,DM]\n"
                     "                            C:  Compute kernel\n"
                     "                            MD: Malloc allocated memory to Device memory memcopy\n"
                     "                            DM: Device Memory to Malloc allocated memory memcopy\n";
  std::cout << help << std::endl;
  std::exit(1);
}

int main(int argc, char *argv[]) {
//    _                       _                                         
//   |_) _. ._ _ o ._   _    /  |     /\  ._ _      ._ _   _  ._ _|_  _ 
//   |  (_| | _> | | | (_|   \_ |_   /--\ | (_| |_| | | | (/_ | | |_ _> 
//                      _|                   _|                         
//
  std::vector<std::string> argl(argv + 1, argv + argc);
  if (argl.empty())
    print_help_and_exit(argv[0], "");

  std::string mode{argl[0]};
  if ((mode != "out_of_order") && (mode != "in_order") && (mode != "serial"))
    print_help_and_exit(argv[0], "Need to specify 'in_order', 'out_of_order', 'serial', option");

  bool enable_profiling = false;
  int n_queues = -1;
  long kernel_tripcount = -1;
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
    } else if (s == "--kernel_tripcount") {
      i++;
      if (i < argl.size()) {
        kernel_tripcount = std::stol(argl[i]);
      } else {
        print_help_and_exit(argv[0],
                            "Need to specify an value for '--kernel_tripcount'");
      }
    } else if (s.rfind("-", 0) == 0) {
      print_help_and_exit(argv[0], "Unsupported option: '" + s + "'");
    } else {
      static std::vector<std::string> command_supported = {"C", "MD", "DM"};
      if (std::find(command_supported.begin(), command_supported.end(), s) == command_supported.end())
        print_help_and_exit(argv[0], "Unsupported value for COMMAND");
      commands.push_back(s);
    }
  }
  if (n_queues == -1)
    n_queues = (mode == "in_order") ? commands.size() : 1;

  if (commands.empty())
    print_help_and_exit(argv[0], "Need to specify COMMANDS (C,MD,DM)");

//                                     __                
//    /\     _|_  _ _|_     ._   _    (_   _  ._ o  _. | 
//   /--\ |_| |_ (_) |_ |_| | | (/_   __) (/_ |  | (_| | 
//                                                       
  if (kernel_tripcount == -1 && (std::count(commands.begin(), commands.end(), "C"))) {
    // We want each command to take the same time. We have only one parameter (kernel_tripcount) 
    // In first approximation for the compute kernel T(kernel_time) -> elapsed_time is linear
    long kernel_tripcount0 = 20000;
    // Some strange HW don't have the same BW for MD and DM. 
    // We will autotunne C so that it take the average time for data transfer.
    std::vector<std::string> copy_commands;
    std::copy_if(commands.begin(), commands.end(), std::back_inserter(copy_commands), [](auto s){return s != "C";} );
    const auto& [_1, commands_times] = bench<float>("serial", copy_commands, enable_profiling, n_queues);
    double copy_time = std::accumulate(commands_times.begin(), commands_times.end(),0) / (1.*commands_times.size());
    const auto& [compute_time0, _2] = bench<float>("serial", {"C"}, enable_profiling, n_queues, kernel_tripcount0);
    kernel_tripcount = (1.*kernel_tripcount0/compute_time0)*copy_time;
    std::cout << "Autotuned Kernel Tripcount " << kernel_tripcount << std::endl;
  }

//    _                             __                   _       _                     
//   /   _  ._ _  ._     _|_  _    (_   _  ._ o  _. |   |_)  _ _|_ _  ._ _  ._   _  _  
//   \_ (_) | | | |_) |_| |_ (/_   __) (/_ |  | (_| |   | \ (/_ | (/_ | (/_ | | (_ (/_ 
//                |                                                                    
 const auto& [serial_total_time, serial_commands_times] = bench<float>("serial", commands, enable_profiling, n_queues, kernel_tripcount);
 std::cout << "Total Time Serial " << serial_total_time << "us" << std::endl;
 for (size_t i=0; i < commands.size(); i++)
    std::cout << "  " << std::setw(2) << commands[i] <<  " " << serial_commands_times[i] << "us" << std::endl;

 const double max_speedup = (1.* serial_total_time) / *std::max_element(serial_commands_times.begin(), serial_commands_times.end());
 std::cout << "Maximum Theoretical Speedup " << max_speedup << "x" << std::endl;

  if (max_speedup <= 1.50)
    std::cerr << "  WARNING: Large Unbalance Between Commands" << std::endl;

  const auto& [concurent_total_time, _] = bench<float>(mode, commands, enable_profiling, n_queues, kernel_tripcount);
  std::cout << "Total Time // (us) " << concurent_total_time << "us" << std::endl;
  const double speedup =  (1. * serial_total_time) / concurent_total_time;
  std::cout << "Speedup Relative to Serial " << speedup << "x" << std::endl;


  if (max_speedup >= 1.3*speedup) {
    std::cout << "FAILURE: Far from Theoretical Speedup" << std::endl;
    return 1;
  }
    
  std::cout << "SUCCESS: Close from Theoretical Speedup" << std::endl;
  return 0;
}
