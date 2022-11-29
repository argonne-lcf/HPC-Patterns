#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include <utility>

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

template <class T> 
static T busy_wait(size_t N, T i) {
  T x = 1.3f;
  T y = i;
  for (size_t j = 0; j < N; j++) {
    MAD_64(x, y);
  }
  return y;
}
extern const std::string alowed_modes;

extern void validate_mode(std::string binname, std::string &mode);
extern void print_help_and_exit(std::string binname, std::string msg);

template <class T>
extern std::pair<long, std::vector<long>> bench(std::string mode, std::vector<std::string> &commands, 
                                                std::unordered_map<std::string, size_t> &commands_parameters, 
                                                bool enable_profiling, int n_queues, int n_repetitions, bool verbose = false); 

