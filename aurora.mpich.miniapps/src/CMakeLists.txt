cmake_minimum_required(VERSION 3.16)
project(minihpa LANGUAGES CXX)

enable_testing()

set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
#set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

option(ENABLE_MPI_SYCL "Enable/Disable MPI + SYCL" ON)
option(ENABLE_MPI_OMP_OFFLOAD "Enable/Disable MPI + OMP OFFLOAD" ON)

function(add_variants variant)
  message("Enabling ${variant}")
  file(GLOB dirs "${PROJECT_SOURCE_DIR}/*/${variant}")
  foreach(dir ${dirs})
    message("  add: ${dir}")
    add_subdirectory(${dir})
  endforeach()
endfunction()

function(add_sycl_options)
  add_compile_options(-fsycl -fsycl-unnamed-lambda)
  add_link_options(-fsycl)
endfunction(add_sycl_options)

function(add_omp_offload_options)
  add_compile_options(-fma -fiopenmp -fopenmp-targets=spir64)
  add_link_options(-fiopenmp -fopenmp-targets=spir64)
endfunction(add_omp_offload_options)

find_package(MPI REQUIRED)

function(add_mpi_options)
  link_libraries(${MPI_CXX_LIBRARIES})
  include_directories(${MPI_CXX_INCLUDE_DIRS})
endfunction(add_mpi_options)

#split test definition to make it app-specific
function(add_mpi_app app)
  add_executable(${app} ${app}.cpp ${ARGN})
  add_test(NAME ${app} COMMAND mpirun -np 4 ./${app} COMMAND_EXPAND_LISTS)
endfunction(add_mpi_app)

#split test definition to make it app-specific
function(add_typed_mpi_app base type)
  set(app "${base}.${type}")
  add_executable(${app} ${base}.cpp)
  target_compile_definitions(${app} PUBLIC APP_DATA_TYPE=${type})
  add_test(NAME ${app} COMMAND mpirun -np 4 ./${app} COMMAND_EXPAND_LISTS)
endfunction(add_typed_mpi_app)

set(BLA_VENDOR Intel10_64lp)
find_package(BLAS REQUIRED)

add_compile_options(-Werror -O2 -Drestrict=__restrict__)

if(ENABLE_CUDA)
  find_package(CUDA REQUIRED)
  add_compile_options("-fsycl-targets=nvptx64-nvidia-cuda-sycldevice")
  set(OMPT_FLAGS "-fopenmp -fopenmp-targets=nvptx64-nvidia-cuda")
  #set(OMPT_FLAGS " -D__GCC_ATOMIC_TEST_AND_SET_TRUEVAL=1 -mp=gpu -gpu=cc80 -D__NO_UDR -D__forceinline=inline -fast -O3 -DNDEBUG ")
else()
endif()

include_directories(include)

if(ENABLE_MPI_SYCL)
  add_variants(mpi-sycl)
endif()
if(ENABLE_MPI_OMP_OFFLOAD)
  add_variants(mpi-omp-offload)
endif()
