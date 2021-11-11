# allreduce

The examples show how to
* Use GPU-aware  point-to-point communictions to perform allreduce of an
  array in a ring topology
* Use GPU-aware `MPI_Allreduce` of the same object

In all cases, all the computations (assignments and additions) are executed on
the device and device pointers are used by the MPI calls. The data are
explicitly copied back to host for validation only.

## mpi-omp-offload

* allreduce-map-mpi-omp-offload.cpp uses malloc on the host and  map clause as
```
#pragma omp target enter data map(alloc                                        \
                                  : VA [0:array_size], VB [0:array_size],      \
                                    VC [0:array_size])
...

#pragma omp target exit data map(delete                                        \
                                 : VA [0:array_size], VB [0:array_size],       \
                                   VC [0:array_size])

```

* allreduce-usm-mpi-omp-offload.cpp lets the user select the allocator
    * `omp_target_alloc` (default)
    * `omp_target_alloc_host` (H)
    * `omp_target_alloc_device` (D)
    * `omp_target_alloc_shared` (S)

## mpi-sycl

* allreduce-mpi-sycl.cpp lets the user select the allocator kind for `aligned_alloc` 
    * `sycl::usm::alloc::host` (H)
    * `sycl::usm::alloc::device` (D)
    * `sycl::usm::alloc::shared` (S, default)
