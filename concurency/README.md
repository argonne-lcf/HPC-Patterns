# Experiment Concurrency


Tests if runnings independent COMMANDS in a mode that allows concurrency/overlaps is faster than running them serially. 

## OMP

With OpenMP one can hope to achieve concurrency using two main strategies (please feel free to make a PR that use task):

### `host_threads`

```
#pragma omp parralel for
for () {
    #pragma omp target
    foo() 
}
```

### `nowait`

```
for () {
    #pragma omp target nowait
    foo()
}
```

### Code

```
Usage: ./omp_con (nowait | host_threads | serial)
                 [--kernel_tripcount=<tripcount>]
                 COMMAND...

Options:
--kernel_tripcount       [default: 10000]
when out_of_order, one per COMMANDS when in order
COMMAND                  [possible values: C,MD,DM]
                            C:  Compute kernel
                            MD: Malloc allocated memory to Device memory memcopy
                            DM: Device Memory to Malloc allocated memory memcopy
```

## SYCL

On SYCL  one can hope to achieve concurrency in two main fashions:

### `our of order queue`

```
sycl::queue Q;
#pragma omp parralel for
for () {
    Q.submit(foo)
}
Q.wait()
```

### Multiple `in order` queue

```
std::vector<sycl::queue> Qs
for ()
    Qs.push_back(sycl::queue(in_order))

for (auto Q: Qs) {
    Q.submit();
}

for (auto Q: Qs) {
    Q.wait();
}
```

### Code

```
Usage: ./sycl_con (in_order | out_of_order)
                  [--enable_profiling]
                  [--n_queues=<queues>]
                  [--kernel_tripcount=<tripcount>]
                  COMMAND...

Options:
--kernel_tripcount       [default: 10000]
--n_queues=<nqueues>     [default: -1]. Number of queues used to run COMMANDS.
                                        If -1: one queue when out_of_order, one per COMMANDS when in order
COMMAND                  [possible values: C,MD,DM]
                            C:  Compute kernel
                            MD: Malloc allocated memory to Device memory memcopy
                            DM: Device Memory to Malloc allocated memory memcopy
```
