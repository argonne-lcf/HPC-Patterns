# Compile

```
mpicxx -fiopenmp -fopenmp-targets=spir64 fma_triad_p2p_stress.cpp -o run_for
```
# Usage

```
mpiexec .. ./run_for <number of seconds>
```



