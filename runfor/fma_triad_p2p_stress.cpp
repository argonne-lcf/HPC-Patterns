#include <algorithm>
#include <assert.h>
#include <iostream>
#include <limits>
#include <mpi.h>
#include <omp.h>
#include <vector>
#include <cstring>

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

int main(int argc, char **argv) {
  MPI_Init(NULL, NULL);

  double MAX_TIME = 60;
  if (argc >= 2)
    MAX_TIME = atof(argv[1]);

  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (world_rank == 0)
    std::cout << "Will run for approx " << MAX_TIME << "s, "
             << "with " << world_size << " ranks"
	     << std::endl;

// Trick to not measure the JIT time
#pragma omp target
  {}

  if (world_rank == 0)
    std::cout << "Jitting done" << std::endl;

  const int globalWI{65536000};
  const int num_iteration{100}; // ~25s for a loop
  std::srand(world_rank*1000); // Need to multipliy to have differnt number whatever...
  std::vector<double> A(globalWI), B(globalWI), C(globalWI);
  std::generate(B.begin(), B.end(), std::rand);
  std::generate(C.begin(), C.end(), std::rand);
  double *Aptr{A.data()};
  double *Bptr{B.data()};
  double *Cptr{C.data()};
  int finish = 0;

  if (world_rank == 0)
    std::cout << "Starting..." << std::endl;
#pragma omp target enter data map(alloc : Aptr[:globalWI])                   \
    map(to : Bptr[:globalWI], Cptr[:globalWI])
  double min_time_bw = std::numeric_limits<double>::max();
  double min_time_flops = std::numeric_limits<double>::max();

  double total_runtime = 0;
  while (true) {
    total_runtime -= omp_get_wtime();
    for (int r = 0; r < num_iteration; r++) {

      {
        // One more sync, why not?
        MPI_Barrier(MPI_COMM_WORLD);
        const double l_start = omp_get_wtime();
#pragma omp target teams distribute parallel for
        for (int64_t i = 0; i < globalWI; i++) {
          double x = 1.3f;
          double y = (double)i;
          for (int j = 0; j < 256; j++) {
            MAD_64(x, y);
          }
          Aptr[i] = y;
        }
        const double l_end = omp_get_wtime();

        double start, end;
        MPI_Reduce(&l_start, &start, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&l_end, &end, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        const double time = end - start;
        min_time_flops = std::min(time, min_time_flops);
      }

      {
        MPI_Barrier(MPI_COMM_WORLD);
        const double l_start = omp_get_wtime();
#pragma omp target teams distribute parallel for
        for (int i = 0; i < globalWI; i++)
          Aptr[i] = 2.0 * Bptr[i] + Cptr[i];
        const double l_end = omp_get_wtime();

        double start, end;
        MPI_Reduce(&l_start, &start, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&l_end, &end, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        const double time = end - start;
        min_time_bw = std::min(time, min_time_bw);
      }
    }

    const double bw =
        (3 * sizeof(double) * globalWI * 1E-9) / min_time_bw;

    const int workPerWI{256 * 64 * 2}; // Indicates flops executed per work-item
    const double gflops =
        (globalWI * static_cast<double>(workPerWI) * 1E-12) /
        min_time_flops;

    if (world_rank == 0) {
      std::cout << "Best Aggreagated BW (GB/s): " << bw*world_size
                << " | Best Average per Rank: " << bw
	      	<< std::endl;
      std::cout << "Best Aggregated TFlops: " << gflops*world_size
	        << " | Best Average per Rank: " << gflops
                << std::endl;
    }

    // Todo: May need to increase the number of message
    // 	    Send Stuff by chunk
    // Do some Exchange
    // Send B to the Left, And C to the right
    MPI_Request sendr, recvr;
    MPI_Status status;
    #pragma omp target data use_device_ptr(Bptr, Aptr)
    {
    MPI_Isend(Bptr, globalWI, MPI_DOUBLE, (world_rank+1)%world_size, 0, MPI_COMM_WORLD, &sendr);
    MPI_Irecv(Aptr, globalWI, MPI_DOUBLE, (world_size+world_rank-1)%world_size, 0, MPI_COMM_WORLD, &recvr);
    MPI_Wait(&sendr, &status);
    MPI_Wait(&recvr, &status);
    omp_target_memcpy(Bptr, Aptr, globalWI *sizeof(double),0,0,0,0);
    }
    #pragma omp target data use_device_ptr(Cptr, Aptr)
    {
    MPI_Isend(Cptr, globalWI, MPI_DOUBLE, (world_size+world_rank-1)%world_size, 0, MPI_COMM_WORLD, &sendr);
    MPI_Irecv(Aptr, globalWI, MPI_DOUBLE, (world_rank+1)%world_size, 0, MPI_COMM_WORLD, &recvr);
    MPI_Wait(&sendr, &status);
    MPI_Wait(&recvr, &status);
    omp_target_memcpy(Cptr, Aptr, globalWI *sizeof(double),0,0,0,0);
    }
    if (world_rank == 0)
        std::cout << "Done Exchanging data with neighboord" << std::endl;
    
    total_runtime += omp_get_wtime();
    if (world_rank == 0) {
      std::cout << "Total Runtime " << total_runtime << " s" << std::endl;
      if (total_runtime > MAX_TIME) {
	      finish = 1;
      }
    }

    MPI_Bcast(&finish, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
    if (finish)
      break;
  }
  //#pragma omp target exit data map(from : Aptr[:globalWI])
  //Add some verification
  if (world_rank == 0)
      std::cout << "SUCCESS: Ran for " << total_runtime << std::endl;
  //
  MPI_Finalize();
}

