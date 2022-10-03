#include <sycl/sycl.hpp>
#include <omp.h>
#include <map>
#include <iostream>
namespace sycl = cl::sycl;

struct syclDeviceInfo {
    sycl::context sycl_context;
    sycl::device sycl_device;
};

// General case where each Context can have target multiple Device
std::vector<struct syclDeviceInfo> xomp_get_infos_devices() {
    static std::vector<struct syclDeviceInfo> ompDeviceId2Context;
    if (!ompDeviceId2Context.empty())
       return ompDeviceId2Context;

    // This datascructure is a vector used to map a OpenMP device ID to a sycl context and device
   ompDeviceId2Context.resize(omp_get_num_devices());
    //   Map each level zero (pcontext) to a vector a sycl::device.
    //   This is requied to create SYCL::context spaming multiple devices.
   std::map<ze_context_handle_t, std::vector<sycl::device>> hContext2device;
   for (int D=0; D< omp_get_num_devices(); D++) {
        omp_interop_t o = 0;
        #pragma omp interop init(prefer_type("sycl"), targetsync: o) device(D)
        int err = -1;
        auto* sycl_context = static_cast<sycl::context *>(omp_get_interop_ptr(o, omp_ipr_device_context, &err));
        assert (err >= 0 && "omp_get_interop_ptr(omp_ipr_device_context)");
        auto* sycl_device =  static_cast<sycl::device *>(omp_get_interop_ptr(o, omp_ipr_device, &err));
        assert (err >= 0 && "omp_get_interop_ptr(omp_ipr_device)");
        #pragma omp interop destroy(o)

        ompDeviceId2Context[D].sycl_context = sycl_context[0];
        ompDeviceId2Context[D].sycl_device = sycl_device[0];
    }
    return ompDeviceId2Context;
}

const struct syclDeviceInfo xomp_get_device_info(const int n) {
    return xomp_get_infos_devices()[n];
}

int main() {

    const int D = omp_get_num_devices() -1;
    omp_set_default_device(D);

     // Now create a SYCL Q who is targetting this Device
    sycl::queue Q { xomp_get_device_info(D).sycl_context, xomp_get_device_info(D).sycl_device};

    const int N = 100;
    int *cpuMem = (int*) malloc(N*sizeof(int));

    std::cout << "OMP -> SYCL" << std::endl;
    int *ompMem = (int*)omp_target_alloc_device(N*sizeof(int), D);
    #pragma omp target is_device_ptr(ompMem)
    for (int i=0 ; i < N; i++)
        ompMem[i] = N;

    std::cout << "   SYCL memcopy using OpenMP pointer" << std::endl;
    Q.memcpy(cpuMem, ompMem, N*sizeof(int) ).wait();

    for (int i=0 ; i < N; i++)
        assert(cpuMem[i] == N);

    std::cout << "SYCL -> OMP" << std::endl;
    int *syclMem = sycl::malloc_device<int>(N,Q);

    std::cout << "  OMP memcopy using SYCL pointer" << std::endl;
    #pragma omp target is_device_ptr(syclMem) map( from:cpuMem[0:N])
    for (int i=0 ; i < N; i++)
        cpuMem[i] = syclMem[i];

    for (int i=0 ; i < N; i++)
        assert(cpuMem[i] == 0);

    std::cout << "Computation Done" << std::endl;
}
