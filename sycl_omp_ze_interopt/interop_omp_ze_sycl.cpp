#include <CL/sycl.hpp>
#include <level_zero/ze_api.h>
#include <CL/sycl/backend/level_zero.hpp>
#include <omp.h>
#include <map>
#include <iostream>
namespace sycl = cl::sycl;

struct syclDeviceInfo {
    sycl::context sycl_context;
    sycl::device sycl_device;
    ze_context_handle_t ze_context;
};

// General case where each Context can have target multiple Device
std::vector<struct syclDeviceInfo> xomp_get_infos_devices() {
    static std::vector<struct syclDeviceInfo> ompDeviceId2Context;
    if (!ompDeviceId2Context.empty())
       return ompDeviceId2Context;

    // This datascructure is a vector used to map a OpenMP device ID to a sycl context and device
   ompDeviceId2Context.resize(omp_get_num_devices());
    //1. Map each level zero (platform,context) to a vector a sycl::device.
    //   This is requied to create SYCL::context spaming multiple devices.
    //   We use a map implace of a inordermap to avoid some implicitly-deleted default constructor error
   std::map<std::pair<ze_driver_handle_t,ze_context_handle_t>, std::vector<sycl::device>> hContext2device;
   for (int D=0; D< omp_get_num_devices(); D++) {
        omp_interop_t o = 0;
        #pragma omp interop init(targetsync: o) device(D)
        int err = -1;
        const ze_driver_handle_t hPlatform = static_cast<ze_driver_handle_t>(omp_get_interop_ptr(o, omp_ipr_platform, &err));
        assert (err >= 0 && "omp_get_interop_ptr(omp_ipr_platform)");
        const ze_context_handle_t hContext = static_cast<ze_context_handle_t>(omp_get_interop_ptr(o, omp_ipr_device_context, &err));
        assert (err >= 0 && "omp_get_interop_ptr(omp_ipr_device_context)");
        // equivalent to:
        //ze_context_handle_t hContext = static_cast<ze_context_handle_t>(omp_target_get_context(0));
        const ze_device_handle_t hDevice =  static_cast<ze_device_handle_t>(omp_get_interop_ptr(o, omp_ipr_device, &err));
        assert (err >= 0 && "omp_get_interop_ptr(omp_ipr_device)");
        #pragma omp interop destroy(o)

        // Store the Level_zero context. This will be required to create the SYCL context latter
        ompDeviceId2Context[D].ze_context = hContext;

        const sycl::platform sycl_platform = sycl::level_zero::make<sycl::platform>(hPlatform);

        ompDeviceId2Context[D].sycl_device = sycl::level_zero::make<sycl::device>(sycl_platform, hDevice);
        hContext2device[std::make_pair(hPlatform,hContext)].push_back(ompDeviceId2Context[D].sycl_device);
    }

    // Construct sycl::contexts who stawn multiple openmp device, if possible.
    // This is N2, but trivial to make it log(N)
    for ( const auto& [ hPlatforContext, sycl_devices]: hContext2device ) {
        const auto& [ _, hContext] = hPlatforContext;
        // This only work because the backend poiter is saved as a shared_pointer in SYCL context with Intel Implementation
        // https://github.com/intel/llvm/blob/ef33c57e48237c7d918f5dab7893554cecc001dd/sycl/source/backend/level_zero.cpp#L59
        // As far as I know this is not required by the SYCL2020 Spec
        const sycl::context sycl_context = sycl::level_zero::make<sycl::context>(sycl_devices, hContext,  sycl::level_zero::ownership::keep);

        for (int D=0; D< omp_get_num_devices(); D++)
           if (ompDeviceId2Context[D].ze_context == hContext)
              ompDeviceId2Context[D].sycl_context = sycl_context;
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

    // OMP -> SYCL
    int *ompMem = (int*)omp_target_alloc_device(N*sizeof(int), D);
    #pragma omp target is_device_ptr(ompMem)
    for (int i=0 ; i < N; i++)
        ompMem[i] = N;

    //Sycl memcopy is using OpenMP pointer
    Q.memcpy(cpuMem, ompMem, N*sizeof(int) ).wait();

    for (int i=0 ; i < N; i++)
        assert(cpuMem[i] == N);

    // SYCL -> OMP
    int *syclMem = sycl::malloc_device<int>(N,Q);
    // Omp memcopy is using SYCL pointer
    #pragma omp target is_device_ptr(syclMem) map( from:cpuMem[0:N])
    for (int i=0 ; i < N; i++)
        cpuMem[i] = syclMem[i];

    for (int i=0 ; i < N; i++)
        assert(cpuMem[i] == 0);

    std::cout << "Computation Done" << std::endl;
}
