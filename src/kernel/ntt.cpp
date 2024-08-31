#include "ntt.h"

template <size_t id>
void fwd_ntt_kernel(sycl::queue& q,
                    sycl::buffer<uint64_t, 1>& data_buf,
                    sycl::buffer<uint64_t, 1>& twiddleFactors_buf,
                    sycl::buffer<uint64_t, 1>& modulus_buf,
                    sycl::buffer<uint64_t, 1>& outData_buf) {

    q.submit([&](sycl::handler& h) {
        // Accessors
        auto data_acc = data_buf.get_access<sycl::access::mode::read_write>(h);
        auto twiddleFactors_acc = twiddleFactors_buf.get_access<sycl::access::mode::read>(h);
        auto modulus_acc = modulus_buf.get_access<sycl::access::mode::read>(h);
        auto outData_acc = outData_buf.get_access<sycl::access::mode::write>(h);

        h.parallel_for<FWD_NTT<id>>(
            sycl::nd_range<1>{sycl::range<1>(FPGA_NTT_SIZE), sycl::range<1>(64)}, [=](sycl::nd_item<1> item) {
                const size_t tid = item.get_global_id(0);
                const size_t N = data_acc.size(); // Get the size of the data using the updated method
                const uint64_t modulus = modulus_acc[0];

                // NTT with Cooley-Tukey Butterfly
                for (size_t s = 1; s < N; s <<= 1) {
                    size_t step = N / (2 * s);
                    for (size_t j = tid; j < s; j += item.get_local_range(0)) {
                        uint64_t w = twiddleFactors_acc[j * step];

                        #pragma unroll 4
                        for (size_t k = j; k < N; k += 2 * s) {
                            uint64_t u = data_acc[k];
                            uint64_t t = (data_acc[k + s] * w) % modulus;

                            data_acc[k] = (u + t < modulus) ? u + t : u + t - modulus;
                            data_acc[k + s] = (u >= t) ? u - t : u + modulus - t;
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                // Write results to outData_acc
                outData_acc[tid] = data_acc[tid];
            });
    });
}

// Explicit instantiation
template void fwd_ntt_kernel<0>(sycl::queue& q,
                                sycl::buffer<uint64_t, 1>& data_buf,
                                sycl::buffer<uint64_t, 1>& twiddleFactors_buf,
                                sycl::buffer<uint64_t, 1>& modulus_buf,
                                sycl::buffer<uint64_t, 1>& outData_buf);
