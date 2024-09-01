// ntt.cpp
#include "ntt.hpp"

// Forward NTT kernel using SYCL
template <size_t id>
void fwd_ntt_kernel(sycl::queue& q,
                    sycl::buffer<uint64_t, 1>& data_buf,
                    sycl::buffer<uint64_t, 1>& twiddleFactors_buf,
                    sycl::buffer<uint64_t, 1>& modulus_buf,
                    sycl::buffer<uint64_t, 1>& outData_buf) {

    q.submit([&](sycl::handler& h) {
        // Accessors (captured by value)
        auto data_acc = data_buf.get_access<sycl::access::mode::read_write>(h);
        auto twiddleFactors_acc = twiddleFactors_buf.get_access<sycl::access::mode::read>(h);
        auto modulus_acc = modulus_buf.get_access<sycl::access::mode::read>(h);
        auto outData_acc = outData_buf.get_access<sycl::access::mode::write>(h);

        // Kernel execution
        h.single_task<FWD_NTT<id>>([=]() {
            const size_t N = data_acc.get_range().size();  // Correctly get the size of the buffer

            for (size_t m = 1, t = N >> 1; m < N; m <<= 1, t >>= 1) {
                size_t k = 0;
                for (size_t i = 0; i < m; i++) {
                    const uint64_t w = twiddleFactors_acc[m + i];

                    for (size_t j = k; j < k + t; j++) {
                        uint64_t a0 = data_acc[j];
                        uint64_t a1 = data_acc[j + t];

                        // Harvey butterfly operation
                        uint64_t u = a0;
                        uint64_t v = a1 * w % modulus_acc[0];

                        data_acc[j] = (u + v) % modulus_acc[0];
                        data_acc[j + t] = (u + modulus_acc[0] - v) % modulus_acc[0];
                    }
                    k += (2 * t);
                }
            }

            // Copy the result to the output buffer
            for (size_t i = 0; i < N; i++) {
                outData_acc[i] = data_acc[i];
            }
        });
    });
}

// Explicit template instantiation for the kernel
template void fwd_ntt_kernel<0>(sycl::queue& q,
                                sycl::buffer<uint64_t, 1>& data_buf,
                                sycl::buffer<uint64_t, 1>& twiddleFactors_buf,
                                sycl::buffer<uint64_t, 1>& modulus_buf,
                                sycl::buffer<uint64_t, 1>& outData_buf);
