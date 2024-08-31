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

        h.single_task<FWD_NTT<id>>([=]() {
            const size_t N = data_acc.size();
            const uint64_t modulus = modulus_acc[0];

            for (size_t t = 1; t < N; t <<= 1) {
                size_t m = t << 1;

                // First loop: j = 0 so w_t^j = 1
                for (size_t s = 0; s < N; s += m) {
                    uint64_t x = data_acc[s + t];
                    data_acc[s + t] = (data_acc[s] + modulus - x) % modulus;
                    data_acc[s] = (data_acc[s] + x) % modulus;
                }

                // General case: j > 0
                for (size_t j = 1; j < t; j++) {
                    uint64_t w = twiddleFactors_acc[t + j];

                    for (size_t s = j; s < N; s += m) {
                        uint64_t x = (data_acc[s + t] * w) % modulus;
                        data_acc[s + t] = (data_acc[s] + modulus - x) % modulus;
                        data_acc[s] = (data_acc[s] + x) % modulus;
                    }
                }
            }

            // Write results to outData_acc
            for (size_t i = 0; i < N; i++) {
                outData_acc[i] = data_acc[i];
            }
        });
    });
}

// Explicit instantiation
template void fwd_ntt_kernel<0>(sycl::queue& q,
                                sycl::buffer<uint64_t, 1>& data_buf,
                                sycl::buffer<uint64_t, 1>& twiddleFactors_buf,
                                sycl::buffer<uint64_t, 1>& modulus_buf,
                                sycl::buffer<uint64_t, 1>& outData_buf);
