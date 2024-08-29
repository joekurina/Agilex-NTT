// ntt.cpp
#include "ntt.h"

template <size_t idx>
class FWD_NTT;

template <size_t id>
void fwd_ntt_kernel(sycl::queue& q,
                    sycl::buffer<uint64_t, 1>& inData_buf,
                    sycl::buffer<uint64_t, 1>& twiddleFactors_buf,
                    sycl::buffer<uint64_t, 1>& modulus_buf,
                    sycl::buffer<uint64_t, 1>& outData_buf) {

    q.submit([&](sycl::handler& h) {
        // Create accessors for the buffers
        auto inData_acc = inData_buf.get_access<sycl::access::mode::read>(h);
        auto twiddleFactors_acc = twiddleFactors_buf.get_access<sycl::access::mode::read>(h);
        auto modulus_acc = modulus_buf.get_access<sycl::access::mode::read>(h);
        auto outData_acc = outData_buf.get_access<sycl::access::mode::write>(h);

        h.single_task<FWD_NTT<id>>([=]() [[intel::kernel_args_restrict]] {
            const size_t N = inData_acc.size();
            uint64_t modulus = modulus_acc[0];
            uint64_t twice_mod = modulus << 1;

            for (size_t i = 0; i < N; ++i) {
                uint64_t x = inData_acc[i];
                uint64_t w = twiddleFactors_acc[i];

                // NTT computation: simplified
                uint64_t tx = x * w % modulus;

                if (tx >= twice_mod) tx -= twice_mod;

                outData_acc[i] = tx;
            }
        });
    });
}

template void fwd_ntt_kernel<0>(sycl::queue& q,
                                sycl::buffer<uint64_t, 1>& inData_buf,
                                sycl::buffer<uint64_t, 1>& twiddleFactors_buf,
                                sycl::buffer<uint64_t, 1>& modulus_buf,
                                sycl::buffer<uint64_t, 1>& outData_buf);
