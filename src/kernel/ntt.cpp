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

        h.single_task<FWD_NTT<id>>([=]() {
            const size_t N = inData_buf.get_count();
            uint64_t modulus = modulus_acc[0];

            // Bit-reversal permutation
            for (size_t i = 0; i < N; i++) {
                size_t j = 0;
                for (size_t k = 0; k < __builtin_ctz(N); k++) {
                    j = (j << 1) | ((i >> k) & 1);
                }
                if (i < j) {
                    std::swap(outData_acc[i], outData_acc[j]);
                }
            }

            // NTT computation
            for (size_t len = 2; len <= N; len <<= 1) {
                uint64_t wlen = twiddleFactors_acc[N / len];
                for (size_t i = 0; i < N; i += len) {
                    uint64_t w = 1;
                    for (size_t j = 0; j < len / 2; j++) {
                        uint64_t u = outData_acc[i + j];
                        uint64_t v = (outData_acc[i + j + len / 2] * w) % modulus;

                        outData_acc[i + j] = (u + v) % modulus;
                        outData_acc[i + j + len / 2] = (u + modulus - v) % modulus;

                        w = (w * wlen) % modulus;
                    }
                }
            }
        });
    }).wait(); // Wait for kernel to finish execution
}

void fwd_ntt(sycl::queue& q,
             sycl::buffer<uint64_t, 1>& inData_buf,
             sycl::buffer<uint64_t, 1>& twiddleFactors_buf,
             sycl::buffer<uint64_t, 1>& modulus_buf,
             sycl::buffer<uint64_t, 1>& outData_buf) {
    fwd_ntt_kernel<0>(q, inData_buf, twiddleFactors_buf, modulus_buf, outData_buf);
}

// Explicit template instantiation for fwd_ntt_kernel
template void fwd_ntt_kernel<0>(sycl::queue& q,
                                sycl::buffer<uint64_t, 1>& inData_buf,
                                sycl::buffer<uint64_t, 1>& twiddleFactors_buf,
                                sycl::buffer<uint64_t, 1>& modulus_buf,
                                sycl::buffer<uint64_t, 1>& outData_buf);
