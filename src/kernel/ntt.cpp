// ntt.cpp
#include "ntt.h"

// Define the WORD_SIZE if not already defined
#ifndef WORD_SIZE
#define WORD_SIZE (sizeof(uint64_t) * 8)  // WORD_SIZE is the number of bits in uint64_t
#endif

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
            const uint64_t N = data_acc.size();
            const uint64_t q = modulus_acc[0];
            const uint64_t q2 = q << 1;
            const uint64_t q4 = q << 2;
            const uint64_t bound_r4 = (N & (N - 1)) == 0 ? N : (N >> 1); // N is a power of 2
            size_t t = N >> 2;

            // Radix-4 NTT computation
            for (size_t m = 1; m < bound_r4; m <<= 2) {
                for (size_t j = 0; j < m; j++) {
                    const uint64_t k = 4 * t * j;

                    // Collect roots manually
                    const uint64_t m1 = 2 * (m + j);
                    mul_op_t roots[5];
                    roots[0].op  = twiddleFactors_acc[m1];
                    roots[1].op  = twiddleFactors_acc[2 * m1];
                    roots[2].op  = twiddleFactors_acc[2 * m1 + 1];
                    roots[3].op  = twiddleFactors_acc[2 * m1 + 2];
                    roots[4].op  = twiddleFactors_acc[2 * m1 + 3];

                    roots[0].con = twiddleFactors_acc[N + m1];
                    roots[1].con = twiddleFactors_acc[N + 2 * m1];
                    roots[2].con = twiddleFactors_acc[N + 2 * m1 + 1];
                    roots[3].con = twiddleFactors_acc[N + 2 * m1 + 2];
                    roots[4].con = twiddleFactors_acc[N + 2 * m1 + 3];

                    for (size_t i = k; i < k + t; i++) {
                        // Inline radix-4 forward butterfly computation
                        uint64_t* X = &data_acc[i];
                        uint64_t* Y = &data_acc[i + t];
                        uint64_t* Z = &data_acc[i + 2 * t];
                        uint64_t* T = &data_acc[i + 3 * t];

                        // Inline fast_dbl_mul_mod_q2 logic
                        const uint64_t Q1 = (roots[1].con * *Y + roots[2].con * *T) >> WORD_SIZE;
                        const uint64_t Y1 = (*Y * roots[1].op) + (*T * roots[2].op) - (Q1 * q);

                        const uint64_t Q2 = (roots[3].con * *Y + roots[4].con * *T) >> WORD_SIZE;
                        const uint64_t Y2 = (*Y * roots[3].op) + (*T * roots[4].op) - (Q2 * q);

                        // Inline reduce_8q_to_4q logic
                        uint64_t T1 = (*X < q4) ? *X : *X - q4;
                        const uint64_t T2 = (*Z * roots[0].op) - (((roots[0].con * *Z) >> WORD_SIZE) * q);

                        *X = (T1 + T2 + Y1);
                        *Y = (T1 + T2 - Y1) + q2;
                        *Z = (T1 - T2 + Y2) + q2;
                        *T = (T1 - T2 - Y2) + q4;
                    }
                }
                t >>= 2;
            }

            // Check whether N=2^m where m is odd. If not, perform extra radix-2 iteration
            if ((N & (N - 1)) != 0) {
                for (size_t i = 0; i < N; i += 2) {
                    const mul_op_t w1 = {twiddleFactors_acc[N + i], twiddleFactors_acc[N + i + 1]};
                    
                    // Inline reduce_8q_to_4q logic
                    uint64_t X1 = (data_acc[i] < q4) ? data_acc[i] : data_acc[i] - q4;

                    // Inline harvey_fwd_butterfly logic
                    const uint64_t Q = (w1.con * data_acc[i + 1]) >> WORD_SIZE;
                    const uint64_t T = (data_acc[i + 1] * w1.op) - (Q * q);

                    data_acc[i] = X1 + T;
                    data_acc[i + 1] = X1 - T + q2;
                }
            }

            // Copy the results to the output buffer
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
