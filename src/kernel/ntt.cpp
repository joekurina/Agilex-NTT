// ntt.cpp
#include "ntt.hpp"

namespace ntt {

// Inline Collect Roots
static inline void collect_roots(mul_op_t       w1[5],
                                 const uint64_t w[],
                                 const uint64_t w_con[],
                                 const size_t   m,
                                 const size_t   j)
{
  const uint64_t m1 = 2 * (m + j);
  w1[0].op          = w[m1];
  w1[1].op          = w[2 * m1];
  w1[2].op          = w[2 * m1 + 1];
  w1[3].op          = w[2 * m1 + 2];
  w1[4].op          = w[2 * m1 + 3];

  w1[0].con = w_con[m1];
  w1[1].con = w_con[2 * m1];
  w1[2].con = w_con[2 * m1 + 1];
  w1[3].con = w_con[2 * m1 + 2];
  w1[4].con = w_con[2 * m1 + 3];
}

// Inline Get Iteration Reminder
static inline uint64_t get_iter_reminder(const uint64_t N)
{
  if(HAS_AN_REM1_POWER(N)) {
    return 1;
  }
  if(HAS_AN_REM2_POWER(N)) {
    return 2;
  }
  if(HAS_AN_REM3_POWER(N)) {
    return 3;
  }
  return 0;
}

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
            const uint64_t N = data_buf.get_count();
            const uint64_t q = modulus_acc[0];
            const uint64_t bound_r4 = HAS_AN_EVEN_POWER(N) ? N : (N >> 1);
            mul_op_t roots[5];
            size_t t = N >> 2;

            // Radix-4 NTT computation
            // Main loop for the radix-4 NTT
            for(size_t m = 1; m < bound_r4; m <<= 2) {
                for(size_t j = 0; j < m; j++) {
                    const uint64_t k = 4 * t * j;

                    // Collect the roots
                    collect_roots(roots, twiddleFactors_acc.get_pointer(), twiddleFactors_acc.get_pointer(), m, j);
                    for(size_t i = k; i < k + t; i++) {
                        radix4_fwd_butterfly(&data_acc[i], &data_acc[i + t], &data_acc[i + 2 * t], &data_acc[i + 3 * t], roots, q);
                    }
                }
                t >>= 2;
            }

            // Check if N = 2^m where m is odd, perform extra radix-2 iteration if needed
            if (!HAS_AN_EVEN_POWER(N)) {
                for(size_t i = 0; i < N; i += 2) {
                    const mul_op_t w1 = {twiddleFactors_acc[N + i], twiddleFactors_acc[N + i]};
                    data_acc[i] = reduce_8q_to_4q(data_acc[i], q);
                    harvey_fwd_butterfly(&data_acc[i], &data_acc[i + 1], w1, q);
                }
            }

            // Write the final results back to the output buffer
            for(size_t i = 0; i < N; i++) {
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

}  // namespace ntt