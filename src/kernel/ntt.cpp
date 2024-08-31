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
                const size_t N = data_acc.size();
                const uint64_t modulus = modulus_acc[0];

                // Cooley-Tukey NTT (bit-reverse input to standard order)
                for (size_t t = 1; t < N; t <<= 1) {
                    size_t m = t << 1;

                    // Handle the case where j=0, w_t^0 = 1
                    for (size_t s = tid; s < N; s += item.get_local_range(0)) {
                        if (s % m < t) {
                            uint64_t x = data_acc[s + t];
                            data_acc[s + t] = (data_acc[s] + modulus - x) % modulus;
                            data_acc[s] = (data_acc[s] + x) % modulus;
                        }
                    }

                    item.barrier(sycl::access::fence_space::local_space);

                    // General case: j > 0
                    for (size_t j = 1; j < t; j++) {
                        uint64_t w = twiddleFactors_acc[t + j];

                        for (size_t s = tid; s < N; s += item.get_local_range(0)) {
                            if (s % m == j) {
                                uint64_t x = (data_acc[s + t] * w) % modulus;
                                data_acc[s + t] = (data_acc[s] + modulus - x) % modulus;
                                data_acc[s] = (data_acc[s] + x) % modulus;
                            }
                        }
                        item.barrier(sycl::access::fence_space::local_space);
                    }
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
