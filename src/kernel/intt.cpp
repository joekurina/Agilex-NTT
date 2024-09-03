// ntt.cpp
#include "ntt.hpp"

// Inverse NTT kernel using SYCL
template <size_t id>
void intt_kernel(sycl::queue& q,
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
            // Perform inverse NTT
        });
    });
}

// Explicit template instantiation for the kernel
template void intt_kernel<0>(sycl::queue& q,
                                sycl::buffer<uint64_t, 1>& data_buf,
                                sycl::buffer<uint64_t, 1>& twiddleFactors_buf,
                                sycl::buffer<uint64_t, 1>& modulus_buf,
                                sycl::buffer<uint64_t, 1>& outData_buf);
