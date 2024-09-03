#ifndef INTT_HPP
#define INTT_HPP

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

using namespace cl::sycl;

template <size_t idx>
class INTT;

template <size_t id>
void intt_kernel(sycl::queue& q,
                    buffer<uint64_t, 1>& data_buf,
                    buffer<uint64_t, 1>& twiddleFactors_buf,
                    buffer<uint64_t, 1>& modulus_buf,
                    buffer<uint64_t, 1>& outData_buf);

#endif // INTT_HPP
