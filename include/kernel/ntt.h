#ifndef NTT_H
#define NTT_H

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#ifndef FPGA_NTT_SIZE
#define FPGA_NTT_SIZE 16384  // Example size
#endif

using namespace cl::sycl;

template <size_t idx>
class FWD_NTT;

template <size_t id>
void fwd_ntt_kernel(sycl::queue& q,
                    buffer<uint64_t, 1>& data_buf,
                    buffer<uint64_t, 1>& twiddleFactors_buf,
                    buffer<uint64_t, 1>& modulus_buf,
                    buffer<uint64_t, 1>& outData_buf);

#endif // NTT_H
