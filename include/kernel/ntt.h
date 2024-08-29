#ifndef NTT_H
#define NTT_H

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#ifndef FPGA_NTT_SIZE
#define FPGA_NTT_SIZE 16384  // Example size
#endif

using namespace cl::sycl;

// Declare the FWD_NTT template class
template <size_t idx>
class FWD_NTT;

// Declare the NTT functions with buffer-based signatures
template <size_t id>
void fwd_ntt_kernel(sycl::queue& q,
                    buffer<uint64_t, 1>& inData_buf,
                    buffer<uint64_t, 1>& twiddleFactors_buf,
                    buffer<uint64_t, 1>& modulus_buf,
                    buffer<uint64_t, 1>& outData_buf);

void fwd_ntt(sycl::queue& q,
             sycl::buffer<uint64_t, 1>& inData_buf,
             sycl::buffer<uint64_t, 1>& twiddleFactors_buf,
             sycl::buffer<uint64_t, 1>& modulus_buf,
             sycl::buffer<uint64_t, 1>& outData_buf);

#endif // NTT_H
