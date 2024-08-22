#ifndef NTT_H
#define NTT_H

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

using namespace cl::sycl;

// Declare the FWD_NTT template class
template <size_t idx>
class FWD_NTT;

// Declare the NTT functions with buffer-based signatures
template <size_t id>
void fwd_ntt_kernel(sycl::queue& q,
                    buffer<uint64_t, 1>& inData_buf,
                    buffer<uint64_t, 1>& inData2_buf,
                    buffer<uint64_t, 1>& modulus_buf,
                    buffer<uint64_t, 1>& twiddleFactors_buf,
                    buffer<uint64_t, 1>& barrettTwiddleFactors_buf,
                    buffer<uint64_t, 1>& outData_buf);

void ntt_input_kernel(buffer<uint64_t, 1>& inData_buf,
                      buffer<uint64_t, 1>& inData2_buf,
                      buffer<uint64_t, 1>& modulus_buf,
                      buffer<uint64_t, 1>& twiddleFactors_buf,
                      buffer<uint64_t, 1>& barrettTwiddleFactors_buf,
                      unsigned int numFrames,
                      sycl::queue& q);

void ntt_output_kernel(buffer<uint64_t, 1>& outData_buf,
                       int numFrames,
                       sycl::queue& q);

// Declare the interface functions
void fwd_ntt(sycl::queue& q,
             buffer<uint64_t, 1>& inData_buf,
             buffer<uint64_t, 1>& inData2_buf,
             buffer<uint64_t, 1>& modulus_buf,
             buffer<uint64_t, 1>& twiddleFactors_buf,
             buffer<uint64_t, 1>& barrettTwiddleFactors_buf,
             buffer<uint64_t, 1>& outData_buf);

sycl::event ntt_input(sycl::queue& q, unsigned int numFrames,
                      buffer<uint64_t, 1>& inData_buf,
                      buffer<uint64_t, 1>& inData2_buf,
                      buffer<uint64_t, 1>& modulus_buf,
                      buffer<uint64_t, 1>& twiddleFactors_buf,
                      buffer<uint64_t, 1>& barrettTwiddleFactors_buf);

sycl::event ntt_output(sycl::queue& q, int numFrames,
                       buffer<uint64_t, 1>& outData_buf);

#endif // NTT_H
