#ifndef NTT_H
#define NTT_H

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#ifndef FPGA_NTT_SIZE
#define FPGA_NTT_SIZE 16384  // Example size
#endif

#if FPGA_NTT_SIZE == 32
#define FPGA_NTT_SIZE_LOG 5
#elif FPGA_NTT_SIZE == 1024
#define FPGA_NTT_SIZE_LOG 10
#elif FPGA_NTT_SIZE == 8192
#define FPGA_NTT_SIZE_LOG 13
#elif FPGA_NTT_SIZE == 16384
#define FPGA_NTT_SIZE_LOG 14
#elif FPGA_NTT_SIZE == 32768
#define FPGA_NTT_SIZE_LOG 15
#else
#error "Unsupported FPGA_NTT_SIZE"
#endif

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
