// ntt.h
#ifndef NTT_H
#define NTT_H

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

using namespace cl::sycl;

// Declare the FWD_NTT template class
template <size_t idx>
class FWD_NTT;

// Declare the NTT functions
template <size_t id>
void fwd_ntt_kernel(sycl::queue& q);

void ntt_input_kernel(unsigned int numFrames, uint64_t* k_inData,
                      uint64_t* k_inData2, uint64_t* k_modulus,
                      uint64_t* k_twiddleFactors,
                      uint64_t* k_barrettTwiddleFactors);

void ntt_output_kernel(int numFrames, uint64_t* k_outData);

// Declare the interface functions for external C linkage
extern "C" {
    void fwd_ntt(sycl::queue& q);
    sycl::event ntt_input(sycl::queue& q, unsigned int numFrames, uint64_t* inData,
                          uint64_t* inData2, uint64_t* modulus,
                          uint64_t* twiddleFactors,
                          uint64_t* barrettTwiddleFactors);
    sycl::event ntt_output(sycl::queue& q, int numFrames, uint64_t* outData_in_svm);
}

#endif // NTT_H
