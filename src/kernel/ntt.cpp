// ntt.cpp
#include "ntt.h"
#include "./utils/pipe_def_marcos.hpp"
#include "./utils/pipe_array.hpp"
#include "./utils/unroller.hpp"

// Define the necessary constants and types
#ifndef NUM_NTT_COMPUTE_UNITS
#define NUM_NTT_COMPUTE_UNITS 1
#else
#pragma clang diagnostic warning "Compiling with external NUM_NTT_COMPUTE_UNITS"
#endif

#ifndef VEC
#define VEC 8
#endif
#define REORDER 1
#define PRINT_ROW_RESULT 0

#ifndef FPGA_NTT_SIZE
#define FPGA_NTT_SIZE 16384
#else
#pragma clang diagnostic warning "Compiling with external FPGA_NTT_SIZE"
#endif

#define HEXL_FPGA_USE_64BIT_MULT

typedef uint64_t unsigned64Bits_t;
typedef unsigned int unsigned32Bits_t;
typedef struct {
    unsigned64Bits_t data[VEC * 2];
} elements_in_t;

typedef struct {
    unsigned64Bits_t data[VEC * 2];
} elements_out_t;

typedef struct {
    unsigned64Bits_t data[VEC * 2];
} WideVecType;

typedef struct {
    unsigned64Bits_t data[64 / sizeof(unsigned64Bits_t)];
} Wide64BytesType;

// Define the pipes
defPipe1d(inDataPipe, WideVecType, 16, NUM_NTT_COMPUTE_UNITS);
defPipe1d(miniBatchSizePipeNTT, unsigned32Bits_t, 16, NUM_NTT_COMPUTE_UNITS);
defPipe1d(outDataPipe, WideVecType, 16, NUM_NTT_COMPUTE_UNITS);
defPipe1d(modulusPipe, unsigned64Bits_t, 16, NUM_NTT_COMPUTE_UNITS);
defPipe1d(twiddleFactorsPipe, Wide64BytesType, 16, NUM_NTT_COMPUTE_UNITS);
defPipe1d(barrettTwiddleFactorsPipe, Wide64BytesType, 16, NUM_NTT_COMPUTE_UNITS);

// Implement the kernel function
template <size_t id>
void fwd_ntt_kernel(sycl::queue& q) {
    q.submit([&](sycl::handler& h) {
        h.single_task<FWD_NTT<id>>([=]() [[intel::kernel_args_restrict]] {
            // The full implementation of your NTT kernel goes here
            // (use the provided kernel code)
        });
    });
}

// Implement the ntt_input_kernel function
void ntt_input_kernel(unsigned int numFrames, uint64_t* k_inData,
                      uint64_t* k_inData2, uint64_t* k_modulus,
                      uint64_t* k_twiddleFactors,
                      uint64_t* k_barrettTwiddleFactors) {
    // The implementation of your ntt_input_kernel goes here
    // (use the provided kernel code)
}

// Implement the ntt_output_kernel function
void ntt_output_kernel(int numFrames, uint64_t* k_outData) {
    // The implementation of your ntt_output_kernel goes here
    // (use the provided kernel code)
}

// Implement the C interface functions
extern "C" {
    void fwd_ntt(sycl::queue& q) {
        Unroller<0, NUM_NTT_COMPUTE_UNITS>::Step(
            [&](auto idx) { fwd_ntt_kernel<idx>(q); });
    }

    sycl::event ntt_input(sycl::queue& q, unsigned int numFrames, uint64_t* inData,
                          uint64_t* inData2, uint64_t* modulus,
                          uint64_t* twiddleFactors,
                          uint64_t* barrettTwiddleFactors) {
        return q.submit([&](sycl::handler& h) {
            h.single_task<FWD_NTT_INPUT>([=]() [[intel::kernel_args_restrict]] {
                ntt_input_kernel(numFrames, inData, inData2, modulus,
                                 twiddleFactors, barrettTwiddleFactors);
            });
        });
    }

    sycl::event ntt_output(sycl::queue& q, int numFrames,
                           uint64_t* outData_in_svm) {
        return q.submit([&](sycl::handler& h) {
            h.single_task<FWD_NTT_OUTPUT>([=]() [[intel::kernel_args_restrict]] {
                ntt_output_kernel(numFrames, outData_in_svm);
            });
        });
    }
}

// Explicit instantiation of the template function for each compute unit
template void fwd_ntt_kernel<0>(sycl::queue& q);
template void fwd_ntt_kernel<1>(sycl::queue& q);
template void fwd_ntt_kernel<2>(sycl::queue& q);
template void fwd_ntt_kernel<3>(sycl::queue& q);
