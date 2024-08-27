// ntt.cpp
#include "ntt.h"
#include "./utils/pipe_def_macros.hpp"
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

#define LHIGH(num, type) \
    ((type)(num) & ~((((type)1) << (sizeof(type) * 8 / 2)) - (type)1))
#define LOW(num, type) \
    ((type)(num) & ((((type)1) << (sizeof(type) * 8 / 2)) - (type)1))
#define HIGH(num, type) ((type)(num) >> (sizeof(type) * 8 / 2))

#define HEXL_FPGA_USE_64BIT_MULT
#ifdef HEXL_FPGA_USE_64BIT_MULT
#pragma clang diagnostic warning "Compiling with HEXL_FPGA_USE_64BIT_MULT"
#endif

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

#if 32 == FPGA_NTT_SIZE
#define FPGA_NTT_SIZE_LOG 5
#elif 1024 == FPGA_NTT_SIZE
#define FPGA_NTT_SIZE_LOG 10
#elif 8192 == FPGA_NTT_SIZE
#define FPGA_NTT_SIZE_LOG 13
#elif 16384 == FPGA_NTT_SIZE
#define FPGA_NTT_SIZE_LOG 14
#elif 32768 == FPGA_NTT_SIZE
#define FPGA_NTT_SIZE_LOG 15
#endif

template <size_t idx>
class FWD_NTT;

// Implement the kernel function
template <size_t id>
void fwd_ntt_kernel(sycl::queue& q) {
    q.submit([&](sycl::handler& h) {
        h.single_task<FWD_NTT<id>>([=]() [[intel::kernel_args_restrict]] {

            // Internal memory allocations, replicated for each compute unit
            [[intel::fpga_memory("BLOCK_RAM")]] [[intel::numbanks(VEC)]] 
            [[intel::max_replicates(2)]] unsigned long X[FPGA_NTT_SIZE / VEC][VEC];
            
            [[intel::fpga_memory("BLOCK_RAM")]] [[intel::numbanks(VEC)]] 
            [[intel::max_replicates(2)]] unsigned long X2[FPGA_NTT_SIZE / VEC][VEC];
            
            [[intel::fpga_memory("BLOCK_RAM")]] [[intel::numbanks(VEC)]] 
            [[intel::max_replicates(2)]] unsigned char Xm[FPGA_NTT_SIZE / VEC][VEC];

            unsigned long local_roots[FPGA_NTT_SIZE];
            unsigned long local_precons[FPGA_NTT_SIZE];

            constexpr int computeUnitID = id;
            constexpr size_t numTwiddlePerWord = sizeof(Wide64BytesType) / sizeof(unsigned64Bits_t);

            for (int i = 0; i < FPGA_NTT_SIZE / VEC; i++) {
#pragma unroll
                for (int j = 0; j < VEC; j++) {
                    Xm[i][j] = 0;
                }
            }

            while (true) {
                unsigned32Bits_t miniBatchSize = miniBatchSizePipeNTT::PipeAt<computeUnitID>::read();

                for (int i = 0; i < FPGA_NTT_SIZE / numTwiddlePerWord; i++) {
                    Wide64BytesType vecTwiddle = twiddleFactorsPipe::PipeAt<computeUnitID>::read();
#pragma unroll
                    for (size_t j = 0; j < numTwiddlePerWord; ++j) {
                        local_roots[i * numTwiddlePerWord + j] = vecTwiddle.data[j];
                    }
                }

                for (int i = 0; i < FPGA_NTT_SIZE / numTwiddlePerWord; i++) {
                    Wide64BytesType vecExponent = barrettTwiddleFactorsPipe::PipeAt<computeUnitID>::read();
#pragma unroll
                    for (size_t j = 0; j < numTwiddlePerWord; ++j) {
                        local_precons[i * numTwiddlePerWord + j] = vecExponent.data[j];
                    }
                }

                unsigned64Bits_t modulus = modulusPipe::PipeAt<computeUnitID>::read();

                for (int mb = 0; mb < miniBatchSize; mb++) {
                    unsigned64Bits_t coeff_mod = modulus;
                    unsigned64Bits_t twice_mod = modulus << 1;
                    unsigned64Bits_t t = (FPGA_NTT_SIZE >> 1);

                    unsigned int t_log = FPGA_NTT_SIZE_LOG - 1;
                    unsigned char Xm_val = 0;
                    size_t s_index = 0;

                    for (unsigned int m = 1; m < FPGA_NTT_SIZE; m <<= 1) {
                        Xm_val++;
                        [[intel::ivdep(X)]] [[intel::ivdep(X2)]] [[intel::ivdep(Xm)]]
                        for (unsigned int k = 0; k < FPGA_NTT_SIZE / 2 / VEC; k++) {
                            [[intel::fpga_register]] unsigned long curX[VEC * 2];
                            [[intel::fpga_register]] unsigned long curX2[VEC * 2];
                            [[intel::fpga_register]] unsigned long curX_rep[VEC * 2];
                            [[intel::fpga_register]] unsigned long curX2_rep[VEC * 2];

                            size_t i0 = (k * VEC + 0) >> t_log;
                            size_t j0 = (k * VEC + 0) & (t - 1);
                            size_t j10 = i0 * 2 * t;

                            bool b_same_vec = ((j10 + j0) / VEC) == ((j10 + j0 + t) / VEC);
                            size_t X_ind = (j10 + j0) / VEC;
                            size_t Xt_ind = (j10 + j0 + t) / VEC + b_same_vec;

                            WideVecType elements_in;
                            if (m == 1) {
                                elements_in = inDataPipe::PipeAt<computeUnitID>::read();
                            }

#pragma unroll
                            for (int n = 0; n < VEC; n++) {
                                size_t i = (k * VEC + n) >> t_log;
                                size_t j = (k * VEC + n) & (t - 1);
                                size_t j1 = i * 2 * t;
                                if (m == 1) {
                                    curX[n] = elements_in.data[n];
                                    curX[n + VEC] = elements_in.data[VEC + n];
                                } else {
                                    curX[n] = X[X_ind][n];
                                    curX[n + VEC] = X[Xt_ind][n];
                                    curX2[n] = X2[X_ind][n];
                                    curX2[n + VEC] = X2[Xt_ind][n];
                                }
                            }

                            WideVecType elements_out;
                            if (t == 1) {
#pragma unroll
                                for (int n = 0; n < VEC; n++) {
                                    const int cur_t = 1;
                                    const int Xn = n / cur_t * (2 * cur_t) + n % cur_t;
                                    const int Xnt = Xn + ((cur_t < VEC) ? cur_t : VEC);
                                    curX_rep[n] = curX[Xn];
                                    curX2_rep[n] = curX2[Xn];
                                    curX_rep[VEC + n] = curX[Xnt];
                                    curX2_rep[VEC + n] = curX2[Xnt];
                                }
#if VEC >= 4
                            } else if (t == 2) {
#pragma unroll
                                for (int n = 0; n < VEC; n++) {
                                    const int cur_t = 2;
                                    const int Xn = n / cur_t * (2 * cur_t) + n % cur_t;
                                    const int Xnt = Xn + ((cur_t < VEC) ? cur_t : VEC);
                                    curX_rep[n] = curX[Xn];
                                    curX2_rep[n] = curX2[Xn];
                                    curX_rep[VEC + n] = curX[Xnt];
                                    curX2_rep[VEC + n] = curX2[Xnt];
                                }
#endif
#if VEC >= 8
                            } else if (t == 4) {
#pragma unroll
                                for (int n = 0; n < VEC; n++) {
                                    const int cur_t = 4;
                                    const int Xn = n / cur_t * (2 * cur_t) + n % cur_t;
                                    const int Xnt = Xn + ((cur_t < VEC) ? cur_t : VEC);
                                    curX_rep[n] = curX[Xn];
                                    curX2_rep[n] = curX2[Xn];
                                    curX_rep[VEC + n] = curX[Xnt];
                                    curX2_rep[VEC + n] = curX2[Xnt];
                                }
#endif
#if VEC >= 16
                            } else if (t == 8) {
#pragma unroll
                                for (int n = 0; n < VEC; n++) {
                                    const int cur_t = 8;
                                    const int Xn = n / cur_t * (2 * cur_t) + n % cur_t;
                                    const int Xnt = Xn + ((cur_t < VEC) ? cur_t : VEC);
                                    curX_rep[n] = curX[Xn];
                                    curX2_rep[n] = curX2[Xn];
                                    curX_rep[VEC + n] = curX[Xnt];
                                    curX2_rep[VEC + n] = curX2[Xnt];
                                }
#endif
#if VEC >= 32
                            } else if (t == 16) {
#pragma unroll
                                for (int n = 0; n < VEC; n++) {
                                    const int cur_t = 16;
                                    const int Xn = n / cur_t * (2 * cur_t) + n % cur_t;
                                    const int Xnt = Xn + ((cur_t < VEC) ? cur_t : VEC);
                                    curX_rep[n] = curX[Xn];
                                    curX2_rep[n] = curX2[Xn];
                                    curX_rep[VEC + n] = curX[Xnt];
                                    curX2_rep[VEC + n] = curX2[Xnt];
                                }
#endif
                            } else {
#pragma unroll
                                for (int n = 0; n < VEC; n++) {
                                    curX_rep[n] = curX[n];
                                    curX2_rep[n] = curX2[n];
                                    curX_rep[VEC + n] = curX[VEC + n];
                                    curX2_rep[VEC + n] = curX2[VEC + n];
                                }
                            }

                            if (m == (FPGA_NTT_SIZE / 2)) {
                                s_index = k * (VEC * 2);
                                outDataPipe::PipeAt<computeUnitID>::write(elements_out);
                            }

#pragma unroll
                            for (int n = 0; n < VEC; n++) {
#if REORDER
                                X[X_ind][n] = curX_rep[n];
                                X2[Xt_ind][n] = curX_rep[n + VEC];
                                Xm[X_ind][n] = Xm_val;
#endif
                            }
                        }

                        t >>= 1;
                        t_log -= 1;
                    }
                }
            }
        });
    });
}

void ntt_input_kernel(buffer<uint64_t, 1>& inData_buf,
                      buffer<uint64_t, 1>& inData2_buf,
                      buffer<uint64_t, 1>& modulus_buf,
                      buffer<uint64_t, 1>& twiddleFactors_buf,
                      buffer<uint64_t, 1>& barrettTwiddleFactors_buf,
                      unsigned int numFrames,
                      sycl::queue& q) {
    
    q.submit([&](sycl::handler& h) {
        // Create accessors for the buffers
        auto inData_acc = inData_buf.get_access<access::mode::read>(h);
        auto inData2_acc = inData2_buf.get_access<access::mode::read>(h);
        auto modulus_acc = modulus_buf.get_access<access::mode::read>(h);
        auto twiddleFactors_acc = twiddleFactors_buf.get_access<access::mode::read>(h);
        auto barrettTwiddleFactors_acc = barrettTwiddleFactors_buf.get_access<access::mode::read>(h);

        h.single_task([=]() {
            // Broadcast miniBatchSize to each NTT autorun kernel instance
            Unroller<0, NUM_NTT_COMPUTE_UNITS>::Step([&](auto i) {
                unsigned32Bits_t fractionalMiniBatch =
                    (numFrames % NUM_NTT_COMPUTE_UNITS) / (i + 1);
                if (fractionalMiniBatch > 0)
                    fractionalMiniBatch = 1;
                else
                    fractionalMiniBatch = 0;
                unsigned32Bits_t miniBatchSize =
                    (numFrames / NUM_NTT_COMPUTE_UNITS) + fractionalMiniBatch;
                miniBatchSizePipeNTT::PipeAt<i>::write(miniBatchSize);
            });

            // Assuming the twiddle factors and the complex root of unity are similar
            // distribute roots of unity to each kernel
            constexpr size_t numTwiddlePerWord =
                sizeof(Wide64BytesType) / sizeof(unsigned64Bits_t);
            constexpr unsigned int iterations = FPGA_NTT_SIZE / numTwiddlePerWord;

            for (size_t i = 0; i < iterations; i++) {
                Wide64BytesType tw;
#pragma unroll
                for (size_t j = 0; j < numTwiddlePerWord; j++) {
                    tw.data[j] = twiddleFactors_acc[i * numTwiddlePerWord + j];
                }

                // Broadcast twiddles to all compute units
                Unroller<0, NUM_NTT_COMPUTE_UNITS>::Step(
                    [&](auto c) { twiddleFactorsPipe::PipeAt<c>::write(tw); });
            }

            for (size_t i = 0; i < FPGA_NTT_SIZE / numTwiddlePerWord; i++) {
                Wide64BytesType tw;
#pragma unroll
                for (size_t j = 0; j < numTwiddlePerWord; j++) {
                    tw.data[j] = barrettTwiddleFactors_acc[i * numTwiddlePerWord + j];
                }

                // Broadcast twiddles to all compute units
                Unroller<0, NUM_NTT_COMPUTE_UNITS>::Step(
                    [&](auto c) { barrettTwiddleFactorsPipe::PipeAt<c>::write(tw); });
            }

            // Broadcast modulus to all the kernels
            unsigned64Bits_t mod = modulus_acc[0];
            Unroller<0, NUM_NTT_COMPUTE_UNITS>::Step(
                [&](auto c) { modulusPipe::PipeAt<c>::write(mod); });

            ////////////////////////////////////////////////////////////////////////////////////
            // Retrieve one NTT data and stream to different kernels, per iteration for
            // top-level loop

            constexpr unsigned int numElementsInVec =
                sizeof(WideVecType) / sizeof(unsigned64Bits_t);
            Unroller<0, NUM_NTT_COMPUTE_UNITS>::Step([&](auto computeUnitID) {
                for (unsigned int b = 0; b < numFrames; b++) {
                    if (b % NUM_NTT_COMPUTE_UNITS == computeUnitID) {
                        for (size_t i = 0; i < FPGA_NTT_SIZE / numElementsInVec; i++) {
                            WideVecType inVec;
                            unsigned long offset = b * FPGA_NTT_SIZE + i * VEC;
#pragma unroll
                            for (size_t j = 0; j < VEC; j++) {
                                inVec.data[j] = inData_acc[offset + j];
                                inVec.data[j + VEC] =
                                    inData2_acc[offset + FPGA_NTT_SIZE / 2 + j];
                            }
                            inDataPipe::PipeAt<computeUnitID>::write(inVec);
                        }
                    }
                }
            });
        });
    });
}

// Implement the ntt_output_kernel function
void ntt_output_kernel(buffer<uint64_t, 1>& outData_buf,
                       int numFrames,
                       sycl::queue& q) {
    
    q.submit([&](sycl::handler& h) {
        // Create accessor for the output buffer
        auto outData_acc = outData_buf.get_access<access::mode::write>(h);

        h.single_task([=]() {
            constexpr unsigned int numElementsInVec =
                sizeof(WideVecType) / sizeof(unsigned64Bits_t);

            Unroller<0, NUM_NTT_COMPUTE_UNITS>::Step([&](auto computeUnitID) {
                for (size_t b = 0; b < numFrames; b++) {
                    if (b % NUM_NTT_COMPUTE_UNITS == computeUnitID) {
                        for (size_t i = 0; i < FPGA_NTT_SIZE / numElementsInVec; i++) {
                            WideVecType oVec =
                                outDataPipe::PipeAt<computeUnitID>::read();
                            unsigned long offset =
                                b * FPGA_NTT_SIZE + i * numElementsInVec;
#pragma unroll
                            for (size_t j = 0; j < numElementsInVec; j++) {
                                outData_acc[offset + j] = oVec.data[j];
                            }
                        }
                    }
                }
            });
        });
    });
}

// Implement the fwd_ntt function
void fwd_ntt(sycl::queue& q,
             buffer<uint64_t, 1>& inData_buf,
             buffer<uint64_t, 1>& inData2_buf,
             buffer<uint64_t, 1>& modulus_buf,
             buffer<uint64_t, 1>& twiddleFactors_buf,
             buffer<uint64_t, 1>& barrettTwiddleFactors_buf,
             buffer<uint64_t, 1>& outData_buf) {
    fwd_ntt_kernel<0>(q, inData_buf, inData2_buf, modulus_buf, 
                      twiddleFactors_buf, barrettTwiddleFactors_buf, outData_buf);
}

// Explicit template instantiation for fwd_ntt_kernel
template void fwd_ntt_kernel<0>(sycl::queue& q,
                                buffer<uint64_t, 1>& inData_buf,
                                buffer<uint64_t, 1>& inData2_buf,
                                buffer<uint64_t, 1>& modulus_buf,
                                buffer<uint64_t, 1>& twiddleFactors_buf,
                                buffer<uint64_t, 1>& barrettTwiddleFactors_buf,
                                buffer<uint64_t, 1>& outData_buf);

class FWD_NTT_INPUT;
class FWD_NTT_OUTPUT;
