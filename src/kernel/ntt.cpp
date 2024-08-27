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
void fwd_ntt_kernel(sycl::queue& q,
                    sycl::buffer<WideVecType, 1>& inData_buf,
                    sycl::buffer<unsigned32Bits_t, 1>& miniBatchSize_buf,
                    sycl::buffer<Wide64BytesType, 1>& twiddleFactors_buf,
                    sycl::buffer<Wide64BytesType, 1>& barrettTwiddleFactors_buf,
                    sycl::buffer<unsigned64Bits_t, 1>& modulus_buf,
                    sycl::buffer<WideVecType, 1>& outData_buf) {

    q.submit([&](sycl::handler& h) {
        // Create accessors for the buffers
        auto inData_acc = inData_buf.get_access<sycl::access::mode::read>(h);
        auto miniBatchSize_acc = miniBatchSize_buf.get_access<sycl::access::mode::read>(h);
        auto twiddleFactors_acc = twiddleFactors_buf.get_access<sycl::access::mode::read>(h);
        auto barrettTwiddleFactors_acc = barrettTwiddleFactors_buf.get_access<sycl::access::mode::read>(h);
        auto modulus_acc = modulus_buf.get_access<sycl::access::mode::read>(h);
        auto outData_acc = outData_buf.get_access<sycl::access::mode::write>(h);

        h.single_task<FWD_NTT<id>>([=]() [[intel::kernel_args_restrict]] {
            [[intel::fpga_memory("BLOCK_RAM")]] [[intel::numbanks(VEC)]] [[intel::max_replicates(2)]]
            unsigned long X[FPGA_NTT_SIZE / VEC][VEC];
            [[intel::fpga_memory("BLOCK_RAM")]] [[intel::numbanks(VEC)]] [[intel::max_replicates(2)]]
            unsigned long X2[FPGA_NTT_SIZE / VEC][VEC];
            [[intel::fpga_memory("BLOCK_RAM")]] [[intel::numbanks(VEC)]] [[intel::max_replicates(2)]]
            unsigned char Xm[FPGA_NTT_SIZE / VEC][VEC];

            unsigned long local_roots[FPGA_NTT_SIZE];
            unsigned long local_precons[FPGA_NTT_SIZE];

            constexpr int computeUnitID = id;
            constexpr size_t numTwiddlePerWord = sizeof(Wide64BytesType) / sizeof(unsigned64Bits_t);

            // Initialize Xm
            for (int i = 0; i < FPGA_NTT_SIZE / VEC; i++) {
#pragma unroll
                for (int j = 0; j < VEC; j++) {
                    Xm[i][j] = 0;
                }
            }

            // Load twiddle factors and precomputed values
            for (int i = 0; i < FPGA_NTT_SIZE / numTwiddlePerWord; i++) {
#pragma unroll
                for (size_t j = 0; j < numTwiddlePerWord; ++j) {
                    local_roots[i * numTwiddlePerWord + j] = twiddleFactors_acc[i * numTwiddlePerWord + j];
                    local_precons[i * numTwiddlePerWord + j] = barrettTwiddleFactors_acc[i * numTwiddlePerWord + j];
                }
            }

            unsigned64Bits_t modulus = modulus_acc[0];

            for (int mb = 0; mb < miniBatchSize_acc[0]; mb++) {
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
                            elements_in = inData_acc[s_index];
                            s_index++;
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
                        }

                        // NTT computation and reduction
#pragma unroll
                        for (int n = 0; n < VEC; n++) {
                            size_t i = (k * VEC + n) >> t_log;
                            size_t j = (k * VEC + n) & (t - 1);
                            size_t j1 = i * 2 * t;
                            const unsigned long W_op = local_roots[m + i];
                            const unsigned long W_precon = local_precons[m + i];

                            unsigned long tx = (Xm[(j1 + j) / VEC][(j1 + j) % VEC] == Xm_val - 1) ? curX[n] : curX2[n];
                            if (tx >= twice_mod) tx -= twice_mod;

                            unsigned long a = (Xm[(j1 + j) / VEC][(j1 + j) % VEC] == Xm_val - 1) ? curX[n + VEC] : curX2[n + VEC];
                            unsigned long b = W_precon;
                            unsigned long Q = W_op * a - ((HIGH(a * b, unsigned long) * coeff_mod) >> 32);

                            X[(j1 + j) / VEC][(j1 + j) % VEC] = tx + Q;
                            X2[(j1 + j + t) / VEC][(j1 + j + t) % VEC] = tx + twice_mod - Q;

                            if (m == (FPGA_NTT_SIZE / 2)) {
                                unsigned long val = tx + Q;
                                if (val >= twice_mod) val -= twice_mod;
                                if (val >= coeff_mod) val -= coeff_mod;
                                elements_out.data[n * 2] = val;

                                unsigned long val2 = tx + twice_mod - Q;
                                if (val2 >= twice_mod) val2 -= twice_mod;
                                if (val2 >= coeff_mod) val2 -= coeff_mod;
                                elements_out.data[n * 2 + 1] = val2;

                                outData_acc[s_index].data[n * 2] = val;
                                outData_acc[s_index].data[n * 2 + 1] = val2;
                            }
                        }
                    }

                    t >>= 1;
                    t_log -= 1;
                }
            }
        });
    });
}

// Implement the fwd_ntt function
void fwd_ntt(sycl::queue& q) {
    fwd_ntt_kernel<0>(q);
}

// Explicit template instantiation for fwd_ntt_kernel
template void fwd_ntt_kernel<0>(sycl::queue& q);

class FWD_NTT_INPUT;
class FWD_NTT_OUTPUT;
