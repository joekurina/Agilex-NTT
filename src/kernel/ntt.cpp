// ntt.cpp
#include "ntt.h"
#include "./utils/unroller.hpp"

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
                    local_roots[i * numTwiddlePerWord + j] = twiddleFactors_acc[i].data[j];
                    local_precons[i * numTwiddlePerWord + j] = barrettTwiddleFactors_acc[i].data[j];
                }
            }

            unsigned64Bits_t modulus = modulus_acc[0];
            size_t s_index = 0;  // Ensure proper indexing

            for (int mb = 0; mb < miniBatchSize_acc[0]; mb++) {
                unsigned64Bits_t coeff_mod = modulus;
                unsigned64Bits_t twice_mod = modulus << 1;
                unsigned64Bits_t t = (FPGA_NTT_SIZE >> 1);

                unsigned int t_log = FPGA_NTT_SIZE_LOG - 1;
                unsigned char Xm_val = 0;

                for (unsigned int m = 1; m < FPGA_NTT_SIZE; m <<= 1) {
                    Xm_val++;
                    [[intel::ivdep(X)]] [[intel::ivdep(X2)]] [[intel::ivdep(Xm)]]
                    for (unsigned int k = 0; k < FPGA_NTT_SIZE / 2 / VEC; k++) {
                        [[intel::fpga_register]] unsigned long curX[VEC * 2];
                        [[intel::fpga_register]] unsigned long curX2[VEC * 2];

                        size_t i0 = (k * VEC) >> t_log;
                        size_t j0 = (k * VEC) & (t - 1);
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

                            // Modular multiplication with reduction
                            unsigned long mul = W_op * a;
                            unsigned long Q = mul - ((HIGH(mul, unsigned long) * coeff_mod) >> 32);

                            X[(j1 + j) / VEC][(j1 + j) % VEC] = tx + Q;
                            X2[(j1 + j + t) / VEC][(j1 + j + t) % VEC] = tx + twice_mod - Q;

                            if (m == (FPGA_NTT_SIZE / 2)) {
                                unsigned long val = tx + Q;
                                if (val >= twice_mod) val -= twice_mod;
                                if (val >= coeff_mod) val -= coeff_mod;

                                unsigned long val2 = tx + twice_mod - Q;
                                if (val2 >= twice_mod) val2 -= twice_mod;
                                if (val2 >= coeff_mod) val2 -= coeff_mod;

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
void fwd_ntt(sycl::queue& q,
             sycl::buffer<WideVecType, 1>& inData_buf,
             sycl::buffer<unsigned32Bits_t, 1>& miniBatchSize_buf,
             sycl::buffer<Wide64BytesType, 1>& twiddleFactors_buf,
             sycl::buffer<Wide64BytesType, 1>& barrettTwiddleFactors_buf,
             sycl::buffer<unsigned64Bits_t, 1>& modulus_buf,
             sycl::buffer<WideVecType, 1>& outData_buf) {
    fwd_ntt_kernel<0>(q, inData_buf, miniBatchSize_buf, twiddleFactors_buf, barrettTwiddleFactors_buf, modulus_buf, outData_buf);
}

// Explicit template instantiation for fwd_ntt_kernel
template void fwd_ntt_kernel<0>(sycl::queue& q,
                                sycl::buffer<WideVecType, 1>& inData_buf,
                                sycl::buffer<unsigned32Bits_t, 1>& miniBatchSize_buf,
                                sycl::buffer<Wide64BytesType, 1>& twiddleFactors_buf,
                                sycl::buffer<Wide64BytesType, 1>& barrettTwiddleFactors_buf,
                                sycl::buffer<unsigned64Bits_t, 1>& modulus_buf,
                                sycl::buffer<WideVecType, 1>& outData_buf);
