#ifndef NTT_H
#define NTT_H

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#ifndef FPGA_NTT_SIZE
#define FPGA_NTT_SIZE 16384  // Example size
#endif

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

using namespace cl::sycl;

// Declare the FWD_NTT template class
template <size_t idx>
class FWD_NTT;

// Declare the NTT functions with buffer-based signatures
template <size_t id>
void fwd_ntt_kernel(sycl::queue& q,
                    buffer<WideVecType, 1>& inData_buf,
                    buffer<unsigned32Bits_t, 1>& miniBatchSize_buf,
                    buffer<Wide64BytesType, 1>& twiddleFactors_buf,
                    buffer<Wide64BytesType, 1>& barrettTwiddleFactors_buf,
                    buffer<unsigned64Bits_t, 1>& modulus_buf,
                    buffer<WideVecType, 1>& outData_buf);

#endif // NTT_H