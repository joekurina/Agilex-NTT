// main.cpp

#include "kernel/ntt.h"
#include "naive_ntt.h" // Include the naive NTT implementation header
#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <dpc_common.hpp>
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>

#define FPGA_NTT_SIZE 16384 // FPGA_NTT_SIZE must be a power of 2

using namespace cl::sycl;

// Helper function to compute (base^exp) % mod
unsigned long powmod(unsigned long base, unsigned long exp, unsigned long mod) {
    unsigned long result = 1;
    base = base % mod;
    while (exp > 0) {
        if (exp % 2 == 1)
            result = (result * base) % mod;
        exp = exp >> 1;
        base = (base * base) % mod;
    }
    return result;
}

// Function to compare two arrays
bool compare_arrays(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            std::cout << "Mismatch at index " << i << ": " << a[i] << " != " << b[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    // Create a device selector
#if defined(FPGA_EMULATOR)
    sycl::ext::intel::fpga_emulator_selector device_selector;
#else
    sycl::ext::intel::fpga_selector device_selector;
#endif

    // Create a SYCL queue
    sycl::queue q(device_selector, dpc_common::exception_handler);

    const size_t dataSize = FPGA_NTT_SIZE;
    const int32_t q_modulus = 65537;  // Example modulus
    const uint16_t primitive_root = 3;  // Assume 3 is a primitive root mod q

    std::cout << "Initializing buffers..." << std::endl;

    // Create buffers for input, twiddle factors, and output data
    buffer<int32_t, 1> inData_buf(dataSize);
    buffer<uint16_t, 1> twiddleFactors_buf(dataSize);
    buffer<int32_t, 1> modulus_buf(1);
    buffer<int32_t, 1> outData_buf(dataSize);

    std::cout << "Initializing input data and twiddle factors..." << std::endl;

    // Open a file to save the generated input data
    std::ofstream input_file("generated_inputs.txt");

    // Vector to store the input data for comparison later
    std::vector<int32_t> input_data(dataSize);
    std::vector<uint16_t> twiddle_factors(dataSize);
    std::vector<int32_t> output_data(dataSize);

    {
        host_accessor inData_acc(inData_buf, write_only);
        host_accessor twiddleFactors_acc(twiddleFactors_buf, write_only);
        host_accessor modulus_acc(modulus_buf, write_only);

        if (input_file.is_open()) {
            for (size_t i = 0; i < dataSize; ++i) {
                inData_acc[i] = i;
                input_data[i] = i;
                input_file << inData_acc[i] << std::endl;
            }

            for (size_t i = 0; i < dataSize; ++i) {
                twiddleFactors_acc[i] = powmod(primitive_root, i, q_modulus);
                twiddle_factors[i] = twiddleFactors_acc[i];
            }

            modulus_acc[0] = q_modulus;
            input_file << "Modulus: " << modulus_acc[0] << std::endl;

            input_file.close();
            std::cout << "Input data saved to generated_inputs.txt." << std::endl;
        } else {
            std::cerr << "Unable to open file for writing input data." << std::endl;
        }
    }

    std::cout << "Calling forward NTT kernel..." << std::endl;
    // Call the NTT function
    fwd_ntt_kernel<0>(q, inData_buf, twiddleFactors_buf, modulus_buf, outData_buf);
    std::cout << "Forward NTT kernel completed." << std::endl;

    std::cout << "Waiting for output event..." << std::endl;
    q.wait();
    std::cout << "Output event completed." << std::endl;

    std::cout << "Saving output data to file..." << std::endl;
    std::ofstream output_file("output_data.txt");

    if (output_file.is_open()) {
        host_accessor outData_acc(outData_buf, read_only);
        for (size_t i = 0; i < dataSize; ++i) {
            output_file << outData_acc[i] << std::endl;
            output_data[i] = outData_acc[i];
        }
        output_file.close();
        std::cout << "Output data saved to output_data.txt." << std::endl;
    } else {
        std::cerr << "Unable to open file for writing output data." << std::endl;
    }

    // Verification using naive NTT
    std::cout << "Verifying output using naive NTT implementation..." << std::endl;
    ntt_ct_rev2std_naive(input_data.data(), dataSize, twiddle_factors.data(), q_modulus);

    if (compare_arrays(input_data, output_data)) {
        std::cout << "Verification successful: FPGA NTT output matches the naive NTT output." << std::endl;
    } else {
        std::cout << "Verification failed: FPGA NTT output does not match the naive NTT output." << std::endl;
    }

    return 0;
}
