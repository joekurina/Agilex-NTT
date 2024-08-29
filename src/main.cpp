// main.cpp

#include "kernel/ntt.h"
#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <dpc_common.hpp>
#include <iostream>
#include <fstream>
#include <cmath>

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
    const unsigned long q_modulus = 65537;  // Example modulus
    const unsigned long primitive_root = 3;  // Assume 3 is a primitive root mod q

    std::cout << "Initializing buffers..." << std::endl;

    // Create buffers for input, twiddle factors, and output data
    buffer<uint64_t, 1> inData_buf(dataSize);
    buffer<uint64_t, 1> twiddleFactors_buf(dataSize);
    buffer<uint64_t, 1> modulus_buf(1);
    buffer<uint64_t, 1> outData_buf(dataSize);

    std::cout << "Initializing input data and twiddle factors..." << std::endl;

    // Open a file to save the generated input data
    std::ofstream input_file("generated_inputs.txt");

    {
        host_accessor inData_acc(inData_buf, write_only);
        host_accessor twiddleFactors_acc(twiddleFactors_buf, write_only);
        host_accessor modulus_acc(modulus_buf, write_only);

        if (input_file.is_open()) {
            for (size_t i = 0; i < dataSize; ++i) {
                inData_acc[i] = i;
                input_file << inData_acc[i] << std::endl;
            }

            for (size_t i = 0; i < dataSize; ++i) {
                twiddleFactors_acc[i] = powmod(primitive_root, i, q_modulus);
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
        }
        output_file.close();
        std::cout << "Output data saved to output_data.txt." << std::endl;
    } else {
        std::cerr << "Unable to open file for writing output data." << std::endl;
    }

    return 0;
}
