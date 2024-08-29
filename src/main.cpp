// main.cpp

#include "kernel/ntt.h"
#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <dpc_common.hpp>
#include <iostream>
#include <fstream>  // Include for file handling
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

    // Define the parameters
    const unsigned int numFrames = 1;
    const size_t dataSize = FPGA_NTT_SIZE; // Defined in ntt.cpp
    
    std::cout << "Initializing buffers..." << std::endl;

    // Create buffers for input and output data
    buffer<WideVecType, 1> inData_buf(dataSize / VEC);
    buffer<unsigned32Bits_t, 1> miniBatchSize_buf(1);
    buffer<Wide64BytesType, 1> twiddleFactors_buf(dataSize / (sizeof(Wide64BytesType) / sizeof(unsigned64Bits_t)));
    buffer<Wide64BytesType, 1> barrettTwiddleFactors_buf(dataSize / (sizeof(Wide64BytesType) / sizeof(unsigned64Bits_t)));
    buffer<unsigned64Bits_t, 1> modulus_buf(1);
    buffer<WideVecType, 1> outData_buf(dataSize / VEC);

    std::cout << "Initializing input data using host accessors..." << std::endl;

    // Open a file to save the generated input data
    std::ofstream input_file("generated_inputs.txt");

    // Initialize input data using a host accessor
    {
        host_accessor inData_acc(inData_buf, write_only);
        host_accessor miniBatchSize_acc(miniBatchSize_buf, write_only);
        host_accessor twiddleFactors_acc(twiddleFactors_buf, write_only);
        host_accessor barrettTwiddleFactors_acc(barrettTwiddleFactors_buf, write_only);
        host_accessor modulus_acc(modulus_buf, write_only);

        miniBatchSize_acc[0] = numFrames;

        const unsigned long q = 65537;  // Example modulus
        const unsigned long primitive_root = 3;  // Assume 3 is a primitive root mod q

        if (input_file.is_open()) {
            for (size_t i = 0; i < dataSize / VEC; ++i) {
                for (size_t j = 0; j < VEC; ++j) {
                    inData_acc[i].data[j] = i * VEC + j;
                    input_file << inData_acc[i].data[j] << std::endl;
                }
            }

            // Initialize the twiddle factors
            for (size_t i = 0; i < dataSize / (sizeof(Wide64BytesType) / sizeof(unsigned64Bits_t)); ++i) {
                for (size_t j = 0; j < sizeof(Wide64BytesType) / sizeof(unsigned64Bits_t); ++j) {
                    unsigned long index = i * (sizeof(Wide64BytesType) / sizeof(unsigned64Bits_t)) + j;
                    twiddleFactors_acc[i].data[j] = powmod(primitive_root, index, q);
                    barrettTwiddleFactors_acc[i].data[j] = (twiddleFactors_acc[i].data[j] * powmod(primitive_root, index, q)) % q; // Example Barrett precomputation
                }
            }

            modulus_acc[0] = q;
            input_file << "Modulus: " << modulus_acc[0] << std::endl;

            input_file.close(); // Close the input file after writing
            std::cout << "Input data saved to generated_inputs.txt." << std::endl;
        } else {
            std::cerr << "Unable to open file for writing input data." << std::endl;
        }
    }

    std::cout << "Calling forward NTT kernel..." << std::endl;
    // Call the NTT function
    fwd_ntt_kernel<0>(q, inData_buf, miniBatchSize_buf, twiddleFactors_buf, barrettTwiddleFactors_buf, modulus_buf, outData_buf);
    std::cout << "Forward NTT kernel completed." << std::endl;

    std::cout << "Waiting for output event..." << std::endl;
    q.wait();
    std::cout << "Output event completed." << std::endl;

    std::cout << "Saving output data to file..." << std::endl;
    std::ofstream output_file("output_data.txt");  // Open file for writing

    if (output_file.is_open()) {
        host_accessor outData_acc(outData_buf, read_only);
        for (size_t i = 0; i < dataSize / VEC; ++i) {
            for (size_t j = 0; j < VEC; ++j) {
                output_file << outData_acc[i].data[j] << std::endl;
            }
        }
        output_file.close();  // Close the file after writing
        std::cout << "Output data saved to output_data.txt." << std::endl;
    } else {
        std::cerr << "Unable to open file for writing output data." << std::endl;
    }

    return 0;
}
