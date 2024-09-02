// main.cpp
#include "ntt_reference.h"
#include "ntt_radix4.h"
#include "kernel/ntt.hpp"
#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <dpc_common.hpp>
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>

#define FPGA_NTT_SIZE 16384 // FPGA_NTT_SIZE must be a power of 2

using namespace cl::sycl;

// Function to compute (base^exp) % mod using modular exponentiation by squaring
uint64_t powmod(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t result = 1;                            // Initialize result to 1 (this will hold the final answer)
    base = base % mod;                              // Take the base modulo `mod` to ensure it's within the modulus
    while (exp > 0) {                               // Loop until `exp` becomes 0
        if (exp % 2 == 1)                           // If `exp` is odd, multiply the current `result` by `base`
            result = (result * base) % mod;         // Update `result` with the product modulo `mod`
        exp = exp / 2;                              // Halve the exponent (this is the "divide and conquer" part)
        base = (base * base) % mod;                 // Square the base and take modulo `mod` (prepare for next iteration)
    }
    return result;                                  // Return the final result (which is (base^exp) % mod)
}


// Function to compare two arrays
bool compare_arrays(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b, uint64_t q_modulus) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        uint64_t a_mod = (a[i] % q_modulus + q_modulus) % q_modulus;  // Adjust for modulus
        uint64_t b_mod = (b[i] % q_modulus + q_modulus) % q_modulus;  // Adjust for modulus
        if (a_mod != b_mod) {
            std::cout << "Mismatch at index " << i << ": " << a_mod << " != " << b_mod << std::endl;
            return false;
        }
    }
    return true;
}

// Function to initialize input data and twiddle factors
void initialize_data(size_t dataSize, uint64_t q_modulus, uint64_t primitive_root, 
                     std::vector<uint64_t>& input_data, std::vector<uint64_t>& twiddle_factors) {
    
    // Open a file to save the generated input data
    std::ofstream input_file("input.txt");

    // Open a file to save the generated twiddle factors
    std::ofstream twiddle_file("twiddle_factors.txt");

    // Check if both files are successfully opened
    if (input_file.is_open() && twiddle_file.is_open()) {
        for (size_t i = 0; i < dataSize; ++i) {
            // Initialize input data with sequential values (0, 1, 2, ..., dataSize-1)
            input_data[i] = i;

            // Write the initialized input data to the input file
            input_file << input_data[i] << std::endl;

            // Calculate the i-th twiddle factor using modular exponentiation
            twiddle_factors[i] = powmod(primitive_root, i, q_modulus);

            // Write the calculated twiddle factor to the twiddle factors file
            twiddle_file << twiddle_factors[i] << std::endl;
        }

        // Write the modulus value to the end of both files for reference
        input_file << "Modulus: " << q_modulus << std::endl;
        twiddle_file << "Modulus: " << q_modulus << std::endl;

        // Close both files after writing
        input_file.close();
        twiddle_file.close();

        // Output messages
        std::cout << "Input data saved to input.txt." << std::endl;
        std::cout << "Twiddle factors saved to twiddle_factors.txt." << std::endl;
    } else {
        // Error message
        std::cerr << "Unable to open file for writing input data or twiddle factors." << std::endl;
    }
}


// Function to perform Reference forward NTT on the CPU, then an inverse NTT, and save the results
void ref_ntt_cpu(size_t dataSize, uint64_t q_modulus, 
                 const std::vector<uint64_t>& twiddle_factors, std::vector<uint64_t>& data) {
    
    // Perform forward NTT (Harvey Lazy) on the data
    fwd_ntt_ref_harvey_lazy(data.data(), dataSize, q_modulus, twiddle_factors.data(), twiddle_factors.data());

    // Save the result of the forward NTT to a file
    std::ofstream output_file("ref_ntt_output.txt");
    if (output_file.is_open()) {
        for (size_t i = 0; i < dataSize; ++i) {
            output_file << data[i] << std::endl;
        }
        output_file.close();
        std::cout << "CPU forward NTT output data saved to ref_ntt_output.txt" << std::endl;
    } else {
        std::cerr << "Unable to open file for writing CPU forward NTT output data." << std::endl;
    }

    // Compute n_inv (modular inverse of N modulo q)
    uint64_t n_inv_value = powmod(dataSize, q_modulus - 2, q_modulus);  // Using Fermat's little theorem

    // Compute the second part of n_inv in 128-bit space to avoid overflow
    __uint128_t n_inv_extended = static_cast<__uint128_t>(n_inv_value) << 64;
    mul_op_t n_inv = {n_inv_value, static_cast<uint64_t>(n_inv_extended / q_modulus)};

    // Perform inverse NTT on the data
    inv_ntt_ref_harvey(data.data(), dataSize, q_modulus, n_inv, 64, twiddle_factors.data(), twiddle_factors.data());

    // Save the result of the inverse NTT to a separate file
    std::ofstream inv_output_file("ref_intt_output.txt");
    if (inv_output_file.is_open()) {
        for (size_t i = 0; i < dataSize; ++i) {
            inv_output_file << data[i] << std::endl;
        }
        inv_output_file.close();
        std::cout << "CPU inverse NTT output data saved to ref_intt_output.txt" << std::endl;
    } else {
        std::cerr << "Unable to open file for writing CPU inverse NTT output data." << std::endl;
    }
}

// Function to perform Radix-4 forward NTT on the CPU, then an inverse NTT, and save the results
void radix4_ntt_cpu(size_t dataSize, uint64_t q_modulus, 
                    const std::vector<uint64_t>& twiddle_factors, std::vector<uint64_t>& data) {
    
    // Perform forward NTT (Radix-4) on the data
    fwd_ntt_radix4_lazy(data.data(), dataSize, q_modulus, twiddle_factors.data(), twiddle_factors.data());

    // Save the result of the forward NTT to a file
    std::ofstream output_file("radix4_ntt_output.txt");
    if (output_file.is_open()) {
        for (size_t i = 0; i < dataSize; ++i) {
            output_file << data[i] << std::endl;
        }
        output_file.close();
        std::cout << "CPU forward NTT output data saved to radix4_ntt_output.txt" << std::endl;
    } else {
        std::cerr << "Unable to open file for writing CPU forward NTT output data." << std::endl;
    }

    // Compute n_inv (modular inverse of N modulo q)
    uint64_t n_inv_value = powmod(dataSize, q_modulus - 2, q_modulus);  // Using Fermat's little theorem
    mul_op_t n_inv = {n_inv_value, (n_inv_value << 64) / q_modulus};

    // Perform inverse NTT on the data
    inv_ntt_radix4(data.data(), dataSize, q_modulus, n_inv, twiddle_factors.data(), twiddle_factors.data());

    // Save the result of the inverse NTT to a separate file
    std::ofstream inv_output_file("radix4_intt_output.txt");
    if (inv_output_file.is_open()) {
        for (size_t i = 0; i < dataSize; ++i) {
            inv_output_file << data[i] << std::endl;
        }
        inv_output_file.close();
        std::cout << "CPU inverse NTT output data saved to radix4_intt_output.txt" << std::endl;
    } else {
        std::cerr << "Unable to open file for writing CPU inverse NTT output data." << std::endl;
    }
}

// Function to perform forward NTT on the FPGA
void fpga_ntt(sycl::queue& q, size_t dataSize, 
                       const std::vector<uint64_t>& input_data,
                       const std::vector<uint64_t>& twiddle_factors,
                       uint64_t q_modulus,
                       sycl::buffer<uint64_t, 1>& outData_buf, 
                       std::vector<uint64_t>& output_data) {
    // Create buffers
    sycl::buffer<uint64_t, 1> inData_buf(input_data.data(), dataSize);
    sycl::buffer<uint64_t, 1> twiddleFactors_buf(twiddle_factors.data(), dataSize);
    sycl::buffer<uint64_t, 1> modulus_buf(&q_modulus, 1);

    // Perform NTT computation on FPGA
    fwd_ntt_kernel<0>(q, inData_buf, twiddleFactors_buf, modulus_buf, outData_buf);
    q.wait();
    std::cout << "FPGA Forward NTT kernel completed." << std::endl;

    // Save output data to a file
    std::ofstream output_file("fpga_output_data.txt");
    if (output_file.is_open()) {
        sycl::host_accessor outData_acc(outData_buf, sycl::read_only);
        for (size_t i = 0; i < dataSize; ++i) {
            output_file << outData_acc[i] << std::endl;
            output_data[i] = outData_acc[i];
        }
        output_file.close();
        std::cout << "FPGA output data saved to fpga_output_data.txt" << std::endl;
    } else {
        std::cerr << "Unable to open file for writing output data." << std::endl;
    }
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
    const uint64_t q_modulus = 65537;  // Example modulus
    const uint64_t primitive_root = 3;  // Assume 3 is a primitive root mod q

    std::vector<uint64_t> input_data(dataSize);
    std::vector<uint64_t> twiddle_factors(dataSize);
    std::vector<uint64_t> output_data(dataSize);

    // Initialize input data and twiddle factors
    initialize_data(dataSize, q_modulus, primitive_root, input_data, twiddle_factors);

    // Create separate copies of input data for each NTT computation
    std::vector<uint64_t> input_data_ref = input_data;
    std::vector<uint64_t> input_data_radix4 = input_data;

    // Create output buffer for FPGA
    sycl::buffer<uint64_t, 1> outData_buf(dataSize);

    // Perform forward NTT (reference) on the CPU and save results
    ref_ntt_cpu(dataSize, q_modulus, twiddle_factors, input_data_ref);

    // Perform forward NTT (radix-4) on the CPU and save results
    radix4_ntt_cpu(dataSize, q_modulus, twiddle_factors, input_data_radix4);

    // Perform forward NTT on the FPGA
    fpga_ntt(q, dataSize, input_data, twiddle_factors, q_modulus, outData_buf, output_data);

    return 0;
}
