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

// Function to compute (base^exp) % mod
uint64_t powmod(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t result = 1;
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

// Function to initialize data and twiddle factors
void initialize_data(size_t dataSize, uint64_t q_modulus, uint64_t primitive_root, 
                     std::vector<uint64_t>& input_data, std::vector<uint64_t>& twiddle_factors, 
                     sycl::buffer<uint64_t, 1>& inData_buf, sycl::buffer<uint64_t, 1>& twiddleFactors_buf, 
                     sycl::buffer<uint64_t, 1>& modulus_buf) {
    std::ofstream input_file("input.txt");

    {
        sycl::host_accessor inData_acc(inData_buf, sycl::write_only);
        sycl::host_accessor twiddleFactors_acc(twiddleFactors_buf, sycl::write_only);
        sycl::host_accessor modulus_acc(modulus_buf, sycl::write_only);

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
            std::cout << "Input data saved to input.txt." << std::endl;
        } else {
            std::cerr << "Unable to open file for writing input data." << std::endl;
        }
    }
}

// Function to perform Reference forward NTT on the CPU and save results
void ref_ntt_cpu(size_t dataSize, uint64_t q_modulus, 
                 const std::vector<uint64_t>& twiddle_factors, std::vector<uint64_t>& data) {
    fwd_ntt_ref_harvey_lazy(data.data(), dataSize, q_modulus, twiddle_factors.data(), twiddle_factors.data());

    std::ofstream output_file("cpu_output_data_reference.txt");
    if (output_file.is_open()) {
        for (size_t i = 0; i < dataSize; ++i) {
            output_file << data[i] << std::endl;
        }
        output_file.close();
        std::cout << "CPU output data saved to cpu_output_data_reference.txt" << std::endl;
    } else {
        std::cerr << "Unable to open file for writing CPU output data." << std::endl;
    }
}

// Function to perform Radix-4 forward NTT on the CPU and save results
void radix4_ntt_cpu(size_t dataSize, uint64_t q_modulus, 
                    const std::vector<uint64_t>& twiddle_factors, std::vector<uint64_t>& data) {
    fwd_ntt_radix4_lazy(data.data(), dataSize, q_modulus, twiddle_factors.data(), twiddle_factors.data());

    std::ofstream output_file("cpu_output_data_radix4.txt");
    if (output_file.is_open()) {
        for (size_t i = 0; i < dataSize; ++i) {
            output_file << data[i] << std::endl;
        }
        output_file.close();
        std::cout << "CPU output data saved to cpu_output_data_radix4.txt" << std::endl;
    } else {
        std::cerr << "Unable to open file for writing CPU output data." << std::endl;
    }
}

// Function to perform forward NTT on the FPGA
void fpga_ntt(sycl::queue& q, size_t dataSize, 
                       sycl::buffer<uint64_t, 1>& inData_buf, sycl::buffer<uint64_t, 1>& twiddleFactors_buf, 
                       sycl::buffer<uint64_t, 1>& modulus_buf, sycl::buffer<uint64_t, 1>& outData_buf, 
                       std::vector<uint64_t>& output_data) {
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

    // Create buffers
    sycl::buffer<uint64_t, 1> inData_buf(dataSize);
    sycl::buffer<uint64_t, 1> twiddleFactors_buf(dataSize);
    sycl::buffer<uint64_t, 1> modulus_buf(1);
    sycl::buffer<uint64_t, 1> outData_buf(dataSize);

    // Initialize input data and twiddle factors
    initialize_data(dataSize, q_modulus, primitive_root, input_data, twiddle_factors, 
                    inData_buf, twiddleFactors_buf, modulus_buf);

    // Create separate copies of input data for each NTT computation
    std::vector<uint64_t> input_data_ref = input_data;
    std::vector<uint64_t> input_data_radix4 = input_data;

    // Perform forward NTT (reference) on the CPU and save results
    ref_ntt_cpu(dataSize, q_modulus, twiddle_factors, input_data_ref);

    // Perform forward NTT (radix-4) on the CPU and save results
    radix4_ntt_cpu(dataSize, q_modulus, twiddle_factors, input_data_radix4);

    // Perform forward NTT on the FPGA
    fpga_ntt(q, dataSize, inData_buf, twiddleFactors_buf, modulus_buf, outData_buf);

    return 0;
}
