#include "defs.h"                   // defs for ntt stuff
#include "fast_mul_operators.h"     // operators required for ntt
#include "ntt_reference.h"
#include "ntt_radix4.h"
#include "kernel/ntt.hpp"
#include "test_cases.h"             // Include the test cases header
#include "tests.h"                  // Include the tests header
#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <dpc_common.hpp>
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>

#define FPGA_NTT_SIZE 16384 // FPGA_NTT_SIZE must be a power of 2

using namespace cl::sycl;

int main() {
    // Initialize test cases
    if (!init_test_cases()) {
        std::cerr << "Failed to initialize test cases" << std::endl;
        return 1;
    }

    // Create output files
    std::ofstream cpu_ref_output("cpu_ref_results.txt");
    std::ofstream cpu_radix4_output("cpu_radix4_results.txt");
    std::ofstream fpga_output("fpga_results.txt");

    if (!cpu_ref_output.is_open() || !cpu_radix4_output.is_open() || !fpga_output.is_open()) {
        std::cerr << "Failed to open output files" << std::endl;
        return 1;
    }

    // Create a device selector
    #if defined(FPGA_EMULATOR)
    sycl::ext::intel::fpga_emulator_selector device_selector;
    #else
    sycl::ext::intel::fpga_selector device_selector;
    #endif

    // Create a SYCL queue
    sycl::queue q(device_selector, dpc_common::exception_handler);

    // Run each test case
    for (size_t i = 0; i < NUM_OF_TEST_CASES; ++i) {
        std::cout << "Running test case " << i + 1 << " of " << NUM_OF_TEST_CASES << std::endl;

        // Prepare buffers for the FPGA kernel
        sycl::buffer<uint64_t, 1> data_buf(tests[i].n);
        sycl::buffer<uint64_t, 1> twiddleFactors_buf(tests[i].w_powers.ptr, tests[i].n);
        sycl::buffer<uint64_t, 1> modulus_buf(&tests[i].q, 1);
        sycl::buffer<uint64_t, 1> outData_buf(tests[i].n);

        // Run the reference CPU NTT
        uint64_t a_ntt[tests[i].n];
        memcpy(a_ntt, tests[i].w_powers.ptr, sizeof(uint64_t) * tests[i].n);
        fwd_ntt_ref_harvey(a_ntt, tests[i].n, tests[i].q, tests[i].w_powers.ptr, tests[i].w_powers_con.ptr);

        // Write the results to the reference CPU output file
        cpu_ref_output << "Test Case " << i + 1 << std::endl;
        for (size_t j = 0; j < tests[i].n; ++j) {
            cpu_ref_output << a_ntt[j] << std::endl;
        }

        // Run the radix-4 CPU NTT
        uint64_t a_radix4[tests[i].n];
        memcpy(a_radix4, tests[i].w_powers.ptr, sizeof(uint64_t) * tests[i].n);
        fwd_ntt_radix4(a_radix4, tests[i].n, tests[i].q, tests[i].w_powers_r4.ptr, tests[i].w_powers_con_r4.ptr);

        // Write the results to the radix-4 CPU output file
        cpu_radix4_output << "Test Case " << i + 1 << std::endl;
        for (size_t j = 0; j < tests[i].n; ++j) {
            cpu_radix4_output << a_radix4[j] << std::endl;
        }

        // Run the FPGA NTT kernel
        {
            auto data_acc = data_buf.get_access<sycl::access::mode::write>();
            for (size_t j = 0; j < tests[i].n; ++j) {
                data_acc[j] = tests[i].w_powers.ptr[j];
            }
        }

        fwd_ntt_kernel<0>(q, data_buf, twiddleFactors_buf, modulus_buf, outData_buf);

        // Retrieve and write FPGA results
        std::vector<uint64_t> output_data(tests[i].n);
        {
            auto outData_acc = outData_buf.get_access<sycl::access::mode::read>();
            fpga_output << "Test Case " << i + 1 << std::endl;
            for (size_t j = 0; j < tests[i].n; ++j) {
                output_data[j] = outData_acc[j];
                fpga_output << output_data[j] << std::endl;
            }
        }

        // Compare FPGA output with reference output
        if (!compare_arrays(output_data, a_ntt, tests[i].q)) {
            std::cerr << "Test case " << i + 1 << " failed!" << std::endl;
        } else {
            std::cout << "Test case " << i + 1 << " passed." << std::endl;
        }
    }

    // Cleanup test cases
    destroy_test_cases();

    // Close output files
    cpu_ref_output.close();
    cpu_radix4_output.close();
    fpga_output.close();

    return 0;
}
