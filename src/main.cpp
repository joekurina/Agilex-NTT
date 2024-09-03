#include "defs.h"
#include "fast_mul_operators.h"
#include "ntt_reference.h"
#include "ntt_radix4.h"
#include "kernel/ntt.hpp"
#include "test_cases.h"
#include "tests.h"
#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <dpc_common.hpp>
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>

#define FPGA_NTT_SIZE 16384 // FPGA_NTT_SIZE must be a power of 2

using namespace cl::sycl;

// Function to test the FPGA kernel against the reference implementation
void test_fpga_kernel(sycl::queue& q, const test_case_t& t, std::ofstream& log_file) {
    // Create buffers for data, twiddle factors, and modulus
    sycl::buffer<uint64_t, 1> data_buf(t.n);
    sycl::buffer<uint64_t, 1> twiddleFactors_buf(t.w_powers.ptr, t.n);
    sycl::buffer<uint64_t, 1> modulus_buf(&t.q, 1);
    sycl::buffer<uint64_t, 1> outData_buf(t.n);

    // Fill the input data buffer with test data
    {
        auto data_acc = data_buf.get_access<sycl::access::mode::write>();
        for (size_t i = 0; i < t.n; ++i) {
            data_acc[i] = i;  // Using simple sequential data for testing
        }
    }

    // Run the FPGA kernel
    fwd_ntt_kernel<0>(q, data_buf, twiddleFactors_buf, modulus_buf, outData_buf);

    // Compare results with the reference NTT implementation
    {
        // Get access to the output data from the FPGA kernel
        auto outData_acc = outData_buf.get_access<sycl::access::mode::read>();

        // Prepare a reference NTT result for comparison
        std::vector<uint64_t> reference_data(t.n);
        std::copy(outData_acc.get_pointer(), outData_acc.get_pointer() + t.n, reference_data.begin());

        fwd_ntt_ref_harvey(reference_data.data(), t.n, t.q, t.w_powers.ptr, t.w_powers_con.ptr);

        // Compare the FPGA result with the reference result
        bool success = true;
        for (size_t i = 0; i < t.n; ++i) {
            if (outData_acc[i] != reference_data[i]) {
                log_file << "Mismatch at index " << i << ": FPGA=" << outData_acc[i]
                          << ", Reference=" << reference_data[i] << std::endl;
                success = false;
            }
        }

        if (success) {
            log_file << "FPGA kernel test passed." << std::endl;
        } else {
            log_file << "FPGA kernel test failed." << std::endl;
        }
    }
}

int main() {
    // Initialize test cases
    if (!init_test_cases()) {
        std::cerr << "Failed to initialize test cases" << std::endl;
        return 1;
    }

    // Open the log file for writing the results
    std::ofstream log_file("test_results.txt");
    if (!log_file.is_open()) {
        std::cerr << "Failed to open log file for writing." << std::endl;
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
        log_file << "Running test case " << i + 1 << " of " << NUM_OF_TEST_CASES << std::endl;

        // Test the CPU implementations
        if (test_correctness(&tests[i]) != SUCCESS) {
            log_file << "Test case " << i + 1 << " failed on CPU!" << std::endl;
        } else {
            log_file << "Test case " << i + 1 << " passed on CPU." << std::endl;
        }

        // Test the FPGA kernel implementation
        test_fpga_kernel(q, tests[i], log_file);
    }

    // Cleanup test cases
    destroy_test_cases();

    // Close the log file
    log_file.close();

    return 0;
}
