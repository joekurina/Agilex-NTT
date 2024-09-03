#include "defs.h"
#include "fast_mul_operators.h"
#include "ntt_reference.h"
#include "ntt_radix4.h"
//#include "kernel/ntt.hpp"
#include "test_cases.h"  // Include the test cases header
#include "tests.h"       // Include the tests header
//#include <CL/sycl.hpp>
//#include <sycl/ext/intel/fpga_extensions.hpp>
//#include <dpc_common.hpp>
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>

#define FPGA_NTT_SIZE 16384 // FPGA_NTT_SIZE must be a power of 2

//using namespace cl::sycl;

int main() {
    // Initialize test cases
    if (!init_test_cases()) {
        std::cerr << "Failed to initialize test cases" << std::endl;
        return 1;
    }

    // Run each test case
    for (size_t i = 0; i < NUM_OF_TEST_CASES; ++i) {
        std::cout << "Running test case " << i + 1 << " of " << NUM_OF_TEST_CASES << std::endl;
        if (test_correctness(&tests[i]) != SUCCESS) {
            std::cerr << "Test case " << i + 1 << " failed!" << std::endl;
        } else {
            std::cout << "Test case " << i + 1 << " passed." << std::endl;
        }
    }

    // Cleanup test cases
    destroy_test_cases();

    return 0;
}
