// main.cpp

#include "ntt.h"
#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <iostream>

using namespace cl::sycl;

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

    // Allocate memory for input and output data
    uint64_t* inData = sycl::malloc_host<uint64_t>(dataSize, q);
    uint64_t* inData2 = sycl::malloc_host<uint64_t>(dataSize, q);
    uint64_t* modulus = sycl::malloc_host<uint64_t>(1, q);
    uint64_t* twiddleFactors = sycl::malloc_host<uint64_t>(dataSize, q);
    uint64_t* barrettTwiddleFactors = sycl::malloc_host<uint64_t>(dataSize, q);
    uint64_t* outData = sycl::malloc_host<uint64_t>(dataSize, q);

    // Initialize input data (example)
    for (size_t i = 0; i < dataSize; ++i) {
        inData[i] = i;
        inData2[i] = i + 1;
        twiddleFactors[i] = i + 2;
        barrettTwiddleFactors[i] = i + 3;
    }
    modulus[0] = 65537; // Example modulus

    // Call the NTT functions
    fwd_ntt(q); // Launch the NTT kernels

    sycl::event input_event = ntt_input(q, numFrames, inData, inData2, modulus,
                                        twiddleFactors, barrettTwiddleFactors);
    
    sycl::event output_event = ntt_output(q, numFrames, outData);

    // Wait for the computation to finish
    output_event.wait();

    // Display the results
    for (size_t i = 0; i < dataSize; ++i) {
        std::cout << outData[i] << std::endl;
    }

    // Free allocated memory
    sycl::free(inData, q);
    sycl::free(inData2, q);
    sycl::free(modulus, q);
    sycl::free(twiddleFactors, q);
    sycl::free(barrettTwiddleFactors, q);
    sycl::free(outData, q);

    return 0;
}
