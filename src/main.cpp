// main.cpp

#include "kernel/ntt.h"
#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <dpc_common.hpp>
#include <iostream>

#define FPGA_NTT_SIZE 16384 // FPGA_NTT_SIZE must be a power of 2

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
    
    std::cout << "Initializing buffers..." << std::endl;

    // Create buffers for input and output data
    buffer<uint64_t, 1> inData_buf(dataSize);
    buffer<uint64_t, 1> inData2_buf(dataSize);
    buffer<uint64_t, 1> modulus_buf(1);
    buffer<uint64_t, 1> twiddleFactors_buf(dataSize);
    buffer<uint64_t, 1> barrettTwiddleFactors_buf(dataSize);
    buffer<uint64_t, 1> outData_buf(dataSize);

    std::cout << "Initializing input data using host accessors..." << std::endl;

    // Initialize input data using a host accessor
    {
        host_accessor inData_acc(inData_buf, write_only);
        host_accessor inData2_acc(inData2_buf, write_only);
        host_accessor modulus_acc(modulus_buf, write_only);
        host_accessor twiddleFactors_acc(twiddleFactors_buf, write_only);
        host_accessor barrettTwiddleFactors_acc(barrettTwiddleFactors_buf, write_only);

        for (size_t i = 0; i < dataSize; ++i) {
            inData_acc[i] = i;
            inData2_acc[i] = i + 1;
            twiddleFactors_acc[i] = i + 2;
            barrettTwiddleFactors_acc[i] = i + 3;
        }
        modulus_acc[0] = 65537; // Example modulus
    }

    std::cout << "Calling NTT input kernel..." << std::endl;
    // Call the input and output functions directly without capturing return values
    ntt_input_kernel(inData_buf, inData2_buf, modulus_buf, twiddleFactors_buf, barrettTwiddleFactors_buf, numFrames, q);
    std::cout << "NTT input kernel completed." << std::endl;

    std::cout << "Calling forward NTT kernel..." << std::endl;
    // Call the NTT function
    fwd_ntt_kernel<0>(q);
    std::cout << "Forward NTT kernel completed." << std::endl;
    
    std::cout << "Calling NTT output kernel..." << std::endl;
    ntt_output_kernel(outData_buf, numFrames, q);
    std::cout << "NTT output kernel completed." << std::endl;

    std::cout << "Waiting for output event..." << std::endl;
    // Wait for the queue to finish processing
    q.wait();  // This waits for all events in the queue to complete
    std::cout << "Output event completed." << std::endl;

    std::cout << "Reading and printing output data..." << std::endl;
    // Display the results using a host accessor
    {
        host_accessor outData_acc(outData_buf, read_only);
        for (size_t i = 0; i < dataSize; ++i) {
            std::cout << outData_acc[i] << std::endl;
        }
    }

    std::cout << "Finished printing output data." << std::endl;

    return 0;
}
