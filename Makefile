TARGET_EMU    = fpga_emu
TARGET_HW     = fpga_hardware
TARGET_REPORT = fpga_report.a

# Source files
C_SRCS   = src/ntt_radix4.c src/ntt_reference.c
CPP_SRCS = src/main.cpp src/kernel/ntt.cpp
SRCS     = $(CPP_SRCS) $(C_SRCS)
OBJS     = $(CPP_SRCS:.cpp=.o) $(C_SRCS:.c=.o)

# Compiler and flags
CXX      = icpx
CC       = icx
CXXFLAGS = -std=c++17 -fsycl -Iinclude -Iinclude/kernel

# Add the C compiler flags from the CMake file
CFLAGS   = -ggdb -O3 -fPIC
CFLAGS  += -fvisibility=hidden -Wall -Wextra -Werror -Wpedantic
CFLAGS  += -Wunused -Wcomment -Wuninitialized -Wshadow
CFLAGS  += -Wwrite-strings -Wformat-security -Wcast-qual -Wunused-result

# Architecture-specific flags
CFLAGS  += -march=native -mno-red-zone

# Include directory for ntt_radix4.h
CFLAGS  += -Iinclude

# Add any additional flags based on conditions
# (Assuming no additional conditions for sanitizers, etc.)

.PHONY: build build_emu build_hw report run_emu run_hw clean run
.DEFAULT_GOAL := build_emu

# Intel-supported FPGA cards 
FPGA_DEVICE_A10 = intel_a10gx_pac:pac_a10
FPGA_DEVICE_S10 = intel_s10sx_pac:pac_s10
FPGA_DEVICE_A7  = de10_agilex:B2E2_8GBx4
FPGA_DEVICE     = $(FPGA_DEVICE_A7)

# Compile flags
EMULATOR_FLAGS  = -fintelfpga -DFPGA_EMULATOR -Xsv
HARDWARE_FLAGS  = -fintelfpga -Xsv -Xshardware -Xsboard=$(FPGA_DEVICE)
REPORT_FLAGS    = $(HARDWARE_FLAGS) -fsycl-link

# Build rules for object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(EMULATOR_FLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Build for FPGA emulator
build: build_emu
build_emu: $(TARGET_EMU)

$(TARGET_EMU): $(OBJS)
	$(CXX) $(CXXFLAGS) $(EMULATOR_FLAGS) -o $@ $(OBJS)

# Generate FPGA optimization report (without compiling all the way to hardware)
report: $(TARGET_REPORT)

$(TARGET_REPORT): $(OBJS)
	$(CXX) $(CXXFLAGS) $(REPORT_FLAGS) -o $@ $(OBJS)

# Build for FPGA hardware
build_hw: $(TARGET_HW)

$(TARGET_HW): $(OBJS)
	$(CXX) $(CXXFLAGS) $(HARDWARE_FLAGS) -o $@ $(OBJS)

# Run on the FPGA emulator
run: run_emu
run_emu: $(TARGET_EMU)
	./$(TARGET_EMU)

# Run on the FPGA card
run_hw: $(TARGET_HW)
	./$(TARGET_HW)

# Clean all
clean:
	-$(RM) $(OBJS) $(TARGET_EMU) $(TARGET_HW) $(TARGET_REPORT) *.d
	-$(RM) -R *.prj
