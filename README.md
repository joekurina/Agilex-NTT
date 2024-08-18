# Performing an NTT on the DE10-Agilex Card

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux Ubuntu LTS 20.04
| Hardware                          | Terasic&reg; DE10-Agilex FPGA Card
| Software                          | Intel&reg; oneAPI DPC++ Compiler, Intel&reg; FPGA Add-on for oneAPI Base Toolkit

## Dependencies
libntl-dev package

## Purpose
This project performs the NTT Tests taken from SEAL Embedded on the DE10-Agilex Card

## Building the `Make based FPGA` Program

### Include Files
The include folder is located at %ONEAPI_ROOT%\dev-utilities\latest\include on your development system.

### On a Linux System
The following instructions assume you are in the project's root folder.

To build the template for FPGA Emulator (fast compile time, targets a CPU-emulated FPGA device):
  ```
  make build_emu
  ```

To generate an FPGA optimization report (fast compile time, partial FPGA HW compile):
  ```
  make report
  ```
Locate the FPGA optimization report, `report.html`, in the `fpga_report.prj/reports/` directory.

To build the template for FPGA Hardware (takes about one hour, the system must
have at least 32 GB of physical dynamic memory):
  ```
  make build_hw
  ```

To run the template on FPGA Emulator:
  ```
  make run_emu
  ```

To run the template on FPGA Hardware:
  ```
  make run_hw
  ```

To clean the build artifacts, use:
  ```
  make clean
  ```
