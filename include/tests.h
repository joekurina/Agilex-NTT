// Copyright IBM Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Begin external C linkage if used in C++ compilation context
EXTERNC_BEGIN

// Include test case definitions
#include "test_cases.h"

// Enumeration for different forward NTT implementations
typedef enum
{
  FIRST_FWD = 0,   // First forward NTT function
  FWD_REF   = FIRST_FWD,  // Reference forward NTT (Radix-2)
  FWD_R4               // Radix-4 forward NTT
} func_num_t;

// Function to test the correctness of NTT implementations
// Takes a pointer to a test case structure and returns an integer status code
int test_correctness(const test_case_t *t);

EXTERNC_END
