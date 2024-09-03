// Copyright IBM Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <string.h>

#include "ntt_radix4.h"
#include "ntt_reference.h"
#include "pre_compute.h"
#include "test_cases.h"
#include "utils.h"

static inline int test_radix2_scalar(const test_case_t *t, uint64_t a_orig[])
{
  uint64_t a[t->n];
  memcpy(a, a_orig, sizeof(a));

  printf("Running fwd_ntt_ref_harvey\n");
  fwd_ntt_ref_harvey(a, t->n, t->q, t->w_powers.ptr, t->w_powers_con.ptr);

  printf("Running inv_ntt_ref_harvey\n");
  inv_ntt_ref_harvey(a, t->n, t->q, t->n_inv, WORD_SIZE, t->w_inv_powers.ptr,
                     t->w_inv_powers_con.ptr);

  GUARD_MSG(memcmp(a_orig, a, sizeof(a)), "Bad results after radix-2 inv\n");

  return SUCCESS;
}

static inline int
test_radix4_scalar(const test_case_t *t, uint64_t a_orig[], uint64_t a_ntt[])
{
  uint64_t a[t->n];
  memcpy(a, a_orig, sizeof(a));

  printf("Running fwd_ntt_radix4\n");
  fwd_ntt_radix4(a, t->n, t->q, t->w_powers_r4.ptr, t->w_powers_con_r4.ptr);
  GUARD_MSG(memcmp(a_ntt, a, sizeof(a)), "Bad results after radix-4 fwd\n");

  printf("Running inv_ntt_radix4\n");
  inv_ntt_radix4(a, t->n, t->q, t->n_inv, t->w_inv_powers_r4.ptr,
                 t->w_inv_powers_con_r4.ptr);

  GUARD_MSG(memcmp(a_orig, a, sizeof(a)), "Bad results after radix-4 inv\n");

  return SUCCESS;
}

int test_correctness(const test_case_t *t)
{
  // Prepare input
  uint64_t a[t->n];
  uint64_t b[t->n];
  uint64_t a_ntt[t->n];
  uint64_t a_cpy[t->n];
  random_buf(a, t->n, t->q);
  memcpy(a_cpy, a, sizeof(a));
  memcpy(b, a, sizeof(a));

  // Prepare a_ntt = NTT(a)
  fwd_ntt_ref_harvey(a_cpy, t->n, t->q, t->w_powers.ptr, t->w_powers_con.ptr);
  memcpy(a_ntt, a_cpy, sizeof(a_cpy));

  GUARD(test_radix2_scalar(t, a));
  GUARD(test_radix4_scalar(t, a, a_ntt))

  return SUCCESS;
}
