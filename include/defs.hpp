// defs.hpp
#pragma once

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <type_traits>

namespace ntt {

constexpr int SUCCESS = 0;
constexpr int ERROR = -1;

#define GUARD(func)                 \
    do {                            \
        if (SUCCESS != (func)) {    \
            return ERROR;           \
        }                           \
    } while (0)

#define GUARD_MSG(func, msg)        \
    do {                            \
        if (SUCCESS != (func)) {    \
            std::cerr << msg;       \
            return ERROR;           \
        }                           \
    } while (0)

constexpr std::size_t WORD_SIZE = 64UL;
constexpr std::size_t VMSL_WORD_SIZE = 56UL;
constexpr std::size_t AVX512_IFMA_WORD_SIZE = 52UL;

#if WORD_SIZE == 64
constexpr uint64_t WORD_SIZE_MASK = static_cast<uint64_t>(-1);
#else
constexpr uint64_t WORD_SIZE_MASK = (1UL << WORD_SIZE) - 1;
#endif

inline uint64_t highWord(uint64_t x) {
    return x >> WORD_SIZE;
}

inline uint64_t lowWord(uint64_t x) {
    return x & WORD_SIZE_MASK;
}

constexpr uint64_t VMSL_WORD_SIZE_MASK = (1UL << VMSL_WORD_SIZE) - 1;

inline uint64_t highVmslWord(uint64_t x) {
    return static_cast<uint64_t>(static_cast<__uint128_t>(x) >> VMSL_WORD_SIZE);
}

inline uint64_t lowVmslWord(uint64_t x) {
    return x & VMSL_WORD_SIZE_MASK;
}

constexpr uint64_t AVX512_IFMA_WORD_SIZE_MASK = (1UL << AVX512_IFMA_WORD_SIZE) - 1;
constexpr uint64_t AVX512_IFMA_MAX_MODULUS = 49UL;
constexpr uint64_t AVX512_IFMA_MAX_MODULUS_MASK = ~(1UL << AVX512_IFMA_MAX_MODULUS) - 1;

// Check whether N=2^m where m is odd by masking it.
constexpr uint64_t ODD_POWER_MASK = 0xaaaaaaaaaaaaaaaa;
constexpr uint64_t REM1_POWER_MASK = 0x2222222222222222;
constexpr uint64_t REM2_POWER_MASK = 0x4444444444444444;
constexpr uint64_t REM3_POWER_MASK = 0x8888888888888888;

inline bool hasEvenPower(uint64_t n) {
    return !(n & ODD_POWER_MASK);
}

inline bool hasRem1Power(uint64_t n) {
    return n & REM1_POWER_MASK;
}

inline bool hasRem2Power(uint64_t n) {
    return n & REM2_POWER_MASK;
}

inline bool hasRem3Power(uint64_t n) {
    return n & REM3_POWER_MASK;
}

} // namespace ntt
