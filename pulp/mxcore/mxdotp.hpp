#pragma once

#include "fp8.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace mxcore {

using DotpWide = __int128;

constexpr std::size_t DotpVectorSize = 8;
constexpr unsigned DotpFractionBits = 3;
constexpr unsigned DotpFp32FractionBits = 23;

constexpr unsigned DotpPrecisionBits = DotpFractionBits + 1; // 4
constexpr unsigned DotpFp32PrecisionBits = DotpFp32FractionBits + 1; // 24

constexpr unsigned DotpProductBits = 2 * DotpPrecisionBits + 1; // 9, signed

constexpr unsigned DotpAnchor = 34; // Fractional point position
constexpr unsigned DotpIntegerBits = 32;
constexpr unsigned DotpProductShiftWidth = 1 + DotpIntegerBits + DotpAnchor; // 67
constexpr int DotpProductShiftOffset = DotpAnchor - 2 * DotpFractionBits; // 28

constexpr int DotpFp32Bias = 127;

// fpnew_mxdotp_multi.sv:129-132
constexpr unsigned DotpSumWidth = DotpProductShiftWidth + 3; // 70
constexpr unsigned DotpFixedSumWidth = 1 + DotpFp32PrecisionBits + 1 + (DotpSumWidth - 1); // 95
constexpr unsigned DotpExtendedSumWidth = DotpFixedSumWidth + DotpFp32PrecisionBits; // 119

// fpnew_mxdotp_multi_modules.sv:301
constexpr int DotpScaleBias = 127;

// fpnew_mxdotp_multi_modules.sv:468-491
constexpr int DotpExponentMajorOffset = DotpFp32Bias - int(DotpAnchor) + int(DotpFixedSumWidth) - 1; // 187

static_assert(DotpExtendedSumWidth < 128, "DotpWide must hold the extended RTL sum");

struct DotpComputeInputs {
    std::array<uint8_t, DotpVectorSize> a{};
    std::array<uint8_t, DotpVectorSize> b{};
    uint8_t scale_a = 0x7f; //default： scale - scale bias = 0
    uint8_t scale_b = 0x7f;
    uint32_t d_bits = 0; // Raw FP32 bits of input D.
};

struct DotpLaneTrace {
    unsigned significand_a = 0; 
    unsigned significand_b = 0;
    int product_signed = 0; // significand_a * significand_b with sign

    int exponent_a = 0; 
    int exponent_b = 0;
    unsigned shift_amount = 0;// 34 - 2 * DotpFractionBits + (exponent_a + exponent_b) = 28 + (exponent_a + exponent_b)
    
    DotpWide shifted_product = 0; // product_signed << shift_amount
};

struct DotpComputeResult {
    std::array<DotpLaneTrace, DotpVectorSize> lanes{};
    // for i_scale_adder
    int scale_exp_a = 0;
    int scale_exp_b = 0;
    int scale_sum = 0;
    int exponent_major = 0; // scale_sum + 187
    int main_lsb_exponent = 0;     // scale_sum - 34
    int extended_lsb_exponent = 0; // scale_sum - 34 - 24

    // for i_accumulator_prep
    int32_t signed_integer_d = 0; 
    int accumulator_shift_amount = 0;

    // for i_accumulator_shift
    DotpWide accumulator_shifted = 0;
    uint32_t accumulator_remaining = 0; 
    bool accumulator_sticky = false;
    bool result_is_accumulator_uncond = false;
    bool result_is_accumulator_if_sop_zero = false;
    bool result_is_accumulator = false; 

    DotpWide sum_product = 0; 
    bool sum_product_is_zero = false; 
    DotpWide sum_product_accumulator = 0; 
    DotpWide sum_product_accumulator_extended = 0; 

    // {is_special, NV, is_inf, inf_sign}。
    // When bit 3 is set, the function returns before arithmetic; its traces and flags
    // remain at their defaults. Select the special result using this verdict.
    uint8_t special_raw = 0;

    uint32_t d_bits = 0; // Original FP32 bits of D, preserved for bypass.
};

DotpComputeResult mxdotp_compute(const DotpComputeInputs &input, Fp8Format format);

} // namespace mxcore
