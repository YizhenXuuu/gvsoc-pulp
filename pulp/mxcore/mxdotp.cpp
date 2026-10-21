// SPDX-License-Identifier: Apache-2.0
#include "mxdotp.hpp"

#include <stdexcept>

namespace mxcore {
namespace {

struct Encoding {
    int exponent_bits;
    int mantissa_bits;
    int bias;
};

Encoding get_fp8_info(Fp8Format format) {
    switch (format) {
        // exponent_bits, mantissa_bits, bias
        case Fp8Format::E5M2: return {5, 2, 15};
        case Fp8Format::E4M3: return {4, 3, 7};
    }
    throw std::invalid_argument("FP8 format must be E5M2 (3) or MX E4M3 (5)");
}

struct Fp32Info {
    bool sign;
    unsigned exponent;
    uint32_t fraction;
    bool normal;
    bool subnormal;
    bool inf;
    bool nan;
};

Fp32Info decode_fp32(uint32_t input) {
    // FP32：1-bit sign, 8-bit exponent, 23-bit fraction
    const unsigned exponent = (input >> DotpFp32FractionBits) & 0xffu;
    const uint32_t fraction = input & ((1u << DotpFp32FractionBits) - 1);
    const bool nan = exponent == 0xff && fraction != 0;
    return {bool(input >> 31), exponent, fraction,
            exponent != 0 && exponent != 0xff,
            exponent == 0 && fraction != 0,
            exponent == 0xff && fraction == 0, nan};
}

uint8_t special_cases(const std::array<Fp8Info, DotpVectorSize> &decoded_a,
                      const std::array<Fp8Info, DotpVectorSize> &decoded_b,
                      const Fp32Info &info_d) {

    bool has_nan_input = info_d.nan;
    bool has_zero_times_infinity = false;
    bool has_positive_infinity = info_d.inf && !info_d.sign;
    bool has_negative_infinity = info_d.inf && info_d.sign;

    for (std::size_t i = 0; i < DotpVectorSize; ++i) {
        const Fp8Info &a = decoded_a[i];
        const Fp8Info &b = decoded_b[i];
        const bool a_is_nan = a.classification == FpClass::NaN;
        const bool b_is_nan = b.classification == FpClass::NaN;
        const bool a_is_zero = a.classification == FpClass::Zero;
        const bool b_is_zero = b.classification == FpClass::Zero;
        const bool a_is_infinity = a.classification == FpClass::Infinity;
        const bool b_is_infinity = b.classification == FpClass::Infinity;

        if (a_is_nan || b_is_nan) {
            has_nan_input = true;
        }

        if ((a_is_zero && b_is_infinity) || (a_is_infinity && b_is_zero)) {
            has_zero_times_infinity = true;
        }

        if (a_is_infinity || b_is_infinity) {
            const bool product_is_negative = a.sign != b.sign;
            if (product_is_negative) {
                has_negative_infinity = true;
            } else {
                has_positive_infinity = true;
            }
        }
    }

    bool is_special = false;
    bool invalid_operation = false; // NV flag.
    bool is_infinity = false;

    if (has_zero_times_infinity) {
        // 1. 0*Inf or Inf*0 produces NaN and sets NV.
        is_special = true;
        invalid_operation = true;
    } else if (has_nan_input) {
        // 2. Input NaN produces NaN without NV
        is_special = true;
    } else if (has_positive_infinity && has_negative_infinity) {
        // 3. +Inf and -Inf contributions produce NaN and set NV.
        is_special = true;
        invalid_operation = true;
    } else if (has_positive_infinity || has_negative_infinity) {
        // 4. Infinity contributions with only one sign produce that infinity.
        is_special = true;
        is_infinity = true;
    }

    // Keep the RTL's raw inf_sign bit even when unused (finite or NaN result).
    const bool infinity_is_negative = !has_positive_infinity;
    // Pack {is_special, NV, is_inf, inf_sign} only after deciding the result.
    return uint8_t((unsigned(is_special) << 3) | (unsigned(invalid_operation) << 2)
                 | (unsigned(is_infinity) << 1) | unsigned(infinity_is_negative));
}

} // namespace

Fp8Info classify_fp8(uint8_t bits, Fp8Format format) {
    const auto fmt = get_fp8_info(format);
    const int mantissa_mask = (1 << fmt.mantissa_bits) - 1;
    const int exponent_mask = (1 << fmt.exponent_bits) - 1;
    const uint8_t exponent = (bits >> fmt.mantissa_bits) & exponent_mask;
    const uint8_t mantissa = bits & mantissa_mask;
    Fp8Info info{bits >> (DotpVectorSize - 1), exponent, mantissa, FpClass::Normal, 0, 0};

    if (exponent == exponent_mask){
        if (format == Fp8Format::E5M2){
            info.classification = (mantissa == 0) ? FpClass::Infinity : FpClass::NaN;
        } else if (format == Fp8Format::E4M3){
            if (mantissa == mantissa_mask) {
                info.classification = FpClass::NaN;
            } 
        }
    }
    
    if (exponent == 0) {
        info.classification = (mantissa == 0) ? FpClass::Zero : FpClass::Subnormal;
    }

    // 3 bits fraction for both E5M2 and E4M3, the significand is 1 for normal, 0 for subnormal/zero.
    info.significand = (unsigned(info.classification == FpClass::Normal) << DotpFractionBits)
                     | (unsigned(mantissa) << (DotpFractionBits - fmt.mantissa_bits));

    info.unbiased_exponent = int(exponent)
                           + int(info.classification == FpClass::Subnormal) - fmt.bias;

    return info;
}

DotpComputeResult mxdotp_compute(const DotpComputeInputs &input, Fp8Format format) {
    std::array<Fp8Info, DotpVectorSize> decoded_a{};
    std::array<Fp8Info, DotpVectorSize> decoded_b{};

    for (std::size_t i = 0; i < DotpVectorSize; ++i) {
        decoded_a[i] = classify_fp8(input.a[i], format);
        decoded_b[i] = classify_fp8(input.b[i], format);
    }

    DotpComputeResult result;
    result.d_bits = input.d_bits;
    const Fp32Info info_d = decode_fp32(input.d_bits);
    // RTL evaluates special cases in parallel; the functional model resolves them first.
    result.special_raw = special_cases(decoded_a, decoded_b, info_d);
    if ((result.special_raw & 0x8u) != 0) {
        return result;
    }

    // 1. i_vector_multiplier_fp8 + i_product_exponent_fp8 + i_product_align_fp8
    for (std::size_t i = 0; i < DotpVectorSize; ++i) {
        const Fp8Info &a = decoded_a[i];
        const Fp8Info &b = decoded_b[i];
        DotpLaneTrace &lane = result.lanes[i];
        lane.significand_a = a.significand;
        lane.significand_b = b.significand;
        lane.exponent_a = a.unbiased_exponent;
        lane.exponent_b = b.unbiased_exponent;
        const int exponent_product = lane.exponent_a + lane.exponent_b;
        lane.shift_amount = unsigned(DotpProductShiftOffset + exponent_product);
        // Special cases were resolved above, so zero operand makes this term zero.
        // Trace fields retain their defaults for the skipped multiplication and alignment.
        if (a.classification == FpClass::Zero || b.classification == FpClass::Zero) {
            continue;
        }
        const int product_unsigned_ab = int(lane.significand_a * lane.significand_b);
        lane.product_signed = a.sign != b.sign ? -product_unsigned_ab : product_unsigned_ab;
        // FP8 aligned products fit in 67 signed bits. Keep the sign through multiplication.
        lane.shifted_product = lane.shift_amount >= DotpProductShiftWidth ? 0
            : DotpWide(lane.product_signed) * (DotpWide(1) << lane.shift_amount);
        result.sum_product += lane.shifted_product;
    }

    // 2. i_scale_adder
    // exponent_major = (rscale_a-127) + (rscale_b-127) + 187
    result.scale_exp_a = int8_t(int(input.scale_a) - DotpScaleBias);
    result.scale_exp_b = int8_t(int(input.scale_b) - DotpScaleBias);
    result.scale_sum = result.scale_exp_a + result.scale_exp_b;
    result.exponent_major = result.scale_sum + DotpExponentMajorOffset;
    result.main_lsb_exponent = result.scale_sum - int(DotpAnchor); // s-34
    result.extended_lsb_exponent = result.main_lsb_exponent - int(DotpFp32PrecisionBits); // s-58

    // 3. i_accumulator_prep
    const uint32_t integer_d = (uint32_t(info_d.normal) << DotpFp32FractionBits) | info_d.fraction;
    result.signed_integer_d = info_d.sign ? -int32_t(integer_d) : int32_t(integer_d);
    // r = 34 - 23 + E + subnormal - 127 - s = 11 + e_D - s
    result.accumulator_shift_amount = int(DotpAnchor) - int(DotpFp32FractionBits)
        + DotpExponentMajorOffset - result.exponent_major
        + int(info_d.exponent) + int(info_d.subnormal) - DotpFp32Bias;

    // 4. i_accumulator_shift
    constexpr int max_acc_shift = DotpFixedSumWidth - DotpFp32PrecisionBits - 1; // 95 - 24 - 1 = 70
    const int amount = result.accumulator_shift_amount;
    if (amount > max_acc_shift) {
        result.result_is_accumulator_uncond = true;
    } else if (amount >= 0) {
        result.accumulator_shifted = DotpWide(result.signed_integer_d) * (DotpWide(1) << unsigned(amount));
    } else {
        const unsigned right = unsigned(-amount);
        const unsigned funnel_width = 2 * DotpFp32PrecisionBits + 1; // 49
        const DotpWide remaining_unit = DotpWide(1) << DotpFp32PrecisionBits;
        const DotpWide funnel_input = DotpWide(result.signed_integer_d) * remaining_unit;
        const DotpWide funnel = right >= funnel_width
            ? (funnel_input < 0 ? -1 : 0)
            : (funnel_input >> right);
        result.accumulator_shifted = funnel >> DotpFp32PrecisionBits;
        result.accumulator_remaining = uint32_t(funnel - result.accumulator_shifted * remaining_unit);

        if (right > DotpFp32PrecisionBits) {
            const unsigned lost = right - DotpFp32PrecisionBits;
            const unsigned mask_width = (lost < DotpFp32PrecisionBits + 1) ? 
                                        lost : DotpFp32PrecisionBits + 1;
            const int32_t discarded_unit = int32_t(1) << mask_width;
            result.accumulator_sticky = result.signed_integer_d % discarded_unit != 0;
            result.result_is_accumulator_if_sop_zero = true;
        }
    }

    // 5. i_fused_sop_accumulator
    result.sum_product_accumulator = result.sum_product + result.accumulator_shifted;
    result.sum_product_is_zero = result.sum_product == 0;
    const DotpWide extended_unit = DotpWide(1) << DotpFp32PrecisionBits;
    result.sum_product_accumulator_extended =
        result.sum_product_accumulator * extended_unit + DotpWide(result.accumulator_remaining);
    result.result_is_accumulator = result.result_is_accumulator_uncond
        || (result.result_is_accumulator_if_sop_zero && result.sum_product_is_zero);
    return result;
}

} // namespace mxcore
