// SPDX-License-Identifier: Apache-2.0
#include "fp8.hpp"

#include <stdexcept>

namespace mxcore {

Encoding get_fp8_info(Fp8Format format) {
    switch (format) {
    // exponent_bits, mantissa_bits, bias
    case Fp8Format::E5M2:
        return {5, 2, 15};
    case Fp8Format::E4M3:
        return {4, 3, 7};
    }
    throw std::invalid_argument("FP8 format must be E5M2 (3) or MX E4M3 (5)");
}

Fp32Info decode_fp32(uint32_t input) {
    // FP32：1-bit sign, 8-bit exponent, 23-bit fraction
    const unsigned exponent = (input >> Fp32FractionBits) & 0xffu;
    const uint32_t fraction = input & ((1u << Fp32FractionBits) - 1);
    const bool     nan = exponent == 0xff && fraction != 0;
    return {bool(input >> 31),
            exponent,
            fraction,
            exponent != 0 && exponent != 0xff,
            exponent == 0 && fraction != 0,
            exponent == 0xff && fraction == 0,
            nan};
}

uint8_t special_cases(const Fp8Info *decoded_a, const Fp8Info *decoded_b, std::size_t vector_size,
                      const Fp32Info &info_d) {

    bool has_nan_input = info_d.nan;
    bool has_zero_times_infinity = false;
    bool has_positive_infinity = info_d.inf && !info_d.sign;
    bool has_negative_infinity = info_d.inf && info_d.sign;

    for (std::size_t i = 0; i < vector_size; ++i) {
        const Fp8Info &a = decoded_a[i];
        const Fp8Info &b = decoded_b[i];
        const bool     a_is_nan = a.classification == FpClass::NaN;
        const bool     b_is_nan = b.classification == FpClass::NaN;
        const bool     a_is_zero = a.classification == FpClass::Zero;
        const bool     b_is_zero = b.classification == FpClass::Zero;
        const bool     a_is_infinity = a.classification == FpClass::Infinity;
        const bool     b_is_infinity = b.classification == FpClass::Infinity;

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
        // 2. All input NaNs produce NaN without setting NV.
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
    return uint8_t((unsigned(is_special) << 3) | (unsigned(invalid_operation) << 2) |
                   (unsigned(is_infinity) << 1) | unsigned(infinity_is_negative));
}

Fp8Info classify_fp8(uint8_t bits, Fp8Format format) {
    const auto    fmt = get_fp8_info(format);
    const int     mantissa_mask = (1 << fmt.mantissa_bits) - 1;
    const int     exponent_mask = (1 << fmt.exponent_bits) - 1;
    const uint8_t exponent = (bits >> fmt.mantissa_bits) & exponent_mask;
    const uint8_t mantissa = bits & mantissa_mask;
    Fp8Info       info{bool(bits & 0x80u), exponent, mantissa, FpClass::Normal, 0, 0};

    if (exponent == exponent_mask) {
        if (format == Fp8Format::E5M2) {
            info.classification = (mantissa == 0) ? FpClass::Infinity : FpClass::NaN;
        } else if (format == Fp8Format::E4M3) {
            if (mantissa == mantissa_mask) {
                info.classification = FpClass::NaN;
            }
        }
    }

    if (exponent == 0) {
        info.classification = (mantissa == 0) ? FpClass::Zero : FpClass::Subnormal;
    }

    // 3 bits fraction for both E5M2 and E4M3, the significand is 1 for normal, 0 for subnormal/zero.
    info.significand = (unsigned(info.classification == FpClass::Normal) << Fp8FractionBits) |
                       (unsigned(mantissa) << (Fp8FractionBits - fmt.mantissa_bits));

    info.unbiased_exponent = int(exponent) + int(info.classification == FpClass::Subnormal) - fmt.bias;

    return info;
}

} // namespace mxcore
