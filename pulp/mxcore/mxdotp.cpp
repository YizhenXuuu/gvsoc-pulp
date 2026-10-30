// SPDX-License-Identifier: Apache-2.0
#include "mxdotp.hpp"

#include <stdexcept>

namespace mxcore {

MXDotp::Parameters MXDotp::make_parameters(std::size_t vector_size) {
    if (vector_size == 0) {
        throw std::invalid_argument("MXDotp vector_size must be positive");
    }

    Parameters params{};
    params.vector_size = vector_size;
    // Integer ceil(log2(vector_size)), including non-power-of-two sizes.
    for (std::size_t remaining = vector_size - 1; remaining != 0; remaining >>= 1) {
        ++params.vector_bits;
    }
    params.sum_width = DotpProductShiftWidth + params.vector_bits;
    params.fixed_sum_width = 1 + DotpFp32PrecisionBits + 1 + (params.sum_width - 1);
    params.extended_sum_width = params.fixed_sum_width + DotpFp32PrecisionBits;
    if (params.extended_sum_width >= DotpWideBits) {
        throw std::invalid_argument("MXDotp vector_size exceeds the signed wide accumulator capacity");
    }
    params.exponent_major_offset = DotpFp32Bias - int(DotpAnchor) + int(params.fixed_sum_width) - 1;
    params.max_acc_shift = int(params.fixed_sum_width) - int(DotpFp32PrecisionBits) - 1;
    return params;
}

MXDotp::MXDotp(std::size_t vector_size) : parameters_(make_parameters(vector_size)) {}

MXDotp::Result MXDotp::compute(const Inputs &input, Fp8Format format) const {
    if (input.a.size() != parameters_.vector_size || input.b.size() != parameters_.vector_size) {
        throw std::invalid_argument("MXDotp input lengths must match the configured vector_size");
    }
    std::vector<Fp8Info> decoded_a(parameters_.vector_size);
    std::vector<Fp8Info> decoded_b(parameters_.vector_size);

    for (std::size_t i = 0; i < parameters_.vector_size; ++i) {
        decoded_a[i] = classify_fp8(input.a[i], format);
        decoded_b[i] = classify_fp8(input.b[i], format);
    }

    Result result(parameters_.vector_size);
    result.d_bits = input.d_bits;
    const Fp32Info info_d = decode_fp32(input.d_bits);
    // RTL evaluates special cases in parallel; the functional model resolves them first.
    result.special_raw = special_cases(decoded_a.data(), decoded_b.data(), decoded_a.size(), info_d);
    if ((result.special_raw & 0x8u) != 0) {
        return result;
    }

    // 1. i_vector_multiplier_fp8 + i_product_exponent_fp8 + i_product_align_fp8
    for (std::size_t i = 0; i < parameters_.vector_size; ++i) {
        const Fp8Info &a = decoded_a[i];
        const Fp8Info &b = decoded_b[i];
        LaneTrace &lane = result.lanes[i];
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
    // Match current RTL signed 8-bit scale arithmetic, including 0xff. The RTL
    // classifier's signed comparison does not flag 0xff as NaN (covered by tests).
    // exponent_major = (rscale_a-127) + (rscale_b-127) + parameters_.exponent_major_offset
    result.scale_exp_a = int8_t(int(input.scale_a) - DotpScaleBias);
    result.scale_exp_b = int8_t(int(input.scale_b) - DotpScaleBias);
    result.scale_sum = result.scale_exp_a + result.scale_exp_b;
    result.exponent_major = result.scale_sum + parameters_.exponent_major_offset;
    result.main_lsb_exponent = result.scale_sum - int(DotpAnchor); // s-34
    result.extended_lsb_exponent = result.main_lsb_exponent - int(DotpFp32PrecisionBits); // s-58

    // 3. i_accumulator_prep
    const uint32_t integer_d = (uint32_t(info_d.normal) << DotpFp32FractionBits) | info_d.fraction;
    result.signed_integer_d = info_d.sign ? -int32_t(integer_d) : int32_t(integer_d);
    // r = 34 - 23 + E + subnormal - 127 - s = 11 + e_D - s
    result.accumulator_shift_amount = int(DotpAnchor) - int(DotpFp32FractionBits)
        + parameters_.exponent_major_offset - result.exponent_major
        + int(info_d.exponent) + int(info_d.subnormal) - DotpFp32Bias;

    // 4. i_accumulator_shift
    const int max_acc_shift = parameters_.max_acc_shift;
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
