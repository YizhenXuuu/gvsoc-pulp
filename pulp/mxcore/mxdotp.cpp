// SPDX-License-Identifier: Apache-2.0
#include "mxdotp.hpp"

#include <stdexcept>
#include <utility>

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

MXDotp::MXDotp(std::size_t vector_size) : parameters_(make_parameters(vector_size)) {
}

MXDotp::Result MXDotp::compute(const Inputs &input, InputFormat format) const {
    get_input_encoding(format);
    const bool packed_fp4 = format == InputFormat::E2M1;
    if (packed_fp4 && parameters_.vector_size != 32) {
        throw std::invalid_argument("FP4 requires VS=32 packed bytes (64 E2M1 elements)");
    }
    if (input.a.size() != parameters_.vector_size || input.b.size() != parameters_.vector_size) {
        throw std::invalid_argument("MXDotp input lengths must match the configured vector_size");
    }
    const std::size_t elements = parameters_.vector_size * (packed_fp4 ? 2 : 1);
    std::vector<InputInfo> decoded_a(elements);
    std::vector<InputInfo> decoded_b(elements);

    for (std::size_t i = 0; i < elements; ++i) {
        const std::size_t byte = packed_fp4 ? i / 2 : i;
        const unsigned shift = packed_fp4 ? 4 * (i % 2) : 0;
        decoded_a[i] = classify_input(input.a[byte] >> shift, format);
        decoded_b[i] = classify_input(input.b[byte] >> shift, format);
    }

    Result result(elements);
    result.d_bits = input.d_bits;
    const Fp32Info info_d = decode_fp32(input.d_bits);
    // RTL evaluates special cases in parallel; the functional model resolves them first.
    result.special_raw = special_cases(decoded_a.data(), decoded_b.data(), decoded_a.size(), info_d);
    // gwaihir's fpnew_mxdotp_multi_opt classifies E8M0 0xff as NaN.
    // A NaN scale overrides an infinity result as well as a finite result.
    if (input.scale_a == 0xff || input.scale_b == 0xff) {
        result.special_raw = 0x8u | (result.special_raw & 0x4u);
    }
    if ((result.special_raw & 0x8u) != 0) {
        return result;
    }

    // 1. Shared product path. FP4 significands are extended losslessly to the
    // common three fraction bits, so both nibble sets use the same 2^-34 anchor.
    for (std::size_t i = 0; i < elements; ++i) {
        const InputInfo &a = decoded_a[i];
        const InputInfo &b = decoded_b[i];
        LaneTrace     &lane = result.lanes[i];
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
        // Aligned input products fit in 67 signed bits. Keep the sign through multiplication.
        lane.shifted_product = lane.shift_amount >= DotpProductShiftWidth
                                   ? 0
                                   : DotpWide(lane.product_signed) * (DotpWide(1) << lane.shift_amount);
        result.sum_product += lane.shifted_product;
    }

    return this->accumulate(std::move(result), input);
}

MXDotp::Result MXDotp::accumulate(Result result, const Inputs &input) const {
    const Fp32Info info_d = decode_fp32(input.d_bits);
    // 2. i_scale_adder
    // Match RTL signed 8-bit scale arithmetic. The 0xff NaN code was
    // handled before this finite datapath.
    // exponent_major = (rscale_a-127) + (rscale_b-127) + parameters_.exponent_major_offset
    result.scale_exp_a = int8_t(int(input.scale_a) - DotpScaleBias);
    result.scale_exp_b = int8_t(int(input.scale_b) - DotpScaleBias);
    result.scale_sum = result.scale_exp_a + result.scale_exp_b;
    result.exponent_major = result.scale_sum + parameters_.exponent_major_offset;
    result.main_lsb_exponent = result.scale_sum - int(DotpAnchor);                        // s-34
    result.extended_lsb_exponent = result.main_lsb_exponent - int(DotpFp32PrecisionBits); // s-58

    // 3. i_accumulator_prep
    const uint32_t integer_d = (uint32_t(info_d.normal) << DotpFp32FractionBits) | info_d.fraction;
    result.signed_integer_d = info_d.sign ? -int32_t(integer_d) : int32_t(integer_d);
    // r = 34 - 23 + E + subnormal - 127 - s = 11 + e_D - s
    result.accumulator_shift_amount = int(DotpAnchor) - int(DotpFp32FractionBits) +
                                      parameters_.exponent_major_offset - result.exponent_major +
                                      int(info_d.exponent) + int(info_d.subnormal) - DotpFp32Bias;

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
        const DotpWide funnel = right >= funnel_width ? (funnel_input < 0 ? -1 : 0) : (funnel_input >> right);
        result.accumulator_shifted = funnel >> DotpFp32PrecisionBits;
        result.accumulator_remaining = uint32_t(funnel - result.accumulator_shifted * remaining_unit);

        if (right > DotpFp32PrecisionBits) {
            const unsigned lost = right - DotpFp32PrecisionBits;
            const unsigned mask_width = (lost < DotpFp32PrecisionBits + 1) ? lost : DotpFp32PrecisionBits + 1;
            const int32_t  discarded_unit = int32_t(1) << mask_width;
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
    result.result_is_accumulator = result.result_is_accumulator_uncond ||
                                   (result.result_is_accumulator_if_sop_zero && result.sum_product_is_zero);
    return result;
}

} // namespace mxcore
