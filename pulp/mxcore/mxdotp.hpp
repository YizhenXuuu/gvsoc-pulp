#pragma once

#include "fp8.hpp"

#include <climits>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace mxcore {

// Functional dot-product unit. The owning GVSoC component/PE passes its configuration
// to the constructor; each instance keeps the same vector size for all operations.
class MXDotp {
public:
    using DotpWide = __int128;

    // FP8/FP32 format and fixed accumulator-layout constants, independent of lane count.
    static constexpr unsigned DotpFractionBits = Fp8FractionBits;
    static constexpr unsigned DotpFp32FractionBits = Fp32FractionBits;
    static constexpr unsigned DotpPrecisionBits = DotpFractionBits + 1;
    static constexpr unsigned DotpFp32PrecisionBits = DotpFp32FractionBits + 1;
    static constexpr unsigned DotpProductBits = 2 * DotpPrecisionBits + 1;
    static constexpr unsigned DotpAnchor = 34;
    static constexpr unsigned DotpIntegerBits = 32;
    static constexpr unsigned DotpProductShiftWidth = 1 + DotpIntegerBits + DotpAnchor;
    static constexpr int DotpProductShiftOffset = DotpAnchor - 2 * DotpFractionBits;
    static constexpr int DotpFp32Bias = Fp32Bias;
    static constexpr int DotpScaleBias = 127;
    static constexpr unsigned DotpWideBits = sizeof(DotpWide) * CHAR_BIT;

    // Derived from vector_size using the same formulas as fpnew_mxdotp_multi.sv.
    // Callers configure vector_size; the other fields are computed by the constructor.
    struct Parameters {
        std::size_t vector_size;
        unsigned vector_bits;
        unsigned sum_width;
        unsigned fixed_sum_width;
        unsigned extended_sum_width;
        int exponent_major_offset;
        int max_acc_shift;
    };

    struct Inputs {
        explicit Inputs(std::size_t vector_size) : a(vector_size), b(vector_size) {}

        std::vector<uint8_t> a{};
        std::vector<uint8_t> b{};
        uint8_t scale_a = DotpScaleBias; // Default scale exponent is zero.
        uint8_t scale_b = DotpScaleBias;
        uint32_t d_bits = 0; // Raw FP32 bits of input D.
    };

    struct LaneTrace {
        unsigned significand_a = 0;
        unsigned significand_b = 0;
        int product_signed = 0; // significand_a * significand_b with sign

        int exponent_a = 0;
        int exponent_b = 0;
        unsigned shift_amount = 0;// 34 - 2 * DotpFractionBits + (exponent_a + exponent_b) = 28 + (exponent_a + exponent_b)

        DotpWide shifted_product = 0; // product_signed << shift_amount
    };

    struct Result {
        explicit Result(std::size_t vector_size) : lanes(vector_size) {}

        std::vector<LaneTrace> lanes{};
        // for i_scale_adder
        int scale_exp_a = 0;
        int scale_exp_b = 0;
        int scale_sum = 0;
        int exponent_major = 0; // scale_sum + parameters().exponent_major_offset
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

    explicit MXDotp(std::size_t vector_size);

    const Parameters &parameters() const noexcept { return parameters_; }
    Inputs make_inputs() const { return Inputs(parameters_.vector_size); }
    Result compute(const Inputs &input, Fp8Format format) const;

private:
    static Parameters make_parameters(std::size_t vector_size);
    Parameters parameters_;
};

} // namespace mxcore
