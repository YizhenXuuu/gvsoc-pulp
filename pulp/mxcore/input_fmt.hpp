#pragma once

#include <cstddef>
#include <cstdint>

namespace mxcore {

// Lossless common significand representation for E5M2, E4M3 and E2M1.
constexpr unsigned InputFractionBits = 3;
constexpr unsigned Fp32FractionBits = 23;
constexpr int      Fp32Bias = 127;

// Values match cvfpu/src/fpnew_pkg.sv. E4M3 uses MX classification rules.
enum class InputFormat : uint8_t {
    E5M2 = 3,
    E4M3 = 5,
    E2M1 = 8
};

// Encodings match cvfpu/src/fpnew_pkg.sv, ROD is a CVFPU extension.
enum class RoundingMode : uint8_t {
    RNE
};
enum class FpClass : uint8_t {
    Zero,
    Subnormal,
    Normal,
    Infinity,
    NaN
};

// Bit positions match fpnew_pkg::status_t: {NV, DZ, OF, UF, NX} = bits [4:0].
enum FpStatus : uint8_t {
    NX = 1u << 0,
    UF = 1u << 1,
    OF = 1u << 2,
    DZ = 1u << 3,
    NV = 1u << 4
};

struct InputEncoding {
    int exponent_bits;
    int mantissa_bits;
    int bias;
};

InputEncoding get_input_encoding(InputFormat format);

struct InputInfo {
    bool    sign;
    uint8_t exponent; // exponent before bias subtraction.
    uint8_t mantissa; // fraction bits without the implicit leading bit.
    FpClass classification;

    // Arithmetic fields prepared by classify_input, using InputFractionBits fraction bits.
    // The integer significand: leading 1 for normal, 0 for subnormal/zero, followed by fraction bits.
    unsigned significand;
    int      unbiased_exponent; // exponent + is_subnormal - bias.
};

// Decode one element: the low four bits for E2M1, or all eight bits for FP8.
InputInfo classify_input(uint8_t bits, InputFormat format);

struct Fp32Info {
    bool     sign;
    unsigned exponent;
    uint32_t fraction;
    bool     normal;
    bool     subnormal;
    bool     inf;
    bool     nan;
};

Fp32Info decode_fp32(uint32_t input);

uint8_t special_cases(const InputInfo *decoded_a, const InputInfo *decoded_b, std::size_t vector_size,
                      const Fp32Info &info_d);

} // namespace mxcore
