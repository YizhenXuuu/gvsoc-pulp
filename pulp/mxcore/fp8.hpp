#pragma once

#include <cstdint>

namespace mxcore {

// Values match cvfpu/src/fpnew_pkg.sv. E4M3 uses MX classification rules.
enum class Fp8Format : uint8_t { E5M2 = 3, E4M3 = 5 };

// Encodings match cvfpu/src/fpnew_pkg.sv, ROD is a CVFPU extension.
enum class RoundingMode : uint8_t {
    RNE
};
enum class FpClass : uint8_t { Zero, Subnormal, Normal, Infinity, NaN };

// Bit positions match fpnew_pkg::status_t: {NV, DZ, OF, UF, NX} = bits [4:0].
enum FpStatus : uint8_t {
    NX = 1u << 0, UF = 1u << 1, OF = 1u << 2,
    DZ = 1u << 3, NV = 1u << 4
};

struct Fp8Info {
    bool sign;
    uint8_t exponent; // exponent before bias subtraction.
    uint8_t mantissa; // fraction bits without the implicit leading bit.
    FpClass classification;

    // Arithmetic fields prepared by classify_fp8, using 3 fraction bits.
    // The integer significand: leading 1 for normal, 0 for subnormal/zero, followed by fraction bits.
    unsigned significand;
    int unbiased_exponent; // exponent + is_subnormal - bias.
};

Fp8Info classify_fp8(uint8_t bits, Fp8Format format);

} // namespace mxcore
