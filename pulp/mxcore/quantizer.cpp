// SPDX-License-Identifier: Apache-2.0
#include "quantizer.hpp"
#include "input_fmt.hpp"

#include <algorithm>
#include <stdexcept>

namespace mxcore {

unsigned output_bits(OutputFormat format) {
    switch (format) {
    case OutputFormat::FP32: return 32;
    case OutputFormat::E5M2: return 8;
    case OutputFormat::E4M3: return 8;
    case OutputFormat::E2M1: return 4;
    case OutputFormat::BF16: return 16;
    }
    throw std::invalid_argument("Unsupported MXCore output format");
}

bool output_has_scale(OutputFormat format) {
    return format == OutputFormat::E5M2 || format == OutputFormat::E4M3 || format == OutputFormat::E2M1;
}

uint16_t quantize_bf16(uint32_t value) {
    if ((value & 0x7f800000u) == 0x7f800000u && (value & 0x007fffffu)) {
        //positive or negative NaN
        return uint16_t(((value >> 16) & 0x8000u) | 0x7fc0u);
    }
    return uint16_t((value + 0x7fffu + ((value >> 16) & 1u)) >> 16);
}

static uint8_t quantize_fp4_element(uint32_t value, uint8_t scale) {
    const auto f = decode_fp32(value);
    uint8_t fields = 0;
    if (f.nan) {
        fields = 0;
    } else if (f.inf) {
        fields = 7;
    } else if (f.exponent || f.fraction) {
        const int scaled = (f.exponent ? int(f.exponent) - 127 : -126) - (int(scale) - 127);
        if (scaled > 2) {
            fields = 7;
        } else {
            const unsigned shift = std::clamp(-scaled, 0, 24);
            const uint32_t mantissa = ((f.exponent ? 0x800000u : 0) | f.fraction) >> shift;
            uint32_t rounded = mantissa >> 22;
            const bool guard = (mantissa >> 21) & 1u;
            const bool sticky = mantissa & 0x1fffffu;
            rounded += guard && (sticky || (rounded & 1));
            if (!shift) {
                const int exponent = scaled + 1 + ((rounded >> 2) & 1);
                fields = exponent > 3 ? 7 : uint8_t((exponent << 1) | (rounded & 1));
            } else {
                fields = (rounded & 2) ? 2 : rounded & 1;
            }
        }
    }
    return uint8_t((unsigned(f.sign) << 3) | fields);
}
static uint8_t quantize_e5m2_element(uint32_t value, uint8_t scale) {
    const auto f = decode_fp32(value);
    uint8_t fields = 0;
    if (f.nan) {
        fields = 0x7d;
    } else if (f.inf) {
        fields = 0x7b;
    } else if (f.exponent || f.fraction) {
        const int scaled = (f.exponent ? int(f.exponent) - 127 : -126) - (int(scale) - 127);
        if (scaled > 15) {
            fields = 0x7b;
        } else {
            const unsigned shift = std::clamp(-14 - scaled, 0, 24);
            const uint32_t mantissa = ((f.exponent ? 0x800000u : 0) | f.fraction) >> shift;
            uint32_t rounded = mantissa >> 21;
            const bool guard = (mantissa >> 20) & 1u;
            const bool sticky = mantissa & 0xfffffu;
            rounded += guard && (sticky || (rounded & 1));
            if (!shift) {
                const int exponent = scaled + 15 + ((rounded >> 3) & 1);
                fields = exponent >= 31 ? 0x7b : uint8_t((exponent << 2) | (rounded & 3));
            } else {
                fields = (rounded & 4) ? 4 : rounded & 3;
            }
        }
    }
    return uint8_t((unsigned(f.sign) << 7) | fields);
}
static uint8_t quantize_e4m3_element(uint32_t value, uint8_t scale) {
    const auto f = decode_fp32(value);
    uint8_t fields = 0;
    if (f.nan) {
        fields = 0x7f;
    } else if (f.inf) {
        fields = 0x7e;
    } else if (f.exponent || f.fraction) {
        const int scaled = (f.exponent ? int(f.exponent) - 127 : -126) - (int(scale) - 127);
        if (scaled > 8) {
            fields = 0x7e;
        } else {
            const unsigned shift = std::clamp(-6 - scaled, 0, 24);
            const uint32_t mantissa = ((f.exponent ? 0x800000u : 0) | f.fraction) >> shift;
            uint32_t rounded = mantissa >> 20;
            const bool guard = (mantissa >> 19) & 1u;
            const bool sticky = mantissa & 0x7ffffu;
            rounded += guard && (sticky || (rounded & 1));
            if (!shift) {
                const int exponent = scaled + 7 + ((rounded >> 4) & 1);
                const bool overflow = exponent > 15 || (exponent == 15 && (rounded & 7) == 7);
                fields = overflow ? 0x7e : uint8_t((exponent << 3) | (rounded & 7));
            } else {
                fields = (rounded & 8) ? 8 : rounded & 7;
            }
        }
    }
    return uint8_t((unsigned(f.sign) << 7) | fields);
}

struct BlockScale {
    uint8_t value;
    bool poison;
};
static BlockScale compute_block_scale(const std::array<uint32_t, 32> &block, int maximum_exponent) {
    unsigned maximum = 0;
    bool special = false;
    for (uint32_t value : block) {
        const unsigned exponent = (value >> 23) & 255;
        maximum = std::max(maximum, exponent);
        special |= exponent == 255;
    }
    const int exponent = int(maximum) - 127 - maximum_exponent;
    return {uint8_t(std::clamp(exponent, -127, 127) + 127), special || exponent > 127};
}
static QuantizedBlock quantize_fp4_block(const std::array<uint32_t, 32> &block, bool enable_poison) {
    const auto scale = compute_block_scale(block, 2);
    const bool poisoned = enable_poison && scale.poison;
    QuantizedBlock result{std::vector<uint8_t>(16, 0), uint8_t(poisoned ? 0xff : scale.value)};
    for (unsigned i = 0; i < block.size(); ++i) {
        const uint8_t value = poisoned ? 0 : quantize_fp4_element(block[i], scale.value);
        result.data[i / 2] |= value << (4 * (i % 2));
    }
    return result;
}
static QuantizedBlock quantize_e5m2_block(const std::array<uint32_t, 32> &block, bool enable_poison) {
    const auto scale = compute_block_scale(block, 15);
    const bool poisoned = enable_poison && scale.poison;
    QuantizedBlock result{std::vector<uint8_t>(32, 0), scale.value};
    for (unsigned i = 0; i < block.size(); ++i)
        result.data[i] = poisoned ? 0x7d : quantize_e5m2_element(block[i], scale.value);
    return result;
}
static QuantizedBlock quantize_e4m3_block(const std::array<uint32_t, 32> &block, bool enable_poison) {
    const auto scale = compute_block_scale(block, 8);
    const bool poisoned = enable_poison && scale.poison;
    QuantizedBlock result{std::vector<uint8_t>(32, 0), scale.value};
    for (unsigned i = 0; i < block.size(); ++i)
        result.data[i] = poisoned ? 0x7f : quantize_e4m3_element(block[i], scale.value);
    return result;
}
QuantizedBlock quantize_block(const std::array<uint32_t, 32> &block,
                              OutputFormat format, bool enable_poison) {
    switch (format) {
    case OutputFormat::E2M1: return quantize_fp4_block(block, enable_poison);
    case OutputFormat::E5M2: return quantize_e5m2_block(block, enable_poison);
    case OutputFormat::E4M3: return quantize_e4m3_block(block, enable_poison);
    default: throw std::invalid_argument("Block quantization requires E5M2, E4M3 or E2M1 output");
    }
}

} // namespace mxcore
