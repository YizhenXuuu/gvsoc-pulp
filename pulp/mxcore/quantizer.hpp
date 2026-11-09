#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace mxcore {

enum class OutputFormat : uint8_t {
    FP32,
    E5M2,
    E2M1,
    BF16,
    E4M3
};

unsigned output_bits(OutputFormat format);
bool output_has_scale(OutputFormat format);
uint16_t quantize_bf16(uint32_t value);

struct QuantizedBlock {
    std::vector<uint8_t> data;
    uint8_t scale;
};

// Port of block_scale and the SAT E5M2/E2M1 RTL quantizers, including poison.
QuantizedBlock quantize_block(const std::array<uint32_t, 32> &block,
                              OutputFormat format, bool enable_poison);

} // namespace mxcore
