// SPDX-License-Identifier: Apache-2.0
// MXCore testbench banked TCDM helpers.
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace mxcore::tb {

inline constexpr unsigned BankBytes = 4;
inline constexpr unsigned BankCount = 16;
inline constexpr unsigned BeatBytes = BankBytes * BankCount;
inline constexpr uint32_t AllBankGrants = (1u << BankCount) - 1;

struct MemoryBeat {
    uint64_t address = 0;
    unsigned size = 0;
    bool write = false;
    std::array<uint8_t, BeatBytes> data{};

    uint64_t word_address() const { return address & ~uint64_t(BankBytes - 1); }

    bool valid(std::size_t memory_size) const {
        const uint64_t base = word_address();
        return size && size <= BeatBytes && base <= memory_size && BeatBytes <= memory_size - base;
    }

    uint8_t byte_enable(unsigned bank) const {
        const unsigned first = bank * BankBytes;
        const unsigned count = first < size ? std::min(BankBytes, size - first) : 0;
        return uint8_t((1u << count) - 1);
    }

    // mxcore_hwpe_wrap broadcasts req to all 16 ports and ANDs their grants.
    // A partially granted beat remains asserted: writes may repeat on a bank.
    bool accept(std::vector<uint8_t> &memory, uint32_t grants) {
        const uint64_t base = word_address();
        for (unsigned bank = 0; bank < BankCount; ++bank) {
            if (!(grants & (1u << bank))) continue;
            const uint8_t enables = byte_enable(bank);
            for (unsigned byte = 0; byte < BankBytes; ++byte) {
                const unsigned offset = bank * BankBytes + byte;
                if (write) {
                    if (enables & (1u << byte)) memory[base + offset] = data[offset];
                } else {
                    data[offset] = memory[base + offset];
                }
            }
        }
        return (grants & AllBankGrants) == AllBankGrants;
    }
};

// Counter-based sampling makes each cycle/bank repeatable, including idle cycles.
// Like tb_dummy_memory, compare a sample in [0,1000]/1000 against ProbStall.
inline uint32_t sample_bank_grants(uint64_t cycle, double probability, uint32_t seed) {
    uint32_t grants = 0;
    for (unsigned bank = 0; bank < BankCount; ++bank) {
        uint64_t value = uint64_t(seed) ^ (cycle * 0x9e3779b97f4a7c15ULL) ^
                         ((uint64_t(bank) + 1) * 0xd1b54a32d192ed03ULL);
        value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
        value ^= value >> 31;
        if (double(value % 1001) / 1000.0 >= probability) grants |= 1u << bank;
    }
    return grants;
}

} // namespace mxcore::tb
