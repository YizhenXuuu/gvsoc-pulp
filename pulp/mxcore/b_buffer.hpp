// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace mxcore {

// Two logical rows, following mxcore_b_buffer_ctrl for vectors B and ScaleB.
// A narrow row is loaded from a slice of the current HCI response without
// consuming that response until its last slice has been loaded.
class BBuffer {
  public:
    using Row = std::vector<uint8_t>;

    void configure(unsigned beat_bytes, unsigned row_bytes, unsigned reuse,
                   unsigned scale_factor, bool short_matrix = false) {
        if (!beat_bytes || !row_bytes || !reuse || !scale_factor ||
            (beat_bytes >= row_bytes ? beat_bytes % row_bytes : row_bytes % beat_bytes)) {
            throw std::invalid_argument("MXCore B buffer requires integral row/beat widths");
        }
        this->beat_bytes = beat_bytes;
        this->row_bytes = row_bytes;
        this->reuse = reuse;
        this->scale_factor = scale_factor;
        this->short_matrix = short_matrix;
        for (auto &row : this->rows) {
            row.assign(row_bytes, 0);
        }
        this->clear();
    }

    void clear() {
        this->full.fill(false);
        this->write_row = this->read_row = 0;
        this->write_word = this->write_slice = 0;
        this->reuse_count = this->scale_count = 0;
    }

    bool input_ready(bool input_valid) const {
        if (this->full[this->write_row]) {
            return false;
        }
        if (this->row_bytes < this->beat_bytes) {
            // RTL SLICE mode asserts ready only when the whole beat is used.
            return input_valid && (this->short_matrix ||
                                  this->write_slice + 1 == this->beat_bytes / this->row_bytes);
        }
        return true;
    }

    bool valid() const { return this->full[this->read_row]; }

    const Row &peek() const {
        if (!this->valid()) {
            throw std::logic_error("MXCore B buffer read before a complete row");
        }
        return this->rows[this->read_row];
    }

    bool empty() const {
        return !this->full[0] && !this->full[1] && !this->write_word && !this->write_slice;
    }

    // Clock both sides using old row_full, as in the RTL always_comb block.
    // Returns whether the upstream response was consumed at this edge.
    bool clock(const uint8_t *beat, bool consume_row) {
        const bool valid_before = this->valid();
        const bool load = beat && !this->full[this->write_row];
        const bool consume_beat = beat && this->input_ready(true);
        if (consume_row && !valid_before) {
            throw std::logic_error("MXCore B buffer consumption without valid");
        }

        if (load) {
            Row &row = this->rows[this->write_row];
            if (this->row_bytes < this->beat_bytes) {
                std::copy_n(beat + this->write_slice * this->row_bytes, this->row_bytes, row.begin());
                this->full[this->write_row] = true;
                this->write_row ^= 1;
                this->write_slice = consume_beat ? 0 : this->write_slice + 1;
            } else {
                std::copy_n(beat, this->beat_bytes, row.begin() + this->write_word * this->beat_bytes);
                if (++this->write_word == this->row_bytes / this->beat_bytes) {
                    this->write_word = 0;
                    this->full[this->write_row] = true;
                    this->write_row ^= 1;
                }
            }
        }

        if (consume_row && ++this->reuse_count == this->reuse) {
            this->reuse_count = 0;
            if (++this->scale_count == this->scale_factor) {
                this->scale_count = 0;
                this->full[this->read_row] = false;
                this->read_row ^= 1;
            }
        }
        return consume_beat;
    }

  private:
    unsigned beat_bytes = 0;
    unsigned row_bytes = 0;
    unsigned reuse = 1;
    unsigned scale_factor = 1;
    bool short_matrix = false;
    std::array<Row, 2> rows;
    std::array<bool, 2> full{};
    unsigned write_row = 0, read_row = 0;
    unsigned write_word = 0, write_slice = 0;
    unsigned reuse_count = 0, scale_count = 0;
};

} // namespace mxcore
