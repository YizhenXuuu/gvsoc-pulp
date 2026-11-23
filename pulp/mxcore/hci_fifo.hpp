// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <deque>
#include <cstdint>
#include <stdexcept>

namespace mxcore {

// Clocked control of one RTL hci_core_fifo channel. Queue entries identify
// requests owned by MXCore; the payload and memory callbacks stay in MXCore.
struct HciFifo {
    static constexpr unsigned Depth = 2;

    std::deque<unsigned> requests;
    std::deque<unsigned> responses;
    int held_response = -1; // hci_core_fifo's tcdm_initiator_r_valid_q register.
    int64_t request_popped_at = -1;

    void clear() {
        requests.clear();
        responses.clear();
        held_response = -1;
        request_popped_at = -1;
    }

    bool empty() const {
        return requests.empty() && responses.empty() && held_response < 0;
    }

    bool request_push_ready(int64_t cycle) const {
        // A callback can deliver grant before the FSM runs at the same edge.
        // Reconstruct the old occupancy instead of allowing full/pop bypass.
        return requests.size() + unsigned(request_popped_at == cycle) < Depth;
    }

    void push_request(unsigned slot, int64_t cycle) {
        if (!request_push_ready(cycle)) {
            throw std::logic_error("MXCore HCI request FIFO overflow");
        }
        requests.push_back(slot);
    }

    void pop_request(unsigned slot, int64_t cycle) {
        if (requests.empty() || requests.front() != slot) {
            throw std::logic_error("MXCore HCI request FIFO order mismatch");
        }
        requests.pop_front();
        request_popped_at = cycle;
    }

    bool can_issue(bool response_valid, bool consumer_ready) const {
        // hci_core_fifo.sv: req = outgoing.valid & incoming.not_full &
        // ~(incoming.almost_full & r_valid & ~target.r_ready).
        return !requests.empty() && responses.size() < Depth &&
               !(responses.size() == Depth - 1 && response_valid && !consumer_ready);
    }

    // Advance the incoming FIFO and its response holding register at one edge.
    // All ready/valid decisions use the old occupancy: a full FIFO cannot push
    // on the edge that pops it, and a new response cannot bypass an empty FIFO.
    // Returns the old head accepted by the downstream consumer, or -1.
    int clock_response(int arrival, bool consumer_ready) {
        if (arrival >= 0 && held_response >= 0) {
            throw std::logic_error("MXCore HCI response holding register overflow");
        }
        const bool push_ready = responses.size() < Depth;
        const int source = arrival >= 0 ? arrival : held_response;
        int popped = -1;
        if (consumer_ready && !responses.empty()) {
            popped = int(responses.front());
            responses.pop_front();
        }
        if (source >= 0 && push_ready) {
            responses.push_back(unsigned(source));
            held_response = -1;
        } else if (arrival >= 0) {
            held_response = arrival;
        }
        return popped;
    }
};

} // namespace mxcore
