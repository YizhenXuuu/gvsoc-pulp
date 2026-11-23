#include <vp/vp.hpp>
#include <vp/itf/io.hpp>
#include <vp/itf/wire.hpp>
#include <vp/register.hpp>
#include <vp/time/time_event.hpp>
#include "tb_tcdm_memory.hpp"
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

class MXCoreTestbench : public vp::Component {
public:
    explicit MXCoreTestbench(vp::ComponentConf &conf)
        : vp::Component(conf), timer(this, step), watchdog(this, timed_out),
          memory_event(this, memory_step), monitor_event(this, monitor), tail_event(this, tail_done),
          wave_clock_event(this, clock_wave), wave_reset(*this, "timing/reset", 1, true, 1),
          wave_clk(*this, "timing/clk", 1, true, 1) {
        input_file = get_js_config()->get_child_str("input_file");
        input_format = get_js_config()->get_child_str("input_format");
        output_format = get_js_config()->get_child_str("output_format");
        output_file = get_js_config()->get_child_str("output_file");
        golden_file = get_js_config()->get_child_str("golden_file");
        preload = get_js_config()->get_child_int("preload");
        standalone = get_js_config()->get_child_int("standalone");
        standalone_input_bits = get_js_config()->get_child_int("standalone_input_bits");
        no_stalls = get_js_config()->get_child_int("no_stalls");
        stall_probability = get_js_config()->get("stall_probability")->get_double();
        const int seed = get_js_config()->get_child_int("stall_seed");
        if (!std::isfinite(stall_probability) || stall_probability < 0 || stall_probability > 1 || seed < 0)
            throw std::invalid_argument("Invalid MXCore memory stall probability/seed");
        stall_seed = unsigned(seed);
        load_grants(get_js_config()->get_child_str("grant_file"));
        M = get_js_config()->get_child_int("m");
        K = get_js_config()->get_child_int("k");
        N = get_js_config()->get_child_int("n");
        VS = get_js_config()->get_child_int("vector_size");
        Reuse = get_js_config()->get_child_int("reuse");
        NPE = get_js_config()->get_child_int("npe");
        if (M > 1023 || K > 4095 || N > 1023)
            throw std::invalid_argument("M/K/N cannot be encoded in the 10/12/10-bit register fields");
        const unsigned input_bits = get_js_config()->get_child_int("input_bits");
        const unsigned input_block = get_js_config()->get_child_int("input_block_size");
        const uint32_t input_control = get_js_config()->get_child_int("input_control");
        const uint32_t output_control = get_js_config()->get_child_int("output_control");
        const unsigned elements_per_dotp = VS * 8 / input_bits;
        const unsigned a_row_tiles = M / Reuse;
        const unsigned b_col_tiles = N / NPE;
        const unsigned inner_tiles = K / VS;
        const unsigned inner_blocks = (K + input_block - 1) / input_block;
        const unsigned compute_chunks = K / elements_per_dotp;
        const unsigned sa_vec_per_block = std::min(32u / VS, compute_chunks);
        result_bits = get_js_config()->get_child_int("result_bits");
        b_base = standalone ? ABase : ABase + M * K * input_bits / 8;
        scale_a_base = standalone ? ABase : b_base + K * N * input_bits / 8;
        scale_b_base = standalone ? ABase : scale_a_base + M * inner_blocks;
        result_base = standalone ? ABase : scale_b_base + N * inner_blocks;
        scale_bytes = result_bits <= 8 ? M * N / 32 : 0;
        result_bytes = M * N * result_bits / 8;
        c_input_bytes = standalone ? M * N * standalone_input_bits / 8 : preload ? M * N * 4 : 0;
        scale_base = result_base + ((preload || standalone) ? c_input_bytes : result_bytes);
        memory.resize(std::max(scale_base + scale_bytes, result_base + result_bytes) + Bandwidth);

        registers = {ABase, b_base, scale_a_base, scale_b_base, result_base, result_base, scale_base,
                     (N << 22) | (K << 10) | M,
                     input_control | (19u << 3) | output_control | (uint32_t(preload) << 26) |
                         (uint32_t(standalone) << 27) |
                         (uint32_t(standalone && standalone_input_bits == 16) << 28),
                     ((sa_vec_per_block & 63u) << 23) | ((inner_blocks & 127u) << 16) |
                         ((inner_tiles & 127u) << 9) | ((b_col_tiles & 31u) << 4) | (a_row_tiles & 15u),
                     Reuse * VS * input_bits, NPE * VS * input_bits,
                     Reuse * NPE * (standalone ? standalone_input_bits : 32),
                     Reuse * NPE * result_bits, Reuse * (NPE / 32) * 8, Reuse * compute_chunks};

        memory_port.set_req_meth(access);
        completion.set_sync_meth(completed);
        compute_completion.set_sync_meth(compute_completed);
        mmio.set_resp_meth(mmio_response);
        mmio.set_grant_meth(mmio_grant);
        new_master_port("mmio", &mmio);
        new_master_port("dut_reset", &dut_reset);
        new_slave_port("memory", &memory_port);
        new_slave_port("compute_completion", &compute_completion);
        new_slave_port("completion", &completion);
    }

    void reset(bool active) override {
        reset_active = active;
        if (active) {
            if (timer.is_enqueued()) timer.cancel();
            if (watchdog.is_enqueued()) watchdog.cancel();
            if (monitor_event.is_enqueued()) monitor_event.cancel();
            if (tail_event.is_enqueued()) tail_event.cancel();
            if (wave_clock_event.is_enqueued()) wave_clock_event.cancel();
            wave_clk.set(1);
            phase = Phase::RESET;
            register_index = 0;
            acquire_reads = 0;
            completion_seen = false;
            compute_seen = false;
            compute_level = completion_level = false;
            reset_release_cycle = trigger_issue_cycle = trigger_accept_cycle = trigger_return_cycle = -1;
            compute_raised_cycle = completion_raised_cycle = compute_sample_cycle = completion_sample_cycle = -1;
            // An asynchronous target still owns its request until response.
            discard_mmio = mmio_state == MmioState::PENDING;
            if (!discard_mmio) mmio_state = MmioState::IDLE;
            // Release pre-reset transactions asynchronously without modifying reloaded data.
            for (auto &transfer : memory_requests) transfer.cancelled = true;
            std::fill(memory.begin(), memory.end(), 0xa5);
            if (dut_reset.is_bound()) dut_reset.sync(true);
            wave_reset.set(1);
        } else {
            // Display the fixed TB clock with rising edges on GVSOC cycle boundaries.
            // This observer never drives the DUT and runs only when its VCD is selected.
            if (wave_clk.reg_event.get_event_active())
                wave_clock_event.enqueue(clock.get_period() / 2);
            // rst_gen counts ten edges, then its reset FF releases on the next edge.
            timer.enqueue(ResetCycles + 1);
            watchdog.enqueue(ResetCycles + 1 + TimeoutCycles);
        }
    }

private:
    enum class Phase {
        RESET, START, SOFT_CLEAR, ACQUIRE, PROGRAM, TRIGGER, TRIGGER_RETURN,
        WAIT_COMPLETION, FINISH_CLEAR, FINISH_CLEAR_RETURN, TAIL, DONE
    };
    enum class MmioState { IDLE, PENDING, RETURNED };

    unsigned M, K, N, VS, NPE, Reuse;
    static constexpr unsigned Bandwidth = mxcore::tb::BeatBytes;
    static constexpr unsigned ABase = 0;
    static constexpr int64_t TimeoutCycles = 1000000;
    static constexpr int64_t ResetCycles = 10;
    static constexpr int64_t TailTimePs = 10000; // RTL #(10ns), independent of clock frequency.

    vp::IoMaster mmio;
    vp::WireMaster<bool> dut_reset;
    vp::IoSlave memory_port;
    vp::WireSlave<bool> completion;
    vp::WireSlave<bool> compute_completion;
    vp::ClockEvent timer;
    vp::ClockEvent watchdog;
    vp::ClockEvent memory_event;
    vp::ClockEvent monitor_event;
    vp::TimeEvent tail_event;
    vp::TimeEvent wave_clock_event;
    vp::Register<uint32_t> wave_reset, wave_clk;
    Phase phase = Phase::RESET;
    MmioState mmio_state = MmioState::IDLE;
    std::array<uint8_t, 4> mmio_data{};
    vp::IoReq mmio_request{0, nullptr, 4, false};
    int64_t mmio_issued_at = 0;
    unsigned register_index = 0;
    unsigned acquire_reads = 0;
    bool reset_active = true;
    bool discard_mmio = false;
    bool completion_seen = false;
    bool compute_seen = false;
    bool compute_level = false;
    bool completion_level = false;
    int64_t reset_release_cycle = -1;
    int64_t trigger_issue_cycle = -1;
    int64_t trigger_accept_cycle = -1;
    int64_t trigger_return_cycle = -1;
    int64_t compute_raised_cycle = -1;
    int64_t completion_raised_cycle = -1;
    int64_t compute_sample_cycle = -1;
    int64_t completion_sample_cycle = -1;
    bool no_stalls = true;
    double stall_probability = 0.1;
    unsigned stall_seed = 1;
    bool replay_grants = false;
    std::map<int64_t, uint32_t> grant_schedule;
    int64_t last_memory_attempt = -1;

    struct MemoryRequest {
        vp::IoReq *request;
        mxcore::tb::MemoryBeat beat;
        bool cancelled = false;
    };
    struct MemoryResponse {
        MemoryRequest transfer;
        int64_t ready_at;
    };
    std::deque<MemoryRequest> memory_requests;
    std::deque<MemoryResponse> memory_responses;
    bool preload = false;
    bool standalone = false;
    unsigned standalone_input_bits = 32;
    unsigned c_input_bytes = 0;
    std::array<uint32_t, 16> registers{};
    std::vector<uint8_t> memory;
    unsigned b_base, scale_a_base, scale_b_base, result_base;
    unsigned result_bits, result_bytes, scale_base, scale_bytes;
    std::string input_file, input_format, output_format;
    std::string output_file, golden_file;
    std::vector<uint8_t> golden_output;

    bool next_memory_token(std::istream &input, std::string &token,
                           unsigned &line, unsigned &token_line, const std::string &path) const {
        token.clear();
        for (;;) {
            const int ch = input.peek();
            if (ch == std::char_traits<char>::eof()) return !token.empty();
            if (std::isspace(static_cast<unsigned char>(ch))) {
                input.get();
                if (ch == '\n') ++line;
                if (!token.empty()) return true;
                continue;
            }
            if (ch == '/') {
                input.get();
                const int kind = input.get();
                if (kind == '/') {
                    int current;
                    while ((current = input.get()) != std::char_traits<char>::eof()) {
                        if (current == '\n') {
                            ++line;
                            break;
                        }
                    }
                } else if (kind == '*') {
                    bool closed = false;
                    int current;
                    while ((current = input.get()) != std::char_traits<char>::eof()) {
                        if (current == '\n') ++line;
                        if (current == '*' && input.peek() == '/') {
                            input.get();
                            closed = true;
                            break;
                        }
                    }
                    if (!closed)
                        throw std::runtime_error("Unterminated comment in hex file " + path);
                } else {
                    throw std::runtime_error("Invalid hex file comment in " + path + " at line " +
                                             std::to_string(line));
                }
                if (!token.empty()) return true;
                continue;
            }
            if (token.empty()) token_line = line;
            token.push_back(char(input.get()));
        }
    }

    uint32_t memory_hex_word(const std::string &token, unsigned line, const std::string &path) const {
        uint32_t value = 0;
        bool has_digit = false;
        for (unsigned char ch : token) {
            if (ch == '_') continue;
            unsigned digit;
            if (ch >= '0' && ch <= '9') digit = ch - '0';
            else if (ch >= 'a' && ch <= 'f') digit = ch - 'a' + 10;
            else if (ch >= 'A' && ch <= 'F') digit = ch - 'A' + 10;
            else throw std::runtime_error("Invalid hex token '" + token + "' in " + path + " at line " +
                                          std::to_string(line) + "; only known 32-bit words are supported");
            if (value > 0x0fffffffu)
                throw std::runtime_error("Hex word exceeds 32 bits in " + path + " at line " +
                                         std::to_string(line));
            value = (value << 4) | digit;
            has_digit = true;
        }
        if (!has_digit)
            throw std::runtime_error("Missing hex value in " + path + " at line " + std::to_string(line));
        return value;
    }

    void load_input() {
        std::ifstream input(input_file);
        if (!input) throw std::runtime_error("Cannot open hex memory file: " + input_file);
        const uint64_t required_bytes = uint64_t(result_base) +
                                        ((preload || standalone) ? c_input_bytes : 0);
        std::vector<bool> loaded((required_bytes + 3) / 4, false);
        uint64_t word_address = ABase / 4;
        unsigned line = 1, token_line = 1;
        std::string token;
        while (next_memory_token(input, token, line, token_line, input_file)) {
            if (token.front() == '@') {
                // $readmemh addresses index memory[] words, not individual bytes.
                word_address = memory_hex_word(token.substr(1), token_line, input_file);
                continue;
            }
            const uint32_t value = memory_hex_word(token, token_line, input_file);
            const uint64_t address = word_address * 4;
            if (address > memory.size() || 4 > memory.size() - address)
                throw std::runtime_error("Hex memory address outside configured TB memory at line " +
                                         std::to_string(token_line));
            for (unsigned byte = 0; byte < 4; ++byte)
                memory[address + byte] = uint8_t(value >> (8 * byte));
            if (word_address < loaded.size()) loaded[word_address] = true;
            ++word_address;
        }
        if (input.bad()) throw std::runtime_error("Cannot read hex memory file: " + input_file);
        const auto missing = std::find(loaded.begin(), loaded.end(), false);
        if (missing != loaded.end()) {
            std::ostringstream address;
            address << std::hex << std::distance(loaded.begin(), missing);
            throw std::runtime_error("Hex memory file is missing required input word @" + address.str());
        }
    }

    void load_golden() {
        std::ifstream input(golden_file);
        if (!input) throw std::runtime_error("Cannot open golden result file: " + golden_file);
        const size_t expected_bytes = size_t(result_bytes) + scale_bytes;
        const size_t expected_words = (expected_bytes + 3) / 4;
        golden_output.clear();
        golden_output.reserve(expected_bytes);
        size_t words = 0;
        unsigned line = 1, token_line = 1;
        std::string token;
        while (next_memory_token(input, token, line, token_line, golden_file)) {
            if (words == expected_words)
                throw std::runtime_error("Golden result file has extra data: " + golden_file +
                                         "; expected " + std::to_string(expected_words) + " 32-bit words");
            const uint32_t value = memory_hex_word(token, token_line, golden_file);
            // PyGolden packs little-endian 32-bit words as [Z][SZ]. Ignore final word padding.
            for (unsigned byte = 0; byte < 4 && golden_output.size() < expected_bytes; ++byte)
                golden_output.push_back(uint8_t(value >> (8 * byte)));
            ++words;
        }
        if (input.bad()) throw std::runtime_error("Cannot read golden result file: " + golden_file);
        if (words != expected_words)
            throw std::runtime_error("Golden result file size mismatch: " + golden_file +
                                     "; expected " + std::to_string(expected_words) +
                                     " 32-bit words, got " + std::to_string(words));
    }

    size_t compare_region(const char *name, unsigned base, size_t golden_offset,
                          size_t count, unsigned bits, unsigned &reported,
                          std::string &first_mismatch) const {
        size_t matched = 0;
        for (size_t i = 0; i < count; ++i) {
            const size_t offset = i * bits / 8;
            uint32_t actual = 0, expected = 0;
            if (bits == 4) {
                const unsigned shift = unsigned(i % 2) * 4;
                actual = (memory[base + offset] >> shift) & 15u;
                expected = (golden_output[golden_offset + offset] >> shift) & 15u;
            } else {
                for (unsigned byte = 0; byte < bits / 8; ++byte) {
                    actual |= uint32_t(memory[base + offset + byte]) << (8 * byte);
                    expected |= uint32_t(golden_output[golden_offset + offset + byte]) << (8 * byte);
                }
            }
            if (actual == expected) {
                ++matched;
            } else if (reported < 8) {
                std::ostringstream detail;
                detail << name << '[' << i << "] @ 0x" << std::hex
                       << base + offset << ": expected=0x" << std::setfill('0') << std::setw(bits / 4)
                       << expected << ", actual=0x" << std::setw(bits / 4) << actual;
                if (first_mismatch.empty()) first_mismatch = detail.str();
                std::cout << "[TB] Mismatch " << detail.str() << '\n';
                ++reported;
            }
        }
        return matched;
    }

    void print_accuracy(const char *name, size_t matched, size_t total) const {
        std::ostringstream summary;
        summary << "[TB] " << name << ": ";
        if (total)
            summary << std::fixed << std::setprecision(6) << 100.0 * double(matched) / double(total);
        else
            summary << "N/A";
        summary << " (" << matched << '/' << total << ')';
        std::cout << summary.str() << '\n';
    }

    bool check_output() const {
        const size_t values = size_t(M) * N;
        unsigned reported = 0;
        std::string first_mismatch;
        const size_t z_matched = compare_region("Z", result_base, 0, values, result_bits, reported, first_mismatch);
        const size_t sz_matched = compare_region("SZ", scale_base, result_bytes, scale_bytes, 8, reported, first_mismatch);
        print_accuracy("Golden accuracy — Z (%)", z_matched, values);
        print_accuracy("Golden accuracy — SZ (%)", sz_matched, scale_bytes);
        const bool passed = z_matched == values && sz_matched == scale_bytes;
        std::cout << "[TB] Golden check (PASS/FAIL): " << (passed ? "PASS" : "FAIL") << '\n'
                  << "[TB] Notes / first mismatch: " << (passed ? "none" : first_mismatch) << '\n';
        return passed;
    }

    void load_grants(const std::string &path) {
        replay_grants = !path.empty();
        if (!replay_grants) return;
        std::ifstream input(path);
        if (!input) throw std::runtime_error("Cannot open MXCore grant file: " + path);
        std::string line;
        unsigned line_number = 0;
        while (std::getline(input, line)) {
            ++line_number;
            const auto comment = line.find('#');
            if (comment != std::string::npos) line.resize(comment);
            std::istringstream fields(line);
            fields >> std::ws;
            if (fields.eof()) continue;
            int64_t cycle;
            std::string mask, extra;
            if (!(fields >> cycle >> mask) || cycle < 0 || (fields >> extra))
                throw std::runtime_error("Invalid MXCore grant file line " + std::to_string(line_number));
            std::size_t end;
            const unsigned long grants = std::stoul(mask, &end, 16);
            if (end != mask.size() || grants > mxcore::tb::AllBankGrants ||
                !grant_schedule.emplace(cycle, uint32_t(grants)).second)
                throw std::runtime_error("Invalid/duplicate MXCore grant mask on line " +
                                         std::to_string(line_number));
        }
        if (!input.eof()) throw std::runtime_error("Cannot read MXCore grant file: " + path);
    }

    uint32_t bank_grants(int64_t cycle) const {
        if (no_stalls) return mxcore::tb::AllBankGrants;
        if (replay_grants) {
            const auto entry = grant_schedule.find(cycle);
            return entry == grant_schedule.end() ? mxcore::tb::AllBankGrants : entry->second;
        }
        return mxcore::tb::sample_bank_grants(uint64_t(cycle), stall_probability, stall_seed);
    }

    void peripheral(unsigned address, bool write, uint32_t value = 0) {
        if (mmio_state != MmioState::IDLE)
            throw std::logic_error("MXCore testbench already has an outstanding register access");
        for (unsigned i = 0; i < mmio_data.size(); ++i) mmio_data[i] = value >> (8 * i);
        mmio_request.init();
        mmio_request.set_addr(address);
        mmio_request.set_data(mmio_data.data());
        mmio_request.set_size(mmio_data.size());
        mmio_request.set_is_write(write);
        mmio_request.set_initiator(0);
        mmio_request.status = vp::IO_REQ_OK;
        mmio_issued_at = clock.get_cycles();
        mmio_state = MmioState::PENDING;

        const auto status = mmio.req(&mmio_request);
        if (phase == Phase::TRIGGER &&
            (status == vp::IO_REQ_OK || status == vp::IO_REQ_PENDING)) {
            trigger_accept_cycle = mmio_issued_at;
        }
        if (status == vp::IO_REQ_OK) {
            handle_mmio_response(&mmio_request);
        } else if (status != vp::IO_REQ_PENDING && status != vp::IO_REQ_DENIED) {
            throw std::runtime_error("MXCore rejected a testbench register access");
        }
    }

    void handle_mmio_response(vp::IoReq *request) {
        if (request != &mmio_request || mmio_state != MmioState::PENDING)
            throw std::logic_error("Unexpected MXCore testbench register response");
        if (discard_mmio) {
            discard_mmio = false;
            mmio_state = MmioState::IDLE;
            if (!reset_active && !timer.is_enqueued()) timer.enqueue(1);
            return;
        }
        if (request->status != vp::IO_REQ_OK)
            throw std::runtime_error("MXCore testbench register access failed");

        mmio_state = MmioState::RETURNED;
        const int64_t now = clock.get_cycles();
        // Async callbacks may already account for some or all of this latency.
        const int64_t ready_at = std::max(now, mmio_issued_at +
            int64_t(std::max(uint64_t(1), request->get_full_latency())));
        timer.enqueue(ready_at - now);
    }

    static void mmio_response(vp::Block *block, vp::IoReq *request) {
        auto &self = *static_cast<MXCoreTestbench *>(block);
        try {
            self.handle_mmio_response(request);
        } catch (const std::exception &error) {
            self.fail(error.what());
        }
    }

    static void mmio_grant(vp::Block *block, vp::IoReq *request) {
        auto &self = *static_cast<MXCoreTestbench *>(block);
        if (request != &self.mmio_request || self.mmio_state != MmioState::PENDING) {
            self.fail("Unexpected MXCore testbench register grant");
            return;
        }
        if (!self.discard_mmio && self.phase == Phase::TRIGGER)
            self.trigger_accept_cycle = self.clock.get_cycles();
        // The target retains the request; wait for response without resubmitting.
    }

    void finish_peripheral() {
        uint32_t readback = 0;
        for (unsigned i = 0; i < mmio_data.size(); ++i)
            readback |= uint32_t(mmio_data[i]) << (8 * i);
        const bool write = mmio_request.get_is_write();
        mmio_state = MmioState::IDLE;

        switch (phase) {
        case Phase::SOFT_CLEAR:
            phase = Phase::ACQUIRE;
            break;
        case Phase::ACQUIRE:
            // The RTL polls while status[31] is set; the watchdog bounds retries.
            if (!(readback & (1u << 31))) {
                phase = Phase::PROGRAM;
            }
            break;
        case Phase::PROGRAM:
            if (++register_index == registers.size()) phase = Phase::TRIGGER;
            break;
        case Phase::TRIGGER:
            if (trigger_accept_cycle < 0)
                throw std::logic_error("MXCore trigger response arrived without acceptance");
            phase = Phase::TRIGGER_RETURN;
            timer.enqueue(1); // PERIPH_WRITE's final termination edge.
            return;
        case Phase::FINISH_CLEAR:
            phase = Phase::FINISH_CLEAR_RETURN;
            timer.enqueue(1);
            return;
        default:
            throw std::logic_error("Unexpected completed MXCore register access");
        }

        // PERIPH_WRITE has a hold/termination edge; every next task has a setup edge.
        // With the registered one-cycle response, issue intervals are 3/2 cycles
        // after writes/reads, respectively, before adding any target stalls.
        timer.enqueue(write ? 2 : 1);
    }

    static void clock_wave(vp::Block *block, vp::TimeEvent *) {
        auto &self = *static_cast<MXCoreTestbench *>(block);
        self.wave_clk.set(self.wave_clk.get() ^ 1u);
        self.wave_clock_event.enqueue(self.clock.get_period() / 2);
    }

    static void step(vp::Block *block, vp::ClockEvent *) {
        auto &self = *static_cast<MXCoreTestbench *>(block);
        try {
            if (self.reset_active) return;
            if (self.phase == Phase::RESET) {
                if (!self.dut_reset.is_bound())
                    throw std::logic_error("MXCore testbench DUT reset is not connected");
                self.reset_release_cycle = self.clock.get_cycles();
                self.dut_reset.sync(false);
                self.wave_reset.set(0);
                self.load_input();
                self.load_golden();
                self.phase = Phase::START;
                self.timer.enqueue(0);
                return;
            }
            if (self.discard_mmio) return;
            if (self.mmio_state == MmioState::RETURNED) {
                self.finish_peripheral();
                return;
            }
            if (self.mmio_state == MmioState::PENDING) return;

            switch (self.phase) {
            case Phase::START:
                std::cout << "[TB] CONFIG M=" << self.M << " K=" << self.K << " N=" << self.N
                          << " VS=" << self.VS << " NPE=" << self.NPE << " REUSE=" << self.Reuse
                          << " input=" << self.input_format << " output=" << self.output_format
                          << " preload=" << self.preload << " standalone=" << self.standalone;
                if (self.standalone) {
                    std::cout << " standalone_input=" << (self.standalone_input_bits == 16 ? "bf16" : "fp32");
                }
                std::cout << " no_stalls=" << self.no_stalls << '\n';
                self.phase = Phase::SOFT_CLEAR;
                // After wait(rst_n), PERIPH_WRITE waits for its setup edge,
                // then the slave samples req at the following rising edge.
                self.timer.enqueue(2);
                break;
            case Phase::SOFT_CLEAR:
                self.peripheral(0x14, true, 0);
                break;
            case Phase::ACQUIRE:
                ++self.acquire_reads;
                self.peripheral(0x04, false);
                break;
            case Phase::PROGRAM:
                self.peripheral(0x20 + self.register_index * 4, true,
                                self.registers[self.register_index]);
                break;
            case Phase::TRIGGER:
                self.trigger_issue_cycle = self.clock.get_cycles();
                if (self.watchdog.is_enqueued()) self.watchdog.cancel();
                self.watchdog.enqueue(TimeoutCycles);
                self.monitor_event.enqueue(1);
                self.peripheral(0x00, true, 0);
                break;
            case Phase::TRIGGER_RETURN:
                self.trigger_return_cycle = self.clock.get_cycles();
                self.phase = Phase::WAIT_COMPLETION;
                if (self.completion_seen) self.begin_finish();
                break;
            case Phase::FINISH_CLEAR:
                self.peripheral(0x14, true, 0);
                break;
            case Phase::FINISH_CLEAR_RETURN:
                self.phase = Phase::TAIL;
                self.tail_event.enqueue(TailTimePs);
                break;
            case Phase::RESET:
            case Phase::WAIT_COMPLETION:
            case Phase::TAIL:
            case Phase::DONE:
                break;
            }
        } catch (const std::exception &error) {
            self.fail(error.what());
        }
    }

    void fail(const std::string &message) {
        std::cerr << "[TB] " << message << '\n';
        if (timer.is_enqueued()) timer.cancel();
        if (watchdog.is_enqueued()) watchdog.cancel();
        if (monitor_event.is_enqueued()) monitor_event.cancel();
        if (tail_event.is_enqueued()) tail_event.cancel();
        if (wave_clock_event.is_enqueued()) wave_clock_event.cancel();
        phase = Phase::DONE;
        time.get_engine()->quit(1);
    }

    static void timed_out(vp::Block *block, vp::ClockEvent *) {
        auto &self = *static_cast<MXCoreTestbench *>(block);
        self.fail("MXCore testbench register access or completion timeout");
    }

    static vp::IoReqStatus access(vp::Block *block, vp::IoReq *request) {
        auto &self = *static_cast<MXCoreTestbench *>(block);
        mxcore::tb::MemoryBeat beat;
        beat.address = request->get_addr();
        const uint64_t size = request->get_size();
        if (!size || size > Bandwidth || !request->get_data()) return vp::IO_REQ_INVALID;
        beat.size = unsigned(size);
        beat.write = request->get_is_write();
        if (!beat.valid(self.memory.size()))
            return vp::IO_REQ_INVALID;
        if (beat.write) std::copy_n(request->get_data(), beat.size, beat.data.begin());
        self.memory_requests.push_back({request, std::move(beat)});
        if (!self.memory_event.is_enqueued()) self.memory_event.enqueue(0);
        // Bank acceptance happens in memory_step, after the issuing callback.
        // Retain the original request and later deliver grant, then response.
        return vp::IO_REQ_DENIED;
    }

    static void memory_step(vp::Block *block, vp::ClockEvent *) {
        auto &self = *static_cast<MXCoreTestbench *>(block);
        const int64_t now = self.clock.get_cycles();

        while (!self.memory_responses.empty() && self.memory_responses.front().ready_at <= now) {
            auto response = std::move(self.memory_responses.front());
            self.memory_responses.pop_front();
            auto &transfer = response.transfer;
            if (!transfer.beat.write)
                std::copy_n(transfer.beat.data.begin(), transfer.beat.size, transfer.request->get_data());
            transfer.request->status = vp::IO_REQ_OK;
            transfer.request->get_resp_port()->resp(transfer.request);
        }

        if (!self.memory_requests.empty() && self.last_memory_attempt != now) {
            self.last_memory_attempt = now;
            auto &transfer = self.memory_requests.front();
            const bool granted = transfer.cancelled || transfer.beat.accept(self.memory, self.bank_grants(now));
            if (granted) {
                vp::IoReq *request = transfer.request;
                self.memory_responses.push_back({std::move(transfer), now + 1});
                self.memory_requests.pop_front();
                // Grants are ANDed within one cycle, never accumulated across cycles.
                request->get_resp_port()->grant(request);
            }
        }
        if (!self.memory_requests.empty() || !self.memory_responses.empty()) self.memory_event.enqueue(1);
    }

    void save_output() const {
        std::ofstream output(output_file, std::ios::binary);
        if (!output) throw std::runtime_error("Cannot create testbench output file");
        output.write(reinterpret_cast<const char *>(memory.data() + result_base), result_bytes);
        output.write(reinterpret_cast<const char *>(memory.data() + scale_base), scale_bytes);
        if (!output) throw std::runtime_error("Cannot write testbench output file");
        std::cout << "[TB] OUTPUT file=" << output_file << '\n';
    }

    static void compute_completed(vp::Block *block, bool value) {
        auto &self = *static_cast<MXCoreTestbench *>(block);
        self.compute_level = value;
        if (!value || self.reset_active || self.phase == Phase::RESET || self.phase == Phase::DONE) return;
        if (self.trigger_issue_cycle < 0) {
            self.fail("MXCore compute event arrived before the testbench trigger");
            return;
        }
        if (self.compute_raised_cycle < 0) self.compute_raised_cycle = self.clock.get_cycles();
    }

    static void completed(vp::Block *block, bool value) {
        auto &self = *static_cast<MXCoreTestbench *>(block);
        self.completion_level = value;
        if (!value || self.reset_active || self.phase == Phase::RESET || self.phase == Phase::DONE) return;
        if (self.trigger_issue_cycle < 0) {
            self.fail("MXCore output event arrived before the testbench trigger");
            return;
        }
        if (self.completion_raised_cycle < 0) self.completion_raised_cycle = self.clock.get_cycles();
    }

    static void monitor(vp::Block *block, vp::ClockEvent *) {
        auto &self = *static_cast<MXCoreTestbench *>(block);
        if (self.reset_active || self.phase == Phase::DONE) return;
        try {
            const int64_t now = self.clock.get_cycles();
            // Sample each event on the next edge, before that edge's DUT updates.
            // Assertion timestamps keep callback ordering from losing a pulse.
            if (!self.compute_seen && self.compute_raised_cycle >= 0 && now > self.compute_raised_cycle) {
                self.compute_seen = true;
                self.compute_sample_cycle = now;
            }
            if (!self.completion_seen && self.completion_raised_cycle >= 0 && now > self.completion_raised_cycle) {
                self.completion_seen = true;
                self.completion_sample_cycle = now;
                if (self.phase == Phase::WAIT_COMPLETION) self.begin_finish();
            }
            if (!self.completion_seen) self.monitor_event.enqueue(1);
        } catch (const std::exception &error) {
            self.fail(error.what());
        }
    }

    void begin_finish() {
        if (!compute_seen || trigger_return_cycle < 0)
            throw std::logic_error("MXCore output completed without compute/trigger completion");
        if (watchdog.is_enqueued()) watchdog.cancel();
        watchdog.enqueue(TimeoutCycles);
        phase = Phase::FINISH_CLEAR;
        timer.enqueue(1); // Setup edge for the RTL TB's final PERIPH_WRITE.
    }

    static void tail_done(vp::Block *block, vp::TimeEvent *) {
        auto &self = *static_cast<MXCoreTestbench *>(block);
        if (self.reset_active || self.phase != Phase::TAIL) return;
        try {
            if (self.compute_level || self.completion_level)
                throw std::logic_error("MXCore completion events remain asserted after final soft clear");
            self.finish_run();
        } catch (const std::exception &error) {
            self.fail(error.what());
        }
    }

    void finish_run() {
        if (timer.is_enqueued()) timer.cancel();
        if (watchdog.is_enqueued()) watchdog.cancel();
        if (monitor_event.is_enqueued()) monitor_event.cancel();
        if (tail_event.is_enqueued()) tail_event.cancel();
        if (wave_clock_event.is_enqueued()) wave_clock_event.cancel();
        phase = Phase::DONE;
        save_output();
        time.get_engine()->quit(check_output() ? 0 : 1);
    }
};

extern "C" vp::Component *gv_new(vp::ComponentConf &conf) { return new MXCoreTestbench(conf); }
