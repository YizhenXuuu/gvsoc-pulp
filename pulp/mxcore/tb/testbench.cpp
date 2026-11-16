#include <vp/vp.hpp>
#include <vp/itf/io.hpp>
#include <vp/itf/wire.hpp>
#include "tb_tcdm_memory.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <deque>
#include <fstream>
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
          memory_event(this, memory_step) {
        input_file = get_js_config()->get_child_str("input_file");
        input_format = get_js_config()->get_child_str("input_format");
        output_format = get_js_config()->get_child_str("output_format");
        output_file = get_js_config()->get_child_str("output_file");
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
        const unsigned inner_blocks = K / input_block;
        const unsigned compute_chunks = K / elements_per_dotp;
        const unsigned reuse_d_reg_size = std::min(32u / VS, compute_chunks);
        result_bits = get_js_config()->get_child_int("result_bits");
        b_base = standalone ? ABase : ABase + M * K * input_bits / 8;
        scale_a_base = standalone ? ABase : b_base + K * N * input_bits / 8;
        scale_b_base = standalone ? ABase : scale_a_base + M * (K / input_block);
        result_base = standalone ? ABase : scale_b_base + N * (K / input_block);
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
                     ((reuse_d_reg_size & 63u) << 23) | ((inner_blocks & 127u) << 16) |
                         ((inner_tiles & 127u) << 9) | ((b_col_tiles & 31u) << 4) | (a_row_tiles & 15u),
                     Reuse * VS * input_bits, NPE * VS * input_bits,
                     Reuse * NPE * (standalone ? standalone_input_bits : 32),
                     Reuse * NPE * result_bits, Reuse * (NPE / 32) * 8, Reuse * compute_chunks};

        memory_port.set_req_meth(access);
        completion.set_sync_meth(completed);
        mmio.set_resp_meth(mmio_response);
        mmio.set_grant_meth(mmio_grant);
        new_master_port("mmio", &mmio);
        new_slave_port("memory", &memory_port);
        new_slave_port("completion", &completion);
    }

    void reset(bool active) override {
        reset_active = active;
        if (active) {
            if (timer.is_enqueued()) timer.cancel();
            if (watchdog.is_enqueued()) watchdog.cancel();
            phase = Phase::START;
            register_index = 0;
            completion_seen = false;
            // An asynchronous target still owns its request until response.
            discard_mmio = mmio_state == MmioState::PENDING;
            if (!discard_mmio) mmio_state = MmioState::IDLE;
            // Release pre-reset transactions asynchronously without modifying reloaded data.
            for (auto &transfer : memory_requests) transfer.cancelled = true;
            std::fill(memory.begin(), memory.end(), 0xa5);
            load_input();
        } else {
            timer.enqueue(10);
            watchdog.enqueue(10 + TimeoutCycles);
        }
    }

private:
    enum class Phase { START, SOFT_CLEAR, ACQUIRE, PROGRAM, TRIGGER, WAIT_COMPLETION, DONE };
    enum class MmioState { IDLE, PENDING, RETURNED };

    unsigned M, K, N, VS, NPE, Reuse;
    static constexpr unsigned Bandwidth = mxcore::tb::BeatBytes;
    static constexpr unsigned ABase = 0;
    static constexpr int64_t TimeoutCycles = 1000000;

    vp::IoMaster mmio;
    vp::IoSlave memory_port;
    vp::WireSlave<bool> completion;
    vp::ClockEvent timer;
    vp::ClockEvent watchdog;
    vp::ClockEvent memory_event;
    Phase phase = Phase::START;
    MmioState mmio_state = MmioState::IDLE;
    std::array<uint8_t, 4> mmio_data{};
    vp::IoReq mmio_request{0, nullptr, 4, false};
    int64_t mmio_issued_at = 0;
    unsigned register_index = 0;
    bool reset_active = true;
    bool discard_mmio = false;
    bool completion_seen = false;
    bool no_stalls = false;
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
    std::string output_file;

    void load_input() {
        std::ifstream input(input_file, std::ios::binary | std::ios::ate);
        if (!input) throw std::runtime_error("Cannot open input binary: " + input_file);
        const std::streamsize bytes = input.tellg();
        if (bytes < 0) throw std::runtime_error("Cannot determine input binary size: " + input_file);
        if (standalone && bytes < std::streamsize(result_base) + c_input_bytes)
            throw std::runtime_error("Standalone binary must contain the complete FP32/BF16 C matrix");
        if (!standalone && preload && bytes < std::streamsize(result_base) + c_input_bytes)
            throw std::runtime_error("Preload binary must contain A, B, ScaleA, ScaleB and FP32 C");
        if (std::size_t(bytes) > memory.size()) memory.resize(bytes, 0xa5);
        input.seekg(0);
        input.read(reinterpret_cast<char *>(memory.data() + ABase), bytes);
        if (!input) throw std::runtime_error("Cannot read input binary: " + input_file);
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
        }
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
            if (int32_t(readback) < 0)
                throw std::runtime_error("MXCore did not grant a job context");
            phase = Phase::PROGRAM;
            break;
        case Phase::PROGRAM:
            if (++register_index == registers.size()) phase = Phase::TRIGGER;
            break;
        case Phase::TRIGGER:
            phase = Phase::WAIT_COMPLETION;
            if (completion_seen) finish_success();
            return;
        default:
            throw std::logic_error("Unexpected completed MXCore register access");
        }

        // PERIPH_WRITE has a hold/termination edge; every next task has a setup edge.
        // With the registered one-cycle response, issue intervals are 3/2 cycles
        // after writes/reads, respectively, before adding any target stalls.
        timer.enqueue(write ? 2 : 1);
    }

    static void step(vp::Block *block, vp::ClockEvent *) {
        auto &self = *static_cast<MXCoreTestbench *>(block);
        try {
            if (self.reset_active || self.discard_mmio) return;
            if (self.mmio_state == MmioState::RETURNED) {
                self.finish_peripheral();
                return;
            }
            if (self.mmio_state == MmioState::PENDING) return;

            switch (self.phase) {
            case Phase::START:
                std::cout << "[TB] Input=" << self.input_file << ", format=" << self.input_format
                          << ", M=" << self.M << " K=" << self.K << " N=" << self.N
                          << ", VS=" << self.VS << " Reuse=" << self.Reuse << " NPE=" << self.NPE
                          << ", output=" << self.output_format << ", preload=" << self.preload
                          << ", standalone=" << self.standalone
                          << ", standalone_input_bits=" << self.standalone_input_bits << '\n';
                self.phase = Phase::SOFT_CLEAR;
                self.timer.enqueue(1);
                break;
            case Phase::SOFT_CLEAR:
                self.peripheral(0x14, true, 0);
                break;
            case Phase::ACQUIRE:
                self.peripheral(0x04, false);
                break;
            case Phase::PROGRAM:
                self.peripheral(0x20 + self.register_index * 4, true,
                                self.registers[self.register_index]);
                break;
            case Phase::TRIGGER:
                std::cout << "[TB] Triggering MXCore\n";
                if (self.watchdog.is_enqueued()) self.watchdog.cancel();
                self.watchdog.enqueue(TimeoutCycles);
                self.peripheral(0x00, true, 0);
                break;
            case Phase::WAIT_COMPLETION:
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
        std::cout << "[TB] Saved " << result_bytes + scale_bytes << " bytes to " << output_file << '\n'
                  << "[TB] Execution finished.\n";
    }

    static void completed(vp::Block *block, bool value) {
        if (!value) return;
        auto &self = *static_cast<MXCoreTestbench *>(block);
        if (self.reset_active || self.phase == Phase::DONE) return;
        try {
            if (self.phase != Phase::TRIGGER && self.phase != Phase::WAIT_COMPLETION)
                throw std::logic_error("MXCore completed before the testbench trigger");
            self.completion_seen = true;
            if (self.phase == Phase::WAIT_COMPLETION) self.finish_success();
        } catch (const std::exception &error) {
            self.fail(error.what());
        }
    }

    void finish_success() {
        if (timer.is_enqueued()) timer.cancel();
        if (watchdog.is_enqueued()) watchdog.cancel();
        phase = Phase::DONE;
        save_output();
        time.get_engine()->quit(0);
    }
};

extern "C" vp::Component *gv_new(vp::ComponentConf &conf) { return new MXCoreTestbench(conf); }
