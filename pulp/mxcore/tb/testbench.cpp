#include <vp/vp.hpp>
#include <vp/itf/io.hpp>
#include <vp/itf/wire.hpp>
#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

class MXCoreTestbench : public vp::Component {
public:
    explicit MXCoreTestbench(vp::ComponentConf &conf)
        : vp::Component(conf), timer(this, step) {
        input_file = get_js_config()->get_child_str("input_file");
        input_format = get_js_config()->get_child_str("input_format");
        output_format = get_js_config()->get_child_str("output_format");
        output_file = get_js_config()->get_child_str("output_file");
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
        b_base = ABase + M * K * input_bits / 8;
        scale_a_base = b_base + K * N * input_bits / 8;
        scale_b_base = scale_a_base + M * (K / input_block);
        result_base = scale_b_base + N * (K / input_block);
        scale_bytes = result_bits <= 8 ? M * N / 32 : 0;
        result_bytes = M * N * result_bits / 8;
        scale_base = result_base + result_bytes;
        memory.resize(scale_base + scale_bytes + Bandwidth);

        registers = {ABase, b_base, scale_a_base, scale_b_base, result_base, result_base, scale_base,
                     (N << 22) | (K << 10) | M,
                     input_control | (19u << 3) | output_control,
                     ((reuse_d_reg_size & 63u) << 23) | ((inner_blocks & 127u) << 16) |
                         ((inner_tiles & 127u) << 9) | ((b_col_tiles & 31u) << 4) | (a_row_tiles & 15u),
                     Reuse * VS * input_bits, NPE * VS * input_bits, Reuse * NPE * 32,
                     Reuse * NPE * result_bits, Reuse * (NPE / 32) * 8, Reuse * compute_chunks};

        memory_port.set_req_meth(access);
        completion.set_sync_meth(completed);
        new_master_port("mmio", &mmio);
        new_slave_port("memory", &memory_port);
        new_slave_port("completion", &completion);
    }

    void reset(bool active) override {
        if (active) {
            if (timer.is_enqueued()) timer.cancel();
            started = false;
            std::fill(memory.begin(), memory.end(), 0xa5);
            load_input();
        } else {
            timer.enqueue(10);
        }
    }

private:
    unsigned M, K, N, VS, NPE, Reuse;
    static constexpr unsigned Bandwidth = 64;
    static constexpr unsigned ABase = 0;
    static constexpr int64_t TimeoutCycles = 1000000;

    vp::IoMaster mmio;
    vp::IoSlave memory_port;
    vp::WireSlave<bool> completion;
    vp::ClockEvent timer;
    bool started = false;
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
        if (std::size_t(bytes) > memory.size()) memory.resize(bytes, 0xa5);
        input.seekg(0);
        input.read(reinterpret_cast<char *>(memory.data() + ABase), bytes);
        if (!input) throw std::runtime_error("Cannot read input binary: " + input_file);
    }

    uint32_t peripheral(unsigned address, bool write, uint32_t value = 0) {
        uint8_t data[4];
        for (unsigned i = 0; i < 4; ++i) data[i] = value >> (8 * i);
        vp::IoReq request(address, data, 4, write);
        request.set_initiator(0);
        if (mmio.req(&request) != vp::IO_REQ_OK)
            throw std::runtime_error("MXCore rejected a testbench register access");
        uint32_t readback = 0;
        for (unsigned i = 0; i < 4; ++i) readback |= uint32_t(data[i]) << (8 * i);
        return readback;
    }

    static void step(vp::Block *block, vp::ClockEvent *) {
        auto &self = *static_cast<MXCoreTestbench *>(block);
        try {
            if (self.started)
                throw std::runtime_error("MXCore testbench completion timeout");

            std::cout << "[TB] Input=" << self.input_file << ", format=" << self.input_format
                      << ", M=" << self.M << " K=" << self.K << " N=" << self.N
                      << ", VS=" << self.VS << " Reuse=" << self.Reuse << " NPE=" << self.NPE
                      << ", output=" << self.output_format << '\n';
            self.peripheral(0x14, true, 0);
            if (int32_t(self.peripheral(0x04, false)) < 0)
                throw std::runtime_error("MXCore did not grant a job context");
            for (unsigned index = 0; index < self.registers.size(); ++index)
                self.peripheral(0x20 + index * 4, true, self.registers[index]);
            std::cout << "[TB] Triggering MXCore\n";
            self.started = true;
            self.timer.enqueue(TimeoutCycles);
            self.peripheral(0x00, true, 0);
        } catch (const std::exception &error) {
            std::cerr << "[TB] " << error.what() << '\n';
            if (self.timer.is_enqueued()) self.timer.cancel();
            self.time.get_engine()->quit(1);
        }
    }

    static vp::IoReqStatus access(vp::Block *block, vp::IoReq *request) {
        auto &self = *static_cast<MXCoreTestbench *>(block);
        const auto address = request->get_addr(), size = request->get_size();
        if (!size || address > self.memory.size() || size > self.memory.size() - address)
            return vp::IO_REQ_INVALID;
        if (request->get_is_write())
            std::memcpy(self.memory.data() + address, request->get_data(), size);
        else
            std::memcpy(request->get_data(), self.memory.data() + address, size);
        request->inc_latency(1);
        return vp::IO_REQ_OK;
    }

    void save_output() const {
        std::ofstream output(output_file, std::ios::binary);
        if (!output) throw std::runtime_error("Cannot create testbench output file");
        output.write(reinterpret_cast<const char *>(memory.data() + result_base), result_bytes + scale_bytes);
        if (!output) throw std::runtime_error("Cannot write testbench output file");
        std::cout << "[TB] Saved " << result_bytes + scale_bytes << " bytes to " << output_file << '\n'
                  << "[TB] Execution finished.\n";
    }

    static void completed(vp::Block *block, bool value) {
        if (!value) return;
        auto &self = *static_cast<MXCoreTestbench *>(block);
        if (self.timer.is_enqueued()) self.timer.cancel();
        self.save_output();
        self.time.get_engine()->quit(0);
    }
};

extern "C" vp::Component *gv_new(vp::ComponentConf &conf) { return new MXCoreTestbench(conf); }
