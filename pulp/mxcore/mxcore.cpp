// SPDX-License-Identifier: Apache-2.0
// MXCore HWPE. Register input, clocked FSM and completion follow LightRedmule's
// component structure; register semantics and the data path follow MXCore RTL.
#include "hci_fifo.hpp"
#include "mxdotp.hpp"
#include "quantizer.hpp"
#include <algorithm>
#include <array>
#include <climits>
#include <cstring>
#include <deque>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <vp/itf/io.hpp>
#include <vp/itf/wire.hpp>
#include <vp/register.hpp>
#include <vp/vp.hpp>

/************************************************************
 *                      Type Definition                       *
 ************************************************************/

typedef uint8_t fp8e4m3;
typedef uint8_t fp8e5m2;
using Bytes = std::vector<uint8_t>;
using FifoLanes = std::vector<std::deque<Bytes>>;
using mxcore::FpStatus;
using mxcore::InputFormat;
using mxcore::OutputFormat;
using mxcore::MXDotp;

enum mxcore_state : uint8_t {
    IDLE,
    PRELOAD,
    ROUTINE,
    STORING
};

enum mxcore_stream : unsigned {
    STREAM_A,
    STREAM_B,
    STREAM_SA,
    STREAM_SB,
    STREAM_Z,
    STREAM_SZ,
    STREAM_C,
    NB_STREAMS
};

enum mxcore_req_state {
    REQ_FREE,
    REQ_QUEUED,
    REQ_PENDING,
    REQ_RETURNED,
    REQ_BUFFERED
};

static constexpr unsigned NB_INPUT_STREAMS = 4;
static constexpr unsigned NB_OUTPUT_STREAMS = 2;
static constexpr unsigned NB_TCDM_CHANNELS = 6;
static constexpr unsigned STREAM_FIFO_DEPTH = 2;
static_assert(STREAM_FIFO_DEPTH == mxcore::HciFifo::Depth, "RTL HCI FIFO depth");
static constexpr unsigned NB_IO_REGS = 16;
static constexpr unsigned MAX_CORES = 16;
static constexpr unsigned NB_EVENTS = 2;

/************************************************************
 *                    Function Declaration                    *
 ************************************************************/

uint8_t matmul_fp8e4m3(uint32_t *z, const uint32_t *d, const fp8e4m3 *a, const fp8e4m3 *b,
                       const uint8_t *scale_a, const uint8_t *scale_b, unsigned m_size, unsigned k_size,
                       unsigned vs, unsigned npe, unsigned reuse, const MXDotp &dotp);

uint8_t matmul_fp8e5m2(uint32_t *z, const uint32_t *d, const fp8e5m2 *a, const fp8e5m2 *b,
                       const uint8_t *scale_a, const uint8_t *scale_b, unsigned m_size, unsigned k_size,
                       unsigned vs, unsigned npe, unsigned reuse, const MXDotp &dotp);

uint8_t matmul_fp4(uint32_t *z, const uint32_t *d, const uint8_t *a, const uint8_t *b,
                   const uint8_t *scale_a, const uint8_t *scale_b,
                   unsigned vs, unsigned npe, const MXDotp &dotp);

/************************************************************
 *                      Class Definition                      *
 ************************************************************/

class MXCore : public vp::Component {
  public:
    MXCore(vp::ComponentConf &config);
    ~MXCore() override;
    void reset(bool active) override;

  private:
    // Peripheral input, clocked state machine and memory callbacks.
    static vp::IoReqStatus req(vp::Block *__this, vp::IoReq *req);
    static void            fsm_handler(vp::Block *__this, vp::ClockEvent *event);
    static void            fsm_update(vp::Block *__this, vp::ClockEvent *event);
    static void            tcdm_response(vp::Block *__this, vp::IoReq *req);
    static void            tcdm_grant(vp::Block *__this, vp::IoReq *req);

    int32_t acquire();
    void    commit(bool start);
    void    start_next_job();
    void    launch_job();
    void    soft_clear(uint32_t value);
    void    clear_engine();
    void    clear_events();
    void    finish_job();
    void    init_mxcore_meta_data();
    void    check_state();
    void    process_compute();
    void    process_preload();
    void    process_standalone();
    void    push_result_row(const std::vector<uint32_t> &row);
    void    retire_result(std::vector<uint32_t> result, uint8_t flags);
    bool    finished() const;

    uint32_t next_addr(unsigned stream);
    uint32_t calculate_tile_base_address(uint32_t base, uint32_t tile_bytes, unsigned tiles_per_row,
                                         unsigned i, unsigned j) const;
    uint32_t inc_addr(uint32_t addr, uint32_t stride) const;
    bool     next_iteration(bool count_in, bool count_out);
    uint32_t get_routine_access_block_number();
    uint32_t get_storing_access_block_number();
    void     process_iter_instruction();
    unsigned process_iter_instruction(uint8_t *buf, unsigned instr);
    bool     stream_done(unsigned stream) const;

    void             init_input_buffers();
    FifoLanes       &get_input_buffer(unsigned stream);
    const FifoLanes &get_input_buffer(unsigned stream) const;
    bool             input_can_push(unsigned stream) const;
    bool             input_valid(unsigned stream) const;
    Bytes            input_peek(unsigned stream) const;
    void             input_pop(unsigned stream);

    void                     init_output_buffers();
    std::deque<Bytes>       &get_output_buffer(unsigned output);
    const std::deque<Bytes> &get_output_buffer(unsigned output) const;
    bool                     output_can_push(unsigned output) const;
    void                     output_push(unsigned output, Bytes row);
    bool                     output_valid(unsigned output, bool final) const;
    unsigned                 output_peek(unsigned output, uint8_t *data) const;
    void                     output_pop(unsigned output);

    bool     issue_slot_gate_ok(unsigned stream);
    static unsigned hci_channel(unsigned stream);
    int      response_arrival(unsigned channel);
    bool     response_consumer_ready(unsigned channel) const;
    void     init_req_slot(unsigned slot_id);
    unsigned alloc_req_slot(unsigned stream);
    void     free_req_slot(unsigned slot_id);
    bool     all_slots_free() const;
    void     enqueue_read_requests();
    void     enqueue_preload_request();
    void     enqueue_write_request(bool result_valid, bool scale_valid);
    int      arbitrate_request();
    void     issue_request(unsigned slot_id);
    void     accept_request(unsigned slot_id);
    void     handle_response(unsigned slot_id, int64_t response_cycle);
    void     process_responses();
    void     trace_compute_state();
    void     trace_responses();
    static void trace_pulse(vp::Event &event, uint32_t value = 1);

    static bool                      power_of_two(unsigned value);
    static void                      append32(Bytes &bytes, uint32_t value);

    // MXCore interfaces
    vp::Trace            trace;
    vp::IoSlave          input_itf;
    vp::WireMaster<bool> done;
    vp::WireMaster<bool> events[MAX_CORES][NB_EVENTS];
    vp::IoMaster         tcdm_itf;

    // MXCore FSM
    vp::ClockEvent        *fsm_event;
    vp::ClockEvent        *fsm_update_event;
    vp::Register<uint32_t> reg_fsm_state;
    vp::Register<uint32_t> reg_busy;
    // State snapshots use registers; handshake/completion pulses use events.
    vp::Event wave_req, wave_gnt, wave_rsp;
    vp::Register<uint32_t> wave_addr, wave_stream;
    vp::Event wave_dotp_fire;
    vp::Register<uint32_t> wave_pipe_valid;
    vp::Event wave_result_fire;
    vp::Register<uint32_t> wave_preload_wait, wave_preload_count;
    vp::Event wave_events;
    uint32_t wave_event_bits = 0;
    uint32_t               tcdm_block_total = 0; // Planned memory transfers for this job.
    uint32_t               fsm_counter = 0;      // Transfers submitted to the TCDM port.
    int64_t                next_bus_cycle = 0;
    int64_t                timer_start = 0; // Job start time in GVSOC picoseconds.
    int64_t                cycle_start = 0;
    int64_t                total_runtime = 0; // Accumulated completed-job time in ns.
    uint64_t               num_matmul = 0;

    // HWPE job control
    enum class StartupPhase { NONE, COMMITTED, ADDRESS_FIFO };
    StartupPhase startup_phase = StartupPhase::NONE;
    int64_t startup_edge = 0;
    bool finish_pending = false; // Output event precedes the registered idle state.
    uint8_t  job_id_counter = 0;
    int      job_state = 0;      // 0: free to acquire; -2: acquired, not committed.
    uint8_t  running_job_id = 0; // RTL RUNNING counts retired jobs, modulo 256.
    uint8_t  finished_count = 0;
    uint32_t context_status = 0; // STATUS storage; survives nonzero SOFT_CLEAR.
    unsigned offloading_core = 0;
    bool     event_pending = false;
    uint8_t  sw_evt = 0;

    // MXCore configuration
    unsigned tcdm_bank_width;
    unsigned tcdm_bank_number;
    unsigned vs;
    unsigned npe;
    unsigned reuse;
    unsigned pipe_regs;
    unsigned preload_threshold;
    unsigned num_cores; // CPU cores with completion-event outputs.
    unsigned bandwidth;
    MXDotp   dotp;

    // MXCore registers
    std::array<uint32_t, NB_IO_REGS> reg_file{}; // gwaihir configuration words at 0x20..0x5c.
    unsigned                         m_size = 0;
    unsigned                         n_size = 0;
    unsigned                         k_size = 0;
    uint32_t                         a_addr = 0;
    uint32_t                         b_addr = 0;
    uint32_t                         scale_a_addr = 0;
    uint32_t                         scale_b_addr = 0;
    uint32_t                         preload_addr = 0;
    uint32_t                         z_addr = 0;
    uint32_t                         scale_z_addr = 0;
    InputFormat                      src_fmt = InputFormat::E5M2;
    unsigned                         elements_per_dotp = 0;
    unsigned                         input_block_size = 32;
    OutputFormat                     output_format = OutputFormat::FP32;
    unsigned                         result_bits = 32;
    bool                             quantize_mx = false;
    bool                             block_poison = false;
    bool                             preload = false;
    bool                             standalone = false;
    bool                             standalone_input_bf16 = false;

    // MXCore meta data
    // Programmed tile counts and tile sizes in bits; checked against M/N/K, VS, NPE and Reuse.
    unsigned a_row_tiles = 0;
    unsigned b_col_tiles = 0;
    unsigned inner_tiles = 0; // Programmed K/VS, distinct from FP4 compute chunks K/(2*VS).
    unsigned inner_blocks = 0;
    unsigned sa_vec_per_block = 0;
    uint32_t a_tile_size = 0;
    uint32_t b_tile_size = 0;
    uint32_t result_tile_size = 0;
    unsigned iter_count = 0;
    unsigned in_count = 0;
    unsigned out_count = 0;

    // Next byte address of each stream; advanced when a request is enqueued.
    uint32_t                         iter_a_addr = 0;
    uint32_t                         iter_b_addr = 0;
    uint32_t                         iter_scale_a_addr = 0;
    uint32_t                         iter_scale_b_addr = 0;
    uint32_t                         iter_z_addr = 0;
    uint32_t                         iter_scale_z_addr = 0;
    uint32_t                         iter_preload_addr = 0;
    std::array<uint32_t, NB_STREAMS> stream_total{};
    std::array<uint32_t, NB_STREAMS> stream_issued{}; // Beats enqueued; drives address progression.
    uint32_t                         a_row_tile_bytes = 0;
    uint32_t                         b_matrix_bytes = 0;
    uint32_t                         scale_a_row_tile_bytes = 0;
    uint32_t                         scale_b_matrix_bytes = 0;

    uint64_t expected_bundles = 0;
    uint64_t compute_count = 0;
    uint64_t retired_count = 0;
    uint64_t expected_rows = 0;
    uint64_t final_rows = 0;
    uint8_t  status = 0;
    unsigned partial_read_row = 0;
    unsigned partial_write_row = 0;
    unsigned result_read_row = 0;

    // Input stream metadata; array order is A, B, Scale A, Scale B.
    std::array<unsigned, NB_INPUT_STREAMS>              input_width{};
    std::array<unsigned, NB_INPUT_STREAMS>              input_push_lane{};
    std::array<unsigned, NB_INPUT_STREAMS>              input_slice{};
    std::array<unsigned, NB_INPUT_STREAMS>              input_row{};
    std::array<unsigned, NB_INPUT_STREAMS>              input_scale_repeat{};
    std::array<unsigned, NB_INPUT_STREAMS>              input_block{};
    std::array<bool, NB_INPUT_STREAMS> input_push_ready{}; // FIFO readiness sampled before the edge.
    bool                               scale_b_short = false;

    // Output stream metadata; array order is result data, result scales.
    std::array<unsigned, NB_OUTPUT_STREAMS> output_row_bytes{};
    std::array<unsigned, NB_OUTPUT_STREAMS> output_lanes{};
    std::array<unsigned, NB_OUTPUT_STREAMS> output_slice{};
    bool                                    result_rr_counter = false;

    // MXCore buffers
    FifoLanes                           a_buffer;
    FifoLanes                           b_buffer;
    FifoLanes                           scale_a_buffer;
    FifoLanes                           scale_b_buffer;
    
    // one NPE-wide result, exception status and valid bit per stage.
    std::vector<std::vector<uint32_t>> dotp_result_buffer;
    std::vector<uint8_t>               dotp_result_status;
    std::vector<bool>                  dotp_result_valid;
    
    //Reuse x NPE FP32 bits, row validity and final-row flags.
    std::vector<std::vector<uint32_t>> partial_sum_buffer;
    std::vector<bool>                  partial_sum_valid;
    std::vector<bool>                  partial_sum_final;
    std::deque<Bytes>                  result_buffer;
    std::deque<Bytes>                  result_scale_buffer;
    std::deque<Bytes>                  preload_buffer;
    unsigned                           preload_row = 0;
    unsigned                           preload_word = 0;
    unsigned                           preload_count = 0;
    unsigned                           preload_final_count = 0;
    bool                               preload_wait = false;
    bool                               preload_push_ready = false;

    // Standalone input repacking bypasses the accumulation buffer and MXDOTP.
    FifoLanes                          standalone_buffer;
    unsigned                           standalone_push_lane = 0;
    unsigned                           standalone_slice = 0;
    uint32_t                           standalone_input_count = 0;
    bool                               standalone_input_done = false;

    struct ReqSlot {
        vp::IoReq       *req;
        uint8_t         *buf;
        unsigned         stream;
        mxcore_req_state state;
        uint64_t epoch;
        int64_t  issued_at;
        int64_t available_at;
        bool response_traced = false;
    };

    std::vector<ReqSlot> req_slots;
    std::array<mxcore::HciFifo, NB_TCDM_CHANNELS> hci_fifos;
    int      denied_req_slot = -1; // Denied request awaiting grant; -1 means none.
    uint64_t epoch = 0;
};

/************************************************************
 *              Component Construction and Reset              *
 ************************************************************/

extern "C" vp::Component *gv_new(vp::ComponentConf &config) {
    return new MXCore(config);
}

MXCore::MXCore(vp::ComponentConf &config)
    : vp::Component(config), reg_fsm_state(*this, "state", 2, true, IDLE),
      reg_busy(*this, "busy", 1, true, 0),
      wave_req(*this, "timing/tcdm_req", 1),
      wave_gnt(*this, "timing/tcdm_gnt", 1),
      wave_rsp(*this, "timing/tcdm_rsp", 1),
      wave_addr(*this, "timing/tcdm_addr", 32, true, 0),
      wave_stream(*this, "timing/tcdm_stream", 3, true, 0),
      wave_dotp_fire(*this, "timing/dotp_fire", 2),
      wave_pipe_valid(*this, "timing/pipe_valid", 4, true, 0),
      wave_result_fire(*this, "timing/result_fire", 1),
      wave_preload_wait(*this, "timing/preload_wait", 1, true, 0),
      wave_preload_count(*this, "timing/preload_count", 32, true, 0),
      wave_events(*this, "timing/events", 2),
      // MXDotp has no default constructor; its arithmetic/configuration is unchanged.
      dotp(get_js_config()->get("vector_size")->get_int()) {

    // Initialize configuration.
    this->tcdm_bank_width = get_js_config()->get("tcdm_bank_width")->get_int();
    this->tcdm_bank_number = get_js_config()->get("tcdm_bank_number")->get_int();
    this->vs = get_js_config()->get("vector_size")->get_int();
    this->npe = get_js_config()->get("npe")->get_int();
    this->reuse = get_js_config()->get("reuse")->get_int();
    this->pipe_regs = get_js_config()->get("num_pipe_regs")->get_int();
    this->preload_threshold = get_js_config()->get("preload_threshold")->get_int();
    this->num_cores = get_js_config()->get("num_cores")->get_int();
    this->bandwidth = this->tcdm_bank_width * this->tcdm_bank_number;

    if (this->pipe_regs != 4) {
        throw std::invalid_argument("MXCore requires four pipeline stages to match fpnew_mxdotp_multi_opt");
    }

    if (!power_of_two(this->vs) || this->vs < 2 || this->vs > 32 || !power_of_two(this->npe) ||
        this->npe < 32 || this->npe > 512 || !power_of_two(this->reuse) || this->reuse < 2 ||
        this->reuse > 512 || !power_of_two(this->bandwidth) || this->bandwidth < std::max(4u, this->vs) ||
        this->bandwidth > 512 || this->preload_threshold > this->reuse ||
        !this->num_cores || this->num_cores > MAX_CORES) {
        throw std::invalid_argument("Unsupported MXCore VS/NPE/Reuse/pipeline/TCDM configuration");
    }

    // Initialize interfaces and the clocked FSM
    this->fsm_event = this->event_new(&MXCore::fsm_handler);
    this->fsm_update_event = this->event_new(&MXCore::fsm_update);
    this->input_itf.set_req_meth(&MXCore::req);
    this->new_slave_port("input", &this->input_itf);
    this->new_master_port("done_irq", &this->done);

    for (unsigned core = 0; core < this->num_cores; ++core) {
        for (unsigned event = 0; event < NB_EVENTS; ++event) {
            this->new_master_port("event_" + std::to_string(core) + "_" + std::to_string(event),
                                  &this->events[core][event]);
        }
    }

    this->new_master_port("tcdm", &this->tcdm_itf);
    this->tcdm_itf.set_resp_meth(&MXCore::tcdm_response);
    this->tcdm_itf.set_grant_meth(&MXCore::tcdm_grant);
    this->traces.new_trace("trace", &this->trace);
    
    this->req_slots.resize(NB_TCDM_CHANNELS * STREAM_FIFO_DEPTH);
    for (unsigned i = 0; i < this->req_slots.size(); ++i) {
        this->init_req_slot(i);
    }

    this->dotp_result_buffer.resize(this->pipe_regs);
    this->dotp_result_status.resize(this->pipe_regs);
    this->dotp_result_valid.assign(this->pipe_regs, false);
}

MXCore::~MXCore() {
    if (this->fsm_event->is_enqueued()) {
        this->event_cancel(this->fsm_event);
    }
    delete this->fsm_event;
    if (this->fsm_update_event->is_enqueued()) {
        this->event_cancel(this->fsm_update_event);
    }
    delete this->fsm_update_event;
    for (auto &slot : this->req_slots) {
        this->tcdm_itf.req_del(slot.req);
        delete[] slot.buf;
    }
}

void MXCore::clear_events() {
    if (this->done.is_bound()) {
        this->done.sync(false);
    }

    for (unsigned core = 0; core < this->num_cores; ++core) {
        for (unsigned event = 0; event < NB_EVENTS; ++event) {
            if (this->events[core][event].is_bound()) {
                this->events[core][event].sync(false);
            }
        }
    }

    this->event_pending = false;
    this->sw_evt = 0;
    this->wave_event_bits = 0;
    this->wave_events.dump(reinterpret_cast<uint8_t *>(&this->wave_event_bits));
}

void MXCore::reset(bool active) {
    if (active) {
        this->soft_clear(0);
        this->total_runtime = 0;
        this->num_matmul = 0;
    }
}

void MXCore::clear_engine() {
    ++this->epoch;
    this->startup_phase = StartupPhase::NONE;
    this->startup_edge = 0;
    this->finish_pending = false;
    this->reg_fsm_state.set(IDLE);
    this->reg_busy.set(0);

    if (this->fsm_event->is_enqueued()) {
        this->event_cancel(this->fsm_event);
    }
    if (this->fsm_update_event->is_enqueued()) {
        this->event_cancel(this->fsm_update_event);
    }

    for (auto &fifo : this->hci_fifos) {
        fifo.clear();
    }
    this->result_rr_counter = false;

    for (unsigned i = 0; i < this->req_slots.size(); ++i) {
        if (this->req_slots[i].state != REQ_PENDING) {
            this->free_req_slot(i);
        }
    }

    this->a_buffer.clear();
    this->b_buffer.clear();
    this->scale_a_buffer.clear();
    this->scale_b_buffer.clear();
    this->partial_sum_buffer.clear();
    this->partial_sum_valid.clear();
    this->partial_sum_final.clear();
    this->result_buffer.clear();
    this->result_scale_buffer.clear();
    this->preload_buffer.clear();
    this->preload_row = this->preload_word = this->preload_count = this->preload_final_count = 0;
    this->preload_wait = this->preload_push_ready = false;
    this->standalone_buffer.clear();
    this->standalone_push_lane = this->standalone_slice = this->standalone_input_count = 0;
    this->standalone_input_done = false;

    for (auto &stage : this->dotp_result_buffer) {
        stage.clear();
    }

    this->dotp_result_valid.assign(this->pipe_regs, false);
    this->job_state = 0;
    this->clear_events();
    uint32_t zero = 0;
    for (vp::Event *event : {&this->wave_req, &this->wave_gnt, &this->wave_rsp,
                            &this->wave_dotp_fire, &this->wave_result_fire}) {
        event->dump(reinterpret_cast<uint8_t *>(&zero));
    }
    this->wave_addr.set(0);
    this->wave_stream.set(0);
    this->trace_compute_state();
}

// every SOFT_CLEAR write clears control/data-path state; only zero
// also clears the register file and its job/finished counters.
void MXCore::soft_clear(uint32_t value) {
    this->clear_engine();
    this->offloading_core = 0;
    if (value == 0) {
        this->reg_file.fill(0);
        this->context_status = 0;
        this->job_id_counter = 0;
        this->running_job_id = 0;
        this->finished_count = 0;
    }
}

/************************************************************
 *              Peripheral Input and Job Control              *
 ************************************************************/

vp::IoReqStatus MXCore::req(vp::Block *__this, vp::IoReq *req) {
    MXCore          *_this = static_cast<MXCore *>(__this);
    const uint64_t  address = req->get_addr();
    const unsigned  size = req->get_size();
    const unsigned  lane = address & 3;
    const uint64_t  offset = address & ~uint64_t(3);
    uint8_t         *data = req->get_data();

    if (!data || !size || size > 4 || lane + size > 4 || offset >= 0x20 + 4 * NB_IO_REGS) {
        return vp::IO_REQ_INVALID;
    }

    req->inc_latency(1); // Registered peripheral response
    
    uint32_t value = 0;
    if (req->get_is_write()) {
        for (unsigned i = 0; i < size; ++i) {
            value |= uint32_t(data[i]) << (8 * (lane + i));
        }

        if (offset >= 0x20) {
            uint32_t      &reg = _this->reg_file[(offset - 0x20) / 4];
            const uint32_t mask = uint32_t((uint64_t(1) << (size * 8)) - 1) << (lane * 8);
            reg = (reg & ~mask) | value;
        } else {
            switch (offset) {
            case 0x00: // TRIGGER: any write commits; zero also starts.
            {
                int core = req->get_initiator();
                if (core == -1 && _this->num_cores == 1) {
                    core = 0;
                }

                if (core < 0 || unsigned(core) >= _this->num_cores) {
                    return vp::IO_REQ_INVALID;
                }

                _this->offloading_core = core;
                try {
                    _this->commit(value == 0);
                } catch (const std::exception &error) {
                    _this->trace.force_warning_no_error("Unsupported MXCore configuration: %s\n",
                                                       error.what());
                    _this->time.get_engine()->quit(1);
                    return vp::IO_REQ_INVALID;
                }
                break;
            }
            case 0x08:
                _this->finished_count = 0;
                break;
            case 0x14:
                _this->soft_clear(value);
                break;
            case 0x1c: // SWEVT is internal to hwpe_ctrl_slave, unused by MXCore.
                _this->sw_evt = (value & 15) < 8 ? 1u << (value & 15) : 0;
                _this->check_state();
                break;
            default:
                break;
            }
        }
    } else {
        if (offset >= 0x20) {
            value = _this->reg_file[(offset - 0x20) / 4];
        } else {
            switch (offset) {
            case 0x00:
                value = 0xdeadbeef;
                break; // No readable TRIGGER storage.
            case 0x04:
                value = uint32_t(_this->acquire());
                break;
            case 0x08:
                value = _this->finished_count;
                _this->finished_count = 0;
                break;
            case 0x0c:
                value = _this->context_status;
                break;
            case 0x10:
                value = _this->running_job_id;
                break;
            default:
                break;
            }
        }
        for (unsigned i = 0; i < size; ++i) {
            data[i] = value >> (8 * (lane + i));
        }
    }
    return vp::IO_REQ_OK;
}

int32_t MXCore::acquire() {
    if (this->job_state == -2) {
        return -2; 
    }
    if (this->reg_busy.get() || this->startup_phase != StartupPhase::NONE) {
        return -1;
    }
    this->job_state = -2;
    return this->job_id_counter++;
}

void MXCore::commit(bool start) {
    this->job_state = 0;

    if (start && !this->reg_busy.get() && this->startup_phase == StartupPhase::NONE) {
        this->start_next_job();
    }
    if (start) {
        this->context_status = 1;
    }
}

void MXCore::start_next_job() {
    this->a_addr = this->reg_file[0];
    this->b_addr = this->reg_file[1];
    this->scale_a_addr = this->reg_file[2];
    this->scale_b_addr = this->reg_file[3];
    this->preload_addr = this->reg_file[4];
    this->z_addr = this->reg_file[5];
    this->scale_z_addr = this->reg_file[6];
    this->m_size = this->reg_file[7] & 0x3ff;
    this->k_size = (this->reg_file[7] >> 10) & 0xfff;
    this->n_size = (this->reg_file[7] >> 22) & 0x3ff;
    const uint32_t ctrl = this->reg_file[8];
    this->standalone = (ctrl >> 27) & 1;
    this->standalone_input_bf16 = this->standalone && ((ctrl >> 28) & 1);
    this->src_fmt = InputFormat((ctrl >> 9) & 15);
    if (!this->standalone) {
        mxcore::get_input_encoding(this->src_fmt);
    }
    const bool fp4 = this->src_fmt == InputFormat::E2M1;
    this->elements_per_dotp = (fp4 ? 2u : 1u) * this->vs;
    this->input_block_size = fp4 ? 64 : 32;

    // Match gwaihir's highest-output-bit priority.
    if (ctrl & (1u << 24)) {
        this->output_format = OutputFormat::BF16;
    } else if (ctrl & (1u << 23)) {
        this->output_format = OutputFormat::E5M2;
    } else if (ctrl & (1u << 22)) {
        this->output_format = OutputFormat::E4M3;
    } else if (ctrl & (1u << 21)) {
        this->output_format = OutputFormat::E2M1;
    } else {
        this->output_format = OutputFormat::FP32;
    }

    this->result_bits = mxcore::output_bits(this->output_format);
    this->quantize_mx = mxcore::output_has_scale(this->output_format);
    this->block_poison = (ctrl >> 25) & 1;
    this->preload = ((ctrl >> 26) & 1) && !this->standalone;
    this->a_row_tiles = this->reg_file[9] & 15; // M / Reuse
    this->b_col_tiles = (this->reg_file[9] >> 4) & 31; // N / NPE
    this->inner_tiles = (this->reg_file[9] >> 9) & 127; // K / VS
    this->inner_blocks = (this->reg_file[9] >> 16) & 127; // ceil(K / input_block_size)
    this->sa_vec_per_block = (this->reg_file[9] >> 23) & 63;
    this->a_tile_size = this->reg_file[10];
    this->b_tile_size = this->reg_file[11];
    this->result_tile_size = this->reg_file[13];
    this->iter_count = this->reg_file[15] & 0xffff; // (K / elements_per_dotp) * Reuse

    if (fp4 && !this->standalone && this->vs != 32) {
        throw std::invalid_argument("gwaihir FP4 input requires hardware VS=32");
    }

    const unsigned op = (ctrl >> 3) & 31;
    // Existing numerical contract: RNE, floating MXDOTP, FP32 accumulation.
    // SDOTP=19 is the workspace testbench encoding; MXDOTPF=22 is accepted too.
    if (!this->standalone &&
        ((ctrl & 7) || ((ctrl >> 13) & 15) || (op != 19 && op != 22) || (ctrl & (1u << 20)))) {
        throw std::invalid_argument("requires RNE, SDOTP/MXDOTPF, FP32 destination and flush=0");
    }

    if (!this->m_size || !this->n_size || this->m_size % this->reuse || this->n_size % this->npe) {
        throw std::invalid_argument("requires full Reuse/NPE tiles");
    }

    // A single short-K tile finishes before the fixed RTL scale reuse counters
    // need to advance to the next tile. Keep other partial-block jobs unsupported.
    const bool single_short_block = this->k_size < this->input_block_size &&
                                    this->m_size == this->reuse && this->n_size == this->npe;
    if (!this->standalone &&
        (!this->k_size || this->k_size % this->elements_per_dotp ||
         (this->k_size % this->input_block_size && !single_short_block) ||
         this->k_size / this->elements_per_dotp * uint64_t(this->reuse) > 65535)) {
        throw std::invalid_argument(
            "requires complete input vectors, a 16-bit iteration count, and full scale blocks "
            "except for a single short-K output tile");
    }

    const unsigned input_bits = fp4 ? 4u : 8u;
    const unsigned compute_chunks = this->standalone ? 0 : this->k_size / this->elements_per_dotp;
    if (this->a_row_tiles != this->m_size / this->reuse || this->b_col_tiles != this->n_size / this->npe ||
        this->result_tile_size != this->reuse * this->npe * this->result_bits ||
        (this->quantize_mx && this->reg_file[14] != this->reuse * (this->npe / 32) * 8)) {
        throw std::invalid_argument("inconsistent programmed output tile counts/sizes");
    }

    if (!this->standalone &&
        (this->inner_tiles != this->k_size / this->vs ||
         this->inner_blocks != (this->k_size + this->input_block_size - 1) / this->input_block_size ||
         this->a_tile_size != this->reuse * this->vs * input_bits ||
         this->b_tile_size != this->npe * this->vs * input_bits ||
         this->iter_count != compute_chunks * this->reuse ||
         this->sa_vec_per_block != std::min(32u / this->vs, compute_chunks))) {
        throw std::invalid_argument("inconsistent programmed GEMM tile sizes/iteration count");
    }

    const unsigned c_bits = this->standalone_input_bf16 ? 16 : 32;
    if ((this->preload || this->standalone) &&
        this->reg_file[12] != this->reuse * this->npe * c_bits) {
        throw std::invalid_argument("inconsistent programmed C input tile size");
    }
    if (this->preload && this->npe * 4 < this->bandwidth) {
        throw std::invalid_argument("Preload requires NPE*32 >= TCDM width");
    }

    if (!this->tcdm_itf.is_bound()) {
        throw std::logic_error("connect MXCore TCDM before triggering");
    }

    this->init_mxcore_meta_data();
    ++this->epoch;
    this->status = 0;
    this->in_count = this->out_count = this->partial_read_row = this->partial_write_row =
        this->result_read_row = 0;
    this->expected_rows = uint64_t(this->m_size) * this->n_size / this->npe;
    this->expected_bundles = this->expected_rows * compute_chunks;
    this->compute_count = 0;
    this->retired_count = 0;
    this->final_rows = 0;
    this->init_input_buffers();
    this->init_output_buffers();
    this->partial_sum_buffer.assign(this->standalone ? 0 : this->reuse,
                                    std::vector<uint32_t>(this->npe));
    this->partial_sum_valid.assign(this->standalone ? 0 : this->reuse, false);
    this->partial_sum_final.assign(this->standalone ? 0 : this->reuse, false);
    this->preload_buffer.clear();
    this->preload_row = this->preload_word = this->preload_count = this->preload_final_count = 0;
    this->preload_wait = this->preload;
    this->preload_push_ready = false;
    this->standalone_push_lane = this->standalone_slice = this->standalone_input_count = 0;
    this->standalone_input_done = false;
    this->standalone_buffer.clear();
    const unsigned standalone_row_bytes = this->npe * c_bits / 8;
    if (this->standalone && !(this->standalone_input_bf16 && standalone_row_bytes == this->bandwidth)) {
        this->standalone_buffer.assign(std::max(1u, standalone_row_bytes / this->bandwidth), {});
    }

    for (auto &stage : this->dotp_result_buffer) {
        stage.clear();
    }

    this->dotp_result_valid.assign(this->pipe_regs, false);
    std::fill(this->dotp_result_status.begin(), this->dotp_result_status.end(), 0);
    this->trace_compute_state();
    this->result_rr_counter = false;
    for (auto &fifo : this->hci_fifos) {
        fifo.clear();
    }
    // TRIGGER clocks counter_pending first. MXCoreIdle consumes that pending
    // context at the next edge; busy must not rise inside the MMIO callback.
    this->startup_phase = StartupPhase::COMMITTED;
    this->startup_edge = this->clock.get_cycles() + 1;
    this->check_state();
}

void MXCore::launch_job() {
    this->reg_fsm_state.set(PRELOAD);
    this->reg_busy.set(1);
    this->timer_start = this->time.get_time();
    this->cycle_start = this->clock.get_cycles();
    this->trace.msg(vp::TraceLevel::DEBUG,
                    "MXCore start: M=%u K=%u N=%u VS=%u NPE=%u Reuse=%u BW=%uB/cycle "
                    "pipe=%u standalone=%u C=%u-bit\n", this->m_size,
                    this->k_size, this->n_size, this->vs, this->npe, this->reuse, this->bandwidth,
                    this->pipe_regs, unsigned(this->standalone), this->standalone_input_bf16 ? 16u : 32u);
}

void MXCore::finish_job() {
    const int64_t period_clk = this->clock.get_cycles() - this->cycle_start;
    const int64_t period_ns = (this->time.get_time() - this->timer_start) / 1000;
    this->total_runtime += period_ns;
    ++this->num_matmul;
    ++this->running_job_id;

    if (this->finished_count < 2) {
        ++this->finished_count;
    }

    // RTL job_done/evt[1] is combinational in Output. state_q returns to idle
    // at the following edge, so busy remains asserted for this event cycle.
    this->finish_pending = true;
    this->trace.msg(vp::TraceLevel::INFO,
                    "MXCore finished: %lld cycles, %lld ns, %llu bundles, status=0x%02x, "
                    "total=%lld ns, GEMM=%llu\n",
                    (long long)period_clk, (long long)period_ns, (unsigned long long)this->compute_count,
                    unsigned(this->status), (long long)this->total_runtime,
                    (unsigned long long)this->num_matmul);
    this->event_pending = true;
    this->check_state(); 
    const unsigned core = this->offloading_core;

    if (this->events[core][1].is_bound()) {
        this->events[core][1].sync(true);
    }
    this->wave_event_bits |= 2;
    this->wave_events.dump(reinterpret_cast<uint8_t *>(&this->wave_event_bits));

    if (this->done.is_bound()) {
        this->done.sync(true);
    }
}

// i_streamer
void MXCore::init_mxcore_meta_data() {
    const uint64_t ar = this->a_row_tiles, bc = this->b_col_tiles, kt = this->inner_tiles,
                   kb = this->inner_blocks;
    const uint64_t bw = this->bandwidth * 8, a_tile = this->a_tile_size, b_tile = this->b_tile_size,
                   z_tile = this->result_tile_size;
    // INNER_TILES retains K/VS; FP4 tile sizes contain four bits per element.
    const uint64_t arow = a_tile * kt, bmat = b_tile * kt * bc;
    // E8M0 scales: ceil(K/32) blocks for FP8, ceil(K/64) for FP4.
    const uint64_t satile = this->reuse * 8, sarow = satile * kb, samat = sarow * ar,
                   sbmat = this->npe * 8 * kb * bc;

    if (z_tile * ar * bc % bw ||
        ((this->preload || this->standalone) && uint64_t(this->reg_file[12]) * ar * bc % bw) ||
        (!this->standalone && (arow % bw || bmat % bw || (sbmat >= bw && sbmat % bw)))) {
        throw std::invalid_argument("Job requires integral RTL stream beats");
    }

    if (!this->standalone && satile > bw && ar != 1) {
        throw std::invalid_argument("RTL large ScaleA branch supports one row tile");
    }

    if (!this->standalone && satile <= bw && sarow % bw && (bc > 1 || samat > bw)) {
        throw std::invalid_argument("RTL ScaleA row-tile replay requires integral beats");
    }

    if (!this->standalone && sbmat < bw && kb * bc * ar != 1) {
        throw std::invalid_argument("RTL short ScaleB buffer cannot replay additional groups");
    }

    const uint64_t scale_bytes = uint64_t(this->m_size) * this->n_size / 32;
    if (this->quantize_mx && (this->bandwidth < this->npe / 32 ||
                                  (scale_bytes >= this->bandwidth && scale_bytes % this->bandwidth))) {
        throw std::invalid_argument("RTL scale output needs supported integral/tail packing");
    }

    this->a_row_tile_bytes = arow / 8;
    this->b_matrix_bytes = bmat / 8;
    this->scale_a_row_tile_bytes = sarow / 8;
    this->scale_b_matrix_bytes = sbmat / 8;

    this->iter_a_addr = this->a_addr;
    this->iter_b_addr = this->b_addr;
    this->iter_scale_a_addr = this->scale_a_addr;
    this->iter_scale_b_addr = this->scale_b_addr;
    this->iter_z_addr = this->z_addr;
    this->iter_scale_z_addr = this->scale_z_addr;
    this->iter_preload_addr = this->preload_addr;
    this->stream_total.fill(0);
    this->stream_issued.fill(0);
    this->fsm_counter = 0;
    this->tcdm_block_total =
        this->get_routine_access_block_number() + this->get_storing_access_block_number();

    const uint32_t bases[NB_STREAMS] = {this->a_addr,       this->b_addr, this->scale_a_addr,
                                        this->scale_b_addr, this->z_addr, this->scale_z_addr,
                                        this->preload_addr};

    for (unsigned stream = 0; stream < NB_STREAMS; ++stream) {
        if (uint64_t(bases[stream]) + uint64_t(this->stream_total[stream]) * this->bandwidth >
            uint64_t(UINT32_MAX) + 1) {
            throw std::invalid_argument("MXCore stream address/count overflow");
        }
    }
}

uint32_t MXCore::get_routine_access_block_number() {
    this->stream_total[STREAM_C] =
        (this->preload || this->standalone)
            ? uint64_t(this->reg_file[12]) * this->a_row_tiles * this->b_col_tiles /
                  (this->bandwidth * 8)
            : 0;
    if (this->standalone) {
        // The four GEMM sources never start in standalone mode.
        return this->stream_total[STREAM_C];
    }

    this->stream_total[STREAM_A] =
        uint64_t(this->a_row_tile_bytes) * this->a_row_tiles * this->b_col_tiles / this->bandwidth;
    this->stream_total[STREAM_B] = uint64_t(this->b_matrix_bytes) * this->a_row_tiles / this->bandwidth;

    if (this->reuse > this->bandwidth) {
        this->stream_total[STREAM_SA] = uint64_t(this->reuse) *
                                        this->sa_vec_per_block * this->inner_blocks *
                                        this->b_col_tiles / this->bandwidth;
    } else {
        this->stream_total[STREAM_SA] =
            (uint64_t(this->scale_a_row_tile_bytes) * this->a_row_tiles * this->b_col_tiles +
             this->bandwidth - 1) /
            this->bandwidth;
    }

    this->stream_total[STREAM_SB] =
        this->scale_b_matrix_bytes < this->bandwidth
            ? 1
            : uint64_t(this->scale_b_matrix_bytes) * this->a_row_tiles / this->bandwidth;

    return this->stream_total[STREAM_A] + this->stream_total[STREAM_B] + this->stream_total[STREAM_SA] +
           this->stream_total[STREAM_SB] + this->stream_total[STREAM_C];
}

uint32_t MXCore::get_storing_access_block_number() {
    this->stream_total[STREAM_Z] =
        uint64_t(this->result_tile_size) * this->a_row_tiles * this->b_col_tiles / (this->bandwidth * 8);
    const uint64_t scale_bytes = uint64_t(this->m_size) * this->n_size / 32;
    this->stream_total[STREAM_SZ] =
        this->quantize_mx ? std::max(uint64_t(1), scale_bytes / this->bandwidth) : 0;
    return this->stream_total[STREAM_Z] + this->stream_total[STREAM_SZ];
}

/************************************************************
 *                TCDM Requests and Responses                 *
 ************************************************************/

void MXCore::init_req_slot(unsigned slot_id) {
    ReqSlot &slot = this->req_slots[slot_id];
    slot.req = this->tcdm_itf.req_new(0, nullptr, 0, false);
    slot.buf = new uint8_t[this->bandwidth];
    slot.stream = STREAM_A;
    slot.state = REQ_FREE;
    slot.epoch = 0;
    slot.issued_at = 0;
    slot.available_at = 0;
    slot.req->arg_alloc(1);
    std::memcpy(slot.req->arg_get(0), &slot_id, sizeof(slot_id));
}

bool MXCore::issue_slot_gate_ok(unsigned stream) {
    return this->hci_fifos[hci_channel(stream)].request_push_ready(this->clock.get_cycles());
}

unsigned MXCore::hci_channel(unsigned stream) {
    // Z and SZ enter the same outgoing FIFO after i_result_fifo_mux.
    return stream == STREAM_C ? 5 : stream == STREAM_SZ ? STREAM_Z : stream;
}

int MXCore::response_arrival(unsigned channel) {
    const auto now = this->clock.get_cycles();
    int arrival = -1;
    for (unsigned i = 0; i < this->req_slots.size(); ++i) {
        const ReqSlot &slot = this->req_slots[i];
        if (slot.epoch != this->epoch || slot.state != REQ_RETURNED || slot.req->get_is_write() ||
            hci_channel(slot.stream) != channel || slot.available_at > now || slot.issued_at >= now) {
            continue;
        }
        if (arrival >= 0) {
            throw std::logic_error("multiple MXCore HCI responses on one channel in one cycle");
        }
        arrival = int(i);
    }
    return arrival;
}

bool MXCore::response_consumer_ready(unsigned channel) const {
    return channel < NB_INPUT_STREAMS ? this->input_push_ready[channel]
           : channel == hci_channel(STREAM_C) ? this->preload_push_ready : true;
}

unsigned MXCore::alloc_req_slot(unsigned stream) {
    unsigned i = 0;
    while (i < this->req_slots.size() && this->req_slots[i].state != REQ_FREE) {
        ++i;
    }
    if (i == this->req_slots.size()) {
        this->req_slots.emplace_back();
        this->init_req_slot(i);
    }
    ReqSlot &slot = this->req_slots[i];
    slot.state = REQ_QUEUED;
    slot.stream = stream;
    slot.epoch = this->epoch;
    slot.available_at = this->clock.get_cycles() + 1;
    slot.response_traced = false;
    slot.req->prepare();
    slot.req->set_data(slot.buf);
    slot.req->status = vp::IO_REQ_OK;
    this->hci_fifos[hci_channel(stream)].push_request(i, this->clock.get_cycles());
    return i;
}

void MXCore::free_req_slot(unsigned slot_id) {
    this->req_slots[slot_id].state = REQ_FREE;
    this->req_slots[slot_id].available_at = 0;
}

bool MXCore::all_slots_free() const {
    for (const ReqSlot &slot : this->req_slots) {
        if (slot.epoch == this->epoch && slot.state != REQ_FREE) {
            return false;
        }
    }
    return true;
}

void MXCore::enqueue_read_requests() {
    for (unsigned i = 0; i < NB_INPUT_STREAMS; ++i) {
        if (!this->stream_done(i) && this->issue_slot_gate_ok(i)) {
            ReqSlot &slot = this->req_slots[this->alloc_req_slot(i)];
            slot.req->set_addr(this->next_addr(i));
            slot.req->set_size(this->bandwidth);
            slot.req->set_is_write(false);
        }
    }
}

void MXCore::enqueue_preload_request() {
    if (!(this->preload || this->standalone) || this->stream_done(STREAM_C) ||
        !this->issue_slot_gate_ok(STREAM_C)) {
        return;
    }
    ReqSlot &slot = this->req_slots[this->alloc_req_slot(STREAM_C)];
    slot.req->set_addr(this->next_addr(STREAM_C));
    slot.req->set_size(this->bandwidth);
    slot.req->set_is_write(false);
}

void MXCore::enqueue_write_request(bool z, bool sz) {
    if ((!z && !sz) || !this->issue_slot_gate_ok(STREAM_Z)) {
        return;
    }
    // hci_core_mux_dynamic's winner loop keeps the last requesting entry.
    // With rr_counter=0, simultaneous Z/SZ requests select SZ first.
    const unsigned i = z && (!sz || this->result_rr_counter) ? STREAM_Z : STREAM_SZ;
    const unsigned slot_id = this->alloc_req_slot(i);
    ReqSlot       &slot = this->req_slots[slot_id];
    const unsigned size = this->process_iter_instruction(slot.buf, i);
    slot.req->set_addr(this->next_addr(i));
    slot.req->set_size(size);
    slot.req->set_is_write(true);
    // Rotate only when a competing request loses an accepted FIFO push.
    if (z && sz) {
        this->result_rr_counter = !this->result_rr_counter;
    }
}

void MXCore::handle_response(unsigned slot_id, int64_t response_cycle) {
    ReqSlot &slot = this->req_slots[slot_id];
    if (slot.req->status == vp::IO_REQ_INVALID) {
        this->trace.fatal("MXCore TCDM request failed\n");
    }
    slot.state = REQ_RETURNED;
    // Physical r_valid time. process_responses() clocks the independent HCI
    // response FIFO before the data can reach the downstream input FIFO.
    slot.available_at = response_cycle;
    this->trace_responses();
    this->check_state();
}

void MXCore::tcdm_response(vp::Block *__this, vp::IoReq *req) {
    MXCore  *_this = static_cast<MXCore *>(__this);
    unsigned slot_id;
    std::memcpy(&slot_id, req->arg_get(0), sizeof(slot_id));
    // Callback time already includes elapsed delay. Latency/duration are
    // relative to issue; do not add them to callback time a second time.
    const ReqSlot &slot = _this->req_slots[slot_id];
    const int64_t  response_cycle =
        std::max(_this->clock.get_cycles(), slot.issued_at + int64_t(req->get_full_latency()));
    _this->handle_response(slot_id, response_cycle);
}

void MXCore::tcdm_grant(vp::Block *__this, vp::IoReq *req) {
    MXCore *_this = static_cast<MXCore *>(__this);
    if (_this->denied_req_slot >= 0 && _this->req_slots[_this->denied_req_slot].req == req) {
        _this->accept_request(unsigned(_this->denied_req_slot));
        if (_this->req_slots[_this->denied_req_slot].epoch == _this->epoch) {
            trace_pulse(_this->wave_gnt);
            uint32_t zero = 0;
            _this->wave_req.dump_next(reinterpret_cast<uint8_t *>(&zero), 1);
        }
        _this->denied_req_slot = -1;
        // A delayed grant occupies this cycle's wide port, even if the FSM
        // executes later in the same cycle.
        _this->next_bus_cycle = std::max(_this->next_bus_cycle, _this->clock.get_cycles() + 1);
    }
    _this->check_state(); // Ownership stays with memory; a grant never resubmits the request.
}

int MXCore::arbitrate_request() {
    const auto now = this->clock.get_cycles();
    if (this->denied_req_slot >= 0 || now < this->next_bus_cycle) {
        return -1;
    }

    for (unsigned stream : {STREAM_C, STREAM_A, STREAM_B, STREAM_SA, STREAM_SB, STREAM_Z}) {
        const unsigned channel = hci_channel(stream);
        const auto &fifo = this->hci_fifos[channel];
        if (fifo.requests.empty()) {
            continue;
        }
        const unsigned index = fifo.requests.front();
        const ReqSlot &slot = this->req_slots[index];
        if (slot.state == REQ_QUEUED && slot.available_at <= now &&
            fifo.can_issue(this->response_arrival(channel) >= 0,
                           this->response_consumer_ready(channel))) {
            return index;
        }
    }
    return -1;
}

void MXCore::issue_request(unsigned slot_id) {
    const auto now = this->clock.get_cycles();
    ReqSlot   &slot = this->req_slots[slot_id];
    const auto &requests = this->hci_fifos[hci_channel(slot.stream)].requests;
    if (requests.empty() || requests.front() != slot_id) {
        throw std::logic_error("MXCore HCI issue order mismatch");
    }
    slot.state = REQ_PENDING;
    slot.issued_at = now;
    ++this->fsm_counter;
    slot.available_at = INT64_MAX;
    uint32_t one = 1;
    this->wave_addr.set(uint32_t(slot.req->get_addr()));
    this->wave_stream.set(slot.stream);
    this->wave_req.dump(reinterpret_cast<uint8_t *>(&one));
    const auto response = this->tcdm_itf.req(slot.req);
    this->next_bus_cycle = now + std::max(uint64_t(1), slot.req->get_duration());
    if (response == vp::IO_REQ_OK || response == vp::IO_REQ_PENDING) {
        // Both statuses accept the request immediately; DENIED waits for grant().
        this->accept_request(slot_id);
        trace_pulse(this->wave_gnt);
        uint32_t zero = 0;
        this->wave_req.dump_next(reinterpret_cast<uint8_t *>(&zero), 1);
    }
    if (response == vp::IO_REQ_OK) {
        this->handle_response(slot_id, now + int64_t(slot.req->get_full_latency()));
    } else if (response == vp::IO_REQ_DENIED) {
        this->denied_req_slot = slot_id;
    } else if (response != vp::IO_REQ_PENDING) {
        this->trace.fatal("MXCore invalid TCDM response\n");
    }
}

void MXCore::accept_request(unsigned slot_id) {
    const ReqSlot &slot = this->req_slots[slot_id];
    if (slot.epoch == this->epoch) {
        this->hci_fifos[hci_channel(slot.stream)].pop_request(slot_id, this->clock.get_cycles());
    }
}

void MXCore::process_responses() {
    const auto now = this->clock.get_cycles();
    for (unsigned i = 0; i < this->req_slots.size(); ++i) {
        const ReqSlot &slot = this->req_slots[i];
        if (slot.state == REQ_RETURNED &&
            ((slot.epoch != this->epoch && slot.available_at <= now) ||
             (slot.req->get_is_write() && slot.available_at + 1 <= now))) {
            this->free_req_slot(i);
        }
    }

    for (unsigned channel = 0; channel < NB_TCDM_CHANNELS; ++channel) {
        const int arrival = this->response_arrival(channel);
        const int popped = this->hci_fifos[channel].clock_response(
            arrival, this->response_consumer_ready(channel));
        if (arrival >= 0) {
            this->req_slots[arrival].state = REQ_BUFFERED;
        }
        if (popped < 0) {
            continue;
        }
        ReqSlot &slot = this->req_slots[popped];
        const unsigned stream = slot.stream;
        if (stream < NB_INPUT_STREAMS) {
            this->process_iter_instruction(slot.buf, stream);
        } else if (stream == STREAM_C) {
            this->preload_buffer.emplace_back(slot.buf, slot.buf + this->bandwidth);
        }

        this->free_req_slot(unsigned(popped));
    }
}

/************************************************************
 *                   Compute Engine and FSM                   *
 ************************************************************/

void MXCore::trace_pulse(vp::Event &event, uint32_t value) {
    if (!value) return;
    event.dump(reinterpret_cast<uint8_t *>(&value));
    uint32_t zero = 0;
    // The trace engine clears the pulse even after the last FSM callback.
    event.dump_next(reinterpret_cast<uint8_t *>(&zero), 1);
}

void MXCore::trace_compute_state() {
    uint32_t valid = 0;
    for (unsigned stage = 0; stage < this->dotp_result_valid.size(); ++stage) {
        valid |= uint32_t(this->dotp_result_valid[stage]) << stage;
    }
    // Post-edge state: bit 0 is the first pipeline stage, bit 3 the last.
    this->wave_pipe_valid.set(valid);
    this->wave_preload_wait.set(this->preload_wait);
    this->wave_preload_count.set(this->preload_count);
}

void MXCore::trace_responses() {
    const auto now = this->clock.get_cycles();
    for (auto &slot : this->req_slots) {
        if (slot.epoch == this->epoch && slot.state == REQ_RETURNED && !slot.req->get_is_write() &&
            !slot.response_traced && slot.available_at <= now) {
            // Observe the bus response, before the extra HCI buffer cycle.
            trace_pulse(this->wave_rsp);
            slot.response_traced = true;
        }
    }
}

void MXCore::push_result_row(const std::vector<uint32_t> &last) {
    Bytes values, scales;

    if (this->quantize_mx) {
        for (unsigned group = 0; group < this->npe; group += 32) {
            std::array<uint32_t, 32> block;
            std::copy_n(last.begin() + group, 32, block.begin());
            auto q = mxcore::quantize_block(block, this->output_format, this->block_poison);
            values.insert(values.end(), q.data.begin(), q.data.end());
            scales.push_back(q.scale);
        }
        this->output_push(1, std::move(scales));
    } else if (this->output_format == OutputFormat::BF16) {
        for (uint32_t bits : last) {
            const uint16_t value = mxcore::quantize_bf16(bits);
            values.push_back(uint8_t(value));
            values.push_back(uint8_t(value >> 8));
        }
    } else {
        for (auto bits : last) {
            append32(values, bits);
        }
    }

    this->output_push(0, std::move(values));
    ++this->final_rows;
    trace_pulse(this->wave_result_fire);
}

void MXCore::process_iter_instruction() {
    this->push_result_row(this->partial_sum_buffer[this->result_read_row]);
    this->partial_sum_valid[this->result_read_row] = false;
    this->partial_sum_final[this->result_read_row] = false;
    this->result_read_row = (this->result_read_row + 1) % this->reuse;
}

unsigned MXCore::process_iter_instruction(uint8_t *buf, unsigned instr) {
    switch (instr) {
    case STREAM_A:
    case STREAM_B:
    case STREAM_SA:
    case STREAM_SB: {
        auto     &fifo = this->get_input_buffer(instr);
        unsigned &lane = this->input_push_lane[instr];
        if (!this->input_can_push(instr)) {
            throw std::logic_error("input FIFO overflow");
        }
        fifo[lane].emplace_back(buf, buf + this->bandwidth);
        lane = (lane + 1) % fifo.size();
        return this->bandwidth;
    }
    case STREAM_Z:
    case STREAM_SZ: {
        const unsigned output = instr == STREAM_Z ? 0 : 1;
        const unsigned size = this->output_peek(output, buf);
        if (!size) {
            throw std::logic_error("empty MXCore store instruction");
        }
        this->output_pop(output);
        return size;
    }
    default:
        throw std::logic_error("invalid MXCore stream instruction");
    }
}

void MXCore::retire_result(std::vector<uint32_t> result, uint8_t flags) {
    const unsigned row = this->partial_write_row;
    this->partial_sum_buffer[row] = std::move(result);
    this->partial_sum_valid[row] = true;
    this->partial_sum_final[row] = this->out_count + this->reuse >= this->iter_count;
    this->status |= flags;
    this->partial_write_row = (row + 1) % this->reuse;
}

void MXCore::process_preload() {
    const Bytes &beat = this->preload_buffer.front();
    auto &row = this->partial_sum_buffer[this->preload_row];
    const unsigned words = this->bandwidth / 4;
    for (unsigned i = 0; i < words; ++i) {
        uint32_t value = 0;
        for (unsigned byte = 0; byte < 4; ++byte) {
            value |= uint32_t(beat[i * 4 + byte]) << (8 * byte);
        }
        row[this->preload_word * words + i] = value;
    }
    this->preload_buffer.pop_front();
    if (++this->preload_word == this->npe / words) {
        this->preload_word = 0;
        this->partial_sum_valid[this->preload_row] = true;
        this->partial_sum_final[this->preload_row] = false;
        this->preload_row = (this->preload_row + 1) % this->reuse;
        ++this->preload_count;
    }
}

void MXCore::process_standalone() {
    if (!this->standalone_input_done &&
        this->standalone_input_count == this->stream_total[STREAM_C]) {
        this->standalone_input_done = true;
        this->reg_fsm_state.set(STORING);
        if (this->events[this->offloading_core][0].is_bound()) {
            this->events[this->offloading_core][0].sync(true);
        }
        this->wave_event_bits |= 1;
        this->wave_events.dump(reinterpret_cast<uint8_t *>(&this->wave_event_bits));
        this->event_pending = true;
    }

    const unsigned input_bytes = this->standalone_input_bf16 ? 2 : 4;
    const unsigned row_bytes = this->npe * input_bytes;
    // The RTL directly connects a BF16 beat when widening it produces one row.
    const bool direct = this->standalone_input_bf16 && row_bytes == this->bandwidth;
    const bool push = !direct && !this->preload_buffer.empty() &&
                      this->standalone_buffer[this->standalone_push_lane].size() < STREAM_FIFO_DEPTH;
    const bool valid =
        direct ? !this->preload_buffer.empty()
               : std::all_of(this->standalone_buffer.begin(), this->standalone_buffer.end(),
                             [](const std::deque<Bytes> &lane) { return !lane.empty(); });
    const bool row_fire = valid && this->final_rows < this->expected_rows && this->output_can_push(0) &&
                          (!this->quantize_mx || this->output_can_push(1));

    if (row_fire) {
        std::vector<uint32_t> row(this->npe);
        for (unsigned i = 0; i < this->npe; ++i) {
            const unsigned offset = this->standalone_slice * row_bytes + i * input_bytes;
            const unsigned lane = offset / this->bandwidth;
            const Bytes &beat = direct ? this->preload_buffer.front()
                                       : this->standalone_buffer[lane].front();
            uint32_t value = 0;
            for (unsigned byte = 0; byte < input_bytes; ++byte) {
                value |= uint32_t(beat[offset % this->bandwidth + byte]) << (8 * byte);
            }
            // BF16 widening is a bit concatenation, including NaN payloads.
            row[i] = this->standalone_input_bf16 ? value << 16 : value;
        }
        this->push_result_row(row);

        if (direct) {
            this->preload_buffer.pop_front();
            ++this->standalone_input_count;
        } else if (row_bytes >= this->bandwidth) {
            for (auto &lane : this->standalone_buffer) {
                lane.pop_front();
            }
        } else if (++this->standalone_slice == this->bandwidth / row_bytes) {
            this->standalone_slice = 0;
            this->standalone_buffer.front().pop_front();
        }
    }

    // Readiness was sampled before popping, as for the RTL FIFO lane array.
    if (push) {
        this->standalone_buffer[this->standalone_push_lane].push_back(
            std::move(this->preload_buffer.front()));
        this->preload_buffer.pop_front();
        this->standalone_push_lane =
            (this->standalone_push_lane + 1) % this->standalone_buffer.size();
        ++this->standalone_input_count;
    }
    if ((push || row_fire) && !this->standalone_input_done) {
        this->reg_fsm_state.set(ROUTINE);
    }
}

void MXCore::process_compute() {
    const unsigned result_row = this->result_read_row;
    const bool     drain = this->partial_sum_valid[result_row] && this->partial_sum_final[result_row] &&
                       this->output_can_push(0) && (!this->quantize_mx || this->output_can_push(1));
    const bool write_ready = !this->partial_sum_valid[this->partial_write_row] &&
                             !(this->preload && this->preload_word &&
                               this->partial_write_row == this->preload_row);
    const bool preload_ready = this->preload_count >= this->preload_threshold;
    const bool preload_fire = this->preload && !this->preload_buffer.empty() &&
                              (this->preload_count < this->reuse + this->preload_final_count ||
                               this->preload_word) && !this->partial_sum_valid[this->preload_row];
    
    std::vector<bool> ready(this->pipe_regs);
    for (int i = int(this->pipe_regs) - 1; i >= 0; --i) {
        ready[i] =
            !this->dotp_result_valid[i] || (i + 1 == int(this->pipe_regs) ? write_ready : ready[i + 1]);
    }

    const bool     first = this->in_count < this->reuse;
    const unsigned feedback_row = this->partial_read_row;
    auto          &partial = this->partial_sum_buffer[feedback_row];
    const bool     feedback_ready =
        (first && !this->preload) ||
        (this->partial_sum_valid[feedback_row] && !this->partial_sum_final[feedback_row]);
    const bool engine_ready = this->compute_count < this->expected_bundles && feedback_ready &&
                              (!this->preload || !this->preload_wait) &&
                              (this->pipe_regs ? ready[0] : write_ready);
    
    bool all_valid = true;
    for (unsigned i = 0; i < NB_INPUT_STREAMS; ++i) {
        all_valid &= this->input_valid(i);
    }
    
    bool                  retire_valid = false;
    std::vector<uint32_t> retired_bits;
    uint8_t               retired_status = 0;
    
    if (this->pipe_regs && this->dotp_result_valid.back() && write_ready) {
        retire_valid = true;
        retired_bits = std::move(this->dotp_result_buffer.back());
        retired_status = this->dotp_result_status.back();
    }
    
    if (drain) {
        this->process_iter_instruction();
    }

    bool                  compute_valid = false;
    std::vector<uint32_t> computed_bits;
    uint8_t               computed_status = 0;
    
    if (engine_ready && all_valid) {
        // RTL vector_a/scale_a/vector_b/scale_b advance together on in_fire.
        std::array<Bytes, NB_INPUT_STREAMS> operands;
        for (unsigned i = 0; i < NB_INPUT_STREAMS; ++i) {
            operands[i] = this->input_peek(i);
            this->input_pop(i);
        }
        compute_valid = true;
        computed_bits.resize(this->npe);
        const uint32_t *d = first && !this->preload ? nullptr : partial.data();

        if (this->src_fmt == InputFormat::E2M1) {
            computed_status = matmul_fp4(computed_bits.data(), d, operands[0].data(),
                                         operands[1].data(), operands[2].data(), operands[3].data(),
                                         this->vs, this->npe, this->dotp);
        } else if (this->src_fmt == InputFormat::E4M3) {
            computed_status = matmul_fp8e4m3(computed_bits.data(), d, operands[0].data(),
                                             operands[1].data(), operands[2].data(), operands[3].data(),
                                             1, this->vs, this->vs, this->npe, this->reuse, this->dotp);
        } else {
            computed_status = matmul_fp8e5m2(computed_bits.data(), d, operands[0].data(),
                                             operands[1].data(), operands[2].data(), operands[3].data(),
                                             1, this->vs, this->vs, this->npe, this->reuse, this->dotp);
        }
        if (!first || this->preload) {
            this->partial_sum_valid[feedback_row] = false;
            this->partial_read_row = (this->partial_read_row + 1) % this->reuse;
        }
    }
    
    if (this->pipe_regs) {
        for (int i = int(this->pipe_regs) - 1; i >= 0; --i) {
            if (ready[i]) {
                if (i) {
                    this->dotp_result_buffer[i] = std::move(this->dotp_result_buffer[i - 1]);
                    this->dotp_result_status[i] = this->dotp_result_status[i - 1];
                    this->dotp_result_valid[i] = this->dotp_result_valid[i - 1];
                } else {
                    this->dotp_result_buffer[0] = std::move(computed_bits);
                    this->dotp_result_status[0] = computed_status;
                    this->dotp_result_valid[0] = compute_valid;
                }
            }
        }
    } else if (compute_valid) {
        retire_valid = true;
        retired_bits = std::move(computed_bits);
        retired_status = computed_status;
    }
    
    if (retire_valid) {
        this->retire_result(std::move(retired_bits), retired_status);
        if (++this->retired_count == this->expected_bundles) {
            //event 0 reports compute completion; event 1 reports completed output stores.
            if (this->events[this->offloading_core][0].is_bound()) {
                this->events[this->offloading_core][0].sync(true);
            }
            this->wave_event_bits |= 1;
            this->wave_events.dump(reinterpret_cast<uint8_t *>(&this->wave_event_bits));
            this->event_pending = true;
        }
    }

    if (preload_fire) {
        this->process_preload();
    }

    if (this->preload && preload_ready) {
        this->preload_wait = false;
    }

    if (this->preload && retire_valid) {
        if (this->out_count + this->reuse >= this->iter_count) {
            ++this->preload_final_count;
        }
        if (this->out_count == this->iter_count - 1) {
            this->preload_count -= this->reuse;
            this->preload_final_count = 0;
            this->preload_wait = this->retired_count != this->expected_bundles;
        }
    }

    const bool last_iteration = this->next_iteration(compute_valid, retire_valid);
    if (compute_valid) {
        this->reg_fsm_state.set(last_iteration ? STORING : ROUTINE);
    }
    // bit 0: MXDOTP input accepted; bit 1: result retired into partial sums.
    trace_pulse(this->wave_dotp_fire, uint32_t(compute_valid) | (uint32_t(retire_valid) << 1));
    this->trace_compute_state();
}

bool MXCore::next_iteration(bool count_in, bool count_out) {
    if (count_in) {
        this->in_count = (this->in_count + 1) % this->iter_count;
        ++this->compute_count;
    }
    
    if (count_out) {
        this->out_count = (this->out_count + 1) % this->iter_count;
    }
    
    return this->compute_count == this->expected_bundles;
}

bool MXCore::finished() const {
    return this->final_rows == this->expected_rows && this->fsm_counter == this->tcdm_block_total &&
           this->all_slots_free() &&
           std::none_of(this->dotp_result_valid.begin(), this->dotp_result_valid.end(),
                        [](bool v) { return v; }) &&
           std::none_of(this->partial_sum_valid.begin(), this->partial_sum_valid.end(),
                        [](bool v) { return v; }) &&
           this->result_buffer.empty() && this->result_scale_buffer.empty() &&
           this->preload_buffer.empty() && !this->preload_word &&
           std::all_of(this->hci_fifos.begin(), this->hci_fifos.end(),
                       [](const mxcore::HciFifo &fifo) { return fifo.empty(); }) &&
           (!this->standalone ||
            (this->standalone_input_done && !this->standalone_slice &&
             std::all_of(this->standalone_buffer.begin(), this->standalone_buffer.end(),
                         [](const std::deque<Bytes> &lane) { return lane.empty(); })));
}

void MXCore::check_state() {
    if (!this->fsm_event->is_enqueued()) {
        this->event_enqueue(this->fsm_event, 1);
    }
}

void MXCore::fsm_handler(vp::Block *__this, vp::ClockEvent *) {
    MXCore *_this = static_cast<MXCore *>(__this);
    // Let response callbacks already scheduled at this edge settle before
    // sampling HCI r_valid/ready. Zero delay preserves this edge's cycle number.
    _this->event_enqueue(_this->fsm_update_event, 0);
}

void MXCore::fsm_update(vp::Block *__this, vp::ClockEvent *) {
    MXCore *_this = static_cast<MXCore *>(__this);
    // Synchronous targets can report a future response via request latency.
    _this->trace_responses();
    
    if (_this->event_pending || _this->sw_evt) {
        _this->clear_events();
    }

    if (_this->finish_pending) {
        _this->finish_pending = false;
        _this->reg_fsm_state.set(IDLE);
        _this->reg_busy.set(0);
        _this->context_status = 0;
        return;
    }

    if (_this->startup_phase != StartupPhase::NONE) {
        const auto now = _this->clock.get_cycles();
        if (now >= _this->startup_edge) {
            if (_this->startup_phase == StartupPhase::COMMITTED) {
                _this->launch_job();
                _this->startup_phase = StartupPhase::ADDRESS_FIFO;
                _this->startup_edge = now + 1;
            } else {
                // addr_valid_q was generated at launch; the address FIFO
                // captures it at this edge. Its old head is usable next edge.
                _this->startup_phase = StartupPhase::NONE;
            }
        }
        _this->check_state();
        return;
    }
    
    if (_this->reg_fsm_state.get() == IDLE) {
        _this->process_responses();
        return;
    }
    
    for (unsigned i = 0; i < NB_INPUT_STREAMS; ++i) {
        _this->input_push_ready[i] = _this->input_can_push(i);
    }
    _this->preload_push_ready = _this->preload_buffer.size() < STREAM_FIFO_DEPTH;
    
    const bool final = _this->final_rows == _this->expected_rows;
    const bool z = !_this->stream_done(STREAM_Z) && _this->output_valid(0, final);
    const bool sz = !_this->stream_done(STREAM_SZ) && _this->output_valid(1, final);
    
    // Standalone selects the quantizer input without running i_engine.
    if (_this->standalone) {
        _this->process_standalone();
    } else {
        _this->process_compute();
    }
    
    _this->enqueue_read_requests();
    _this->enqueue_preload_request();
    _this->enqueue_write_request(z, sz);
    const int slot_id = _this->arbitrate_request();
    
    if (slot_id >= 0) {
        _this->issue_request(slot_id);
    }
    
    _this->process_responses();
    
    if (_this->finished()) {
        _this->finish_job();
    }
    
    if (_this->reg_fsm_state.get() != IDLE) {
        _this->check_state();
    }
}

/************************************************************
 *                     Address Generation                     *
 ************************************************************/

uint32_t MXCore::calculate_tile_base_address(uint32_t base, uint32_t tile_bytes, unsigned tiles_per_row,
                                             unsigned i, unsigned j) const {
    return base + (uint64_t(i) * tiles_per_row + j) * tile_bytes;
}

uint32_t MXCore::inc_addr(uint32_t addr, uint32_t stride) const {
    return addr + stride;
}

uint32_t MXCore::next_addr(unsigned stream) {
    const uint32_t count = this->stream_issued[stream] + 1;
    const bool     more = count < this->stream_total[stream];
    uint32_t       addr = 0;
    
    switch (stream) {
    case STREAM_A: {
        addr = this->iter_a_addr;
        if (more) {
            this->iter_a_addr = this->inc_addr(addr, this->bandwidth);
            const unsigned row_beats = this->a_row_tile_bytes / this->bandwidth;
            if (count % row_beats == 0) {
                // A_D1_STRIDE=0: replay a row tile for each B column tile.
                const unsigned row_tile = count / row_beats / this->b_col_tiles;
                this->iter_a_addr = this->calculate_tile_base_address(this->a_addr, this->a_tile_size / 8,
                                                                      this->inner_tiles, row_tile, 0);
            }
        }
        break;
    }
    
    case STREAM_B:
        addr = this->iter_b_addr;
        
        if (more) {
            this->iter_b_addr = count % (this->b_matrix_bytes / this->bandwidth) == 0
                                    ? this->b_addr
                                    : this->inc_addr(addr, this->bandwidth);
        }
        break;

    case STREAM_SA: {
        addr = this->iter_scale_a_addr;
        if (more) {
            this->iter_scale_a_addr = this->inc_addr(addr, this->bandwidth);
            if (this->reuse > this->bandwidth) {
                const unsigned tile_beats = this->reuse / this->bandwidth;
                if (count % tile_beats == 0) {
                    // Large Scale A: the programmed replay count is clamped for short K.
                    const unsigned block = count / tile_beats /
                                             this->sa_vec_per_block % this->inner_blocks;
                    this->iter_scale_a_addr = this->calculate_tile_base_address(
                        this->scale_a_addr, this->reuse, this->inner_blocks, 0, block);
                }
            } else {
                const unsigned row_beats =
                    (this->scale_a_row_tile_bytes + this->bandwidth - 1) / this->bandwidth;
                if (count % row_beats == 0) {
                    const unsigned row_tile = count / row_beats / this->b_col_tiles;
                    this->iter_scale_a_addr = this->calculate_tile_base_address(
                        this->scale_a_addr, this->reuse, this->inner_blocks, row_tile, 0);
                }
            }
        }
        break;
    }

    case STREAM_SB:
        addr = this->iter_scale_b_addr;
        if (more) {
            this->iter_scale_b_addr = count % (this->scale_b_matrix_bytes / this->bandwidth) == 0
                                          ? this->scale_b_addr
                                          : this->inc_addr(addr, this->bandwidth);
        }
        break;

    case STREAM_Z:
        addr = this->iter_z_addr;
        if (more)
            this->iter_z_addr = this->inc_addr(addr, this->bandwidth);
        break;
    case STREAM_SZ:
        addr = this->iter_scale_z_addr;
        if (more)
            this->iter_scale_z_addr = this->inc_addr(addr, this->bandwidth);
        break;
    case STREAM_C:
        addr = this->iter_preload_addr;
        if (more)
            this->iter_preload_addr = this->inc_addr(addr, this->bandwidth);
        break;
    default:
        throw std::logic_error("invalid MXCore stream instruction");
    }

    ++this->stream_issued[stream];
    return addr;
}

bool MXCore::stream_done(unsigned stream) const {
    return this->stream_issued[stream] == this->stream_total[stream];
}

/************************************************************
 *                       Input Buffers                        *
 ************************************************************/

void MXCore::init_input_buffers() {
    this->input_width = {this->vs, this->npe * this->vs, 1, this->npe};
    this->input_push_lane.fill(0);
    this->input_slice.fill(0);
    this->input_row.fill(0);
    this->input_scale_repeat.fill(0);
    this->input_block.fill(0);
    
    for (unsigned stream = 0; stream < NB_INPUT_STREAMS; ++stream) {
        const unsigned lanes = std::max(1u, this->input_width[stream] / this->bandwidth);
        this->get_input_buffer(stream).assign(lanes, {});
    }
    
    this->scale_b_short =
        uint64_t(this->npe) * this->inner_blocks * this->b_col_tiles < this->bandwidth;
}

FifoLanes &MXCore::get_input_buffer(unsigned stream) {
    switch (stream) {
        case STREAM_A:
            return this->a_buffer;
        case STREAM_B:
            return this->b_buffer;
        case STREAM_SA:
            return this->scale_a_buffer;
        case STREAM_SB:
            return this->scale_b_buffer;
        default:
            throw std::logic_error("invalid MXCore input stream");
    }
}

const FifoLanes &MXCore::get_input_buffer(unsigned stream) const {
    switch (stream) {
        case STREAM_A:
            return this->a_buffer;
        case STREAM_B:
            return this->b_buffer;
        case STREAM_SA:
            return this->scale_a_buffer;
        case STREAM_SB:
            return this->scale_b_buffer;
        default:
            throw std::logic_error("invalid MXCore input stream");
    }
}

bool MXCore::input_can_push(unsigned stream) const {
    const auto     &fifo = this->get_input_buffer(stream);
    const unsigned &push_lane = this->input_push_lane[stream];

    return fifo[push_lane].size() < STREAM_FIFO_DEPTH;
}

bool MXCore::input_valid(unsigned stream) const {
    const auto &fifo = this->get_input_buffer(stream);

    return std::all_of(fifo.begin(), fifo.end(), [](const auto &q) { return !q.empty(); });
}

Bytes MXCore::input_peek(unsigned stream) const {
    const unsigned  beat = this->bandwidth;
    const unsigned  output = this->input_width[stream];
    const unsigned  reuse = this->reuse;
    const auto     &fifo = this->get_input_buffer(stream);
    const unsigned &slice = this->input_slice[stream];
    const unsigned &row = this->input_row[stream];
    const unsigned &block = this->input_block[stream];

    Bytes result;
    if (stream == STREAM_SA) {
        // i_scale_a_buffer
        const unsigned index = reuse > beat ? row % beat : block * reuse + row;
        return Bytes{fifo[0].front().at(index)};
    }
    
    if (output < beat) {
        const auto &front = fifo[0].front();
        result.assign(front.begin() + slice * output, front.begin() + (slice + 1) * output);
    } else {
        for (const auto &q : fifo) {
            result.insert(result.end(), q.front().begin(), q.front().end());
        }
    }
    
    return result;
}

void MXCore::input_pop(unsigned stream) {
    const unsigned beat = this->bandwidth;
    const unsigned output = this->input_width[stream];
    const unsigned reuse = this->reuse;
    const unsigned factor = this->input_block_size / this->elements_per_dotp;
    const bool     short_b = this->scale_b_short;
    auto          &fifo = this->get_input_buffer(stream);
    unsigned      &slice = this->input_slice[stream];
    unsigned      &row = this->input_row[stream];
    unsigned      &scale_repeat = this->input_scale_repeat[stream];
    unsigned      &block = this->input_block[stream];

    if (stream == STREAM_SA) {
        const unsigned scale_count = reuse > beat ? 1 : factor;
        if (++row == reuse) {
            row = 0;
            if (++scale_repeat == scale_count) {
                scale_repeat = 0;
                if (beat > reuse) {
                    if (++block == beat / reuse) {
                        block = 0;
                        fifo[0].pop_front();
                    }
                } else {
                    fifo[0].pop_front();
                }
            }
        } else if (reuse > beat && row % beat == 0) {
            fifo[0].pop_front();
        }
        return;
    }
    
    const unsigned repetitions = stream == STREAM_A ? 1 : reuse;
    if (++row != repetitions) {
        return;
    }
    
    row = 0;
    const unsigned scales = stream == STREAM_SB ? factor : 1;
    if (++scale_repeat != scales) {
        return;
    }
    
    scale_repeat = 0;
    if (output < beat && !(stream == STREAM_SB && short_b)) {
        if (++slice != beat / output) {
            return;
        }
        slice = 0;
    }
    
    for (auto &q : fifo) {
        q.pop_front();
    }
}

/************************************************************
 *                       Output Buffers                       *
 ************************************************************/

void MXCore::init_output_buffers() {
    this->result_buffer.clear();
    this->result_scale_buffer.clear();
    this->output_row_bytes = {this->npe * this->result_bits / 8, this->npe / 32};
    
    for (unsigned output = 0; output < NB_OUTPUT_STREAMS; ++output) {
        this->output_lanes[output] = std::max(1u, this->bandwidth / this->output_row_bytes[output]);
    }
    this->output_slice.fill(0);
}

std::deque<Bytes> &MXCore::get_output_buffer(unsigned output) {
    return output == 0 ? this->result_buffer : this->result_scale_buffer;
}

const std::deque<Bytes> &MXCore::get_output_buffer(unsigned output) const {
    return output == 0 ? this->result_buffer : this->result_scale_buffer;
}

bool MXCore::output_can_push(unsigned output) const {
    const unsigned lanes = this->output_lanes[output];
    const auto    &rows = this->get_output_buffer(output);

    return rows.size() < STREAM_FIFO_DEPTH * lanes;
}

void MXCore::output_push(unsigned output, Bytes row) {
    const unsigned row_bytes = this->output_row_bytes[output];
    auto          &rows = this->get_output_buffer(output);

    if (!this->output_can_push(output) || row.size() != row_bytes) {
        throw std::logic_error("result FIFO overflow/row size");
    }
    rows.push_back(std::move(row));
}

bool MXCore::output_valid(unsigned output, bool final) const {
    const unsigned beat = this->bandwidth;
    const unsigned row_bytes = this->output_row_bytes[output];
    const unsigned lanes = this->output_lanes[output];
    const auto    &rows = this->get_output_buffer(output);
    const bool     scales = output == 1;

    return row_bytes >= beat ? !rows.empty() : rows.size() >= lanes || (scales && final && !rows.empty());
}

unsigned MXCore::output_peek(unsigned output, uint8_t *data) const {
    const unsigned  beat = this->bandwidth;
    const unsigned  row_bytes = this->output_row_bytes[output];
    const unsigned  lanes = this->output_lanes[output];
    const unsigned &slice = this->output_slice[output];
    const auto     &rows = this->get_output_buffer(output);

    if (row_bytes >= beat) {
        std::memcpy(data, rows.front().data() + slice * beat, beat);
        return beat;
    }
    
    const unsigned count = std::min<std::size_t>(lanes, rows.size());
    for (unsigned i = 0; i < count; ++i) {
        std::memcpy(data + i * row_bytes, rows[i].data(), row_bytes);
    }
    return count * row_bytes;
}

void MXCore::output_pop(unsigned output) {
    const unsigned beat = this->bandwidth;
    const unsigned row_bytes = this->output_row_bytes[output];
    const unsigned lanes = this->output_lanes[output];
    unsigned      &slice = this->output_slice[output];
    auto          &rows = this->get_output_buffer(output);

    if (row_bytes > beat) {
        if (++slice != row_bytes / beat) {
            return;
        }
        slice = 0;
    }
    
    for (unsigned i = 0, n = std::min<std::size_t>(lanes, rows.size()); i < n; ++i) {
        rows.pop_front();
    }
}

void MXCore::append32(Bytes &bytes, uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) {
        bytes.push_back(uint8_t(value >> (8 * i)));
    }
}

/************************************************************
 *               MXFP8 / MXFP4 Matrix Multiplication           *
 ************************************************************/

struct MxcoreFp32Result {
    uint32_t bits;
    uint8_t  status;
};

static MxcoreFp32Result round_fp32(const MXDotp::Result &r, const MXDotp::Parameters &p) {
    // i_special_assemble
    if (r.special_raw & 8) {
        return {(r.special_raw & 2) ? 0x7f800000u | ((r.special_raw & 1u) << 31) : 0x7fc00000u,
                uint8_t((r.special_raw & 4) ? FpStatus::NV : 0)};
    }
    
    if (r.result_is_accumulator) {
        return {r.d_bits, uint8_t(r.sum_product_is_zero ? 0 : FpStatus::NX)};
    }
    
    using U = unsigned __int128;
    constexpr unsigned precision = MXDotp::DotpFp32PrecisionBits;
    const unsigned     width = p.extended_sum_width;
    const U            mask = (U(1) << width) - 1;
    
    // i_twos_compl
    const bool sign = r.sum_product_accumulator_extended < 0;
    U          magnitude = U(r.sum_product_accumulator_extended) & mask;
    if (sign) {
        magnitude = ((~magnitude) + !r.accumulator_sticky) & mask;
    }
    
    // i_norm_lzc
    unsigned leading = 0;
    if (magnitude) {
        for (U bit = U(1) << (width - 1); !(magnitude & bit); bit >>= 1) {
            ++leading;
        }
    }
    
    // i_norm_finalize
    const int      tentative = r.exponent_major - int(leading);
    const bool     normal = tentative > 0 && magnitude;
    const unsigned shift = normal ? leading + 1 : unsigned(r.exponent_major) & 511u;
    const U        window = shift >= width ? 0 : (magnitude << shift) & mask;
    const unsigned tail = width - precision;
    const uint32_t mantissa = uint32_t(window >> tail);
    
    // i_rounder
    const bool overflow = normal && tentative >= 255;
    uint32_t   absolute = overflow ? 0x7f7fffffu : (normal ? unsigned(tentative) << 23 : 0) | (mantissa >> 1);
    const bool guard = (mantissa & 1) || overflow;
    const bool sticky = (window & ((U(1) << tail) - 1)) || r.accumulator_sticky || overflow;
    const bool inexact = guard || sticky;
    absolute += guard && (sticky || (absolute & 1));
    const bool overflow_after = (absolute >> 23) == 255;
    
    return {(uint32_t(sign) << 31) | absolute,
            uint8_t((inexact || overflow_after ? FpStatus::NX : 0) |
                    (overflow || overflow_after ? FpStatus::OF : 0) |
                    ((absolute >> 23) == 0 && inexact ? FpStatus::UF : 0))};
}

uint8_t matmul_fp4(uint32_t *z, const uint32_t *d, const uint8_t *a, const uint8_t *b,
                   const uint8_t *scale_a, const uint8_t *scale_b,
                   unsigned vs, unsigned npe, const MXDotp &dotp) {
    uint8_t status = 0;
    for (unsigned col = 0; col < npe; ++col) {
        auto input = dotp.make_inputs();
        std::copy_n(a, vs, input.a.begin());
        std::copy_n(b + col * vs, vs, input.b.begin());
        input.scale_a = scale_a[0];
        input.scale_b = scale_b[col];
        input.d_bits = d ? d[col] : 0;
        const auto result = round_fp32(dotp.compute(input, InputFormat::E2M1), dotp.parameters());
        z[col] = result.bits;
        status |= result.status;
    }
    return status;
}

uint8_t matmul_fp8e4m3(uint32_t *z, const uint32_t *d, const fp8e4m3 *a, const fp8e4m3 *b,
                       const uint8_t *scale_a, const uint8_t *scale_b, unsigned m_size, unsigned k_size,
                       unsigned vs, unsigned npe, unsigned reuse, const MXDotp &dotp) {
    if (!reuse || m_size > reuse || !vs || vs != dotp.parameters().vector_size || k_size % vs) {
        throw std::invalid_argument(
            "FP8 matmul requires rows <= Reuse, matching VS and K divisible by VS");
    }
    
    uint8_t status = 0;
    for (unsigned row = 0; row < m_size; ++row) {
        for (unsigned col = 0; col < npe; ++col) {
            uint32_t accumulator = d ? d[row * npe + col] : 0;
            for (unsigned inner = 0; inner < k_size; inner += vs) {
                auto input = dotp.make_inputs();
                // Match the RTL operand/scale swap: B per PE, broadcast A.
                std::copy_n(b + ((inner / vs) * npe + col) * vs, vs, input.a.begin());
                std::copy_n(a + ((inner / vs) * m_size + row) * vs, vs, input.b.begin());
                input.scale_a = scale_b[(inner / 32) * npe + col];
                input.scale_b = scale_a[(inner / 32) * m_size + row];
                input.d_bits = accumulator;
                // i_fpnew_mxdotp_multi
                const auto result =
                    round_fp32(dotp.compute(input, InputFormat::E4M3), dotp.parameters());
                accumulator = result.bits;
                status |= result.status;
            }
            z[row * npe + col] = accumulator;
        }
    }
    return status;
}

uint8_t matmul_fp8e5m2(uint32_t *z, const uint32_t *d, const fp8e5m2 *a, const fp8e5m2 *b,
                       const uint8_t *scale_a, const uint8_t *scale_b, unsigned m_size, unsigned k_size,
                       unsigned vs, unsigned npe, unsigned reuse, const MXDotp &dotp) {
    if (!reuse || m_size > reuse || !vs || vs != dotp.parameters().vector_size || k_size % vs) {
        throw std::invalid_argument(
            "FP8 matmul requires rows <= Reuse, matching VS and K divisible by VS");
    }
    
    uint8_t status = 0;
    for (unsigned row = 0; row < m_size; ++row) {
        for (unsigned col = 0; col < npe; ++col) {
            uint32_t accumulator = d ? d[row * npe + col] : 0;
            for (unsigned inner = 0; inner < k_size; inner += vs) {
                auto input = dotp.make_inputs();
                // Match the RTL operand/scale swap: B per PE, broadcast A.
                std::copy_n(b + ((inner / vs) * npe + col) * vs, vs, input.a.begin());
                std::copy_n(a + ((inner / vs) * m_size + row) * vs, vs, input.b.begin());
                input.scale_a = scale_b[(inner / 32) * npe + col];
                input.scale_b = scale_a[(inner / 32) * m_size + row];
                input.d_bits = accumulator;
                // i_fpnew_mxdotp_multi
                const auto result =
                    round_fp32(dotp.compute(input, InputFormat::E5M2), dotp.parameters());
                accumulator = result.bits;
                status |= result.status;
            }
            z[row * npe + col] = accumulator;
        }
    }
    return status;
}

/************************************************************
 *                    Configuration Check                     *
 ************************************************************/

bool MXCore::power_of_two(unsigned x) {
    return x && !(x & (x - 1));
}
