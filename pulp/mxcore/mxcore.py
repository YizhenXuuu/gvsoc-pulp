# SPDX-License-Identifier: Apache-2.0
"""MXCore HWPE: register input, shared TCDM, stream buffers and compute.

The input port accepts peripheral IO requests at relative byte offsets 0x00-0x50.
Completion is a pulse on done_irq and event_<initiator>_0 after all stores finish.
The IO initiator is the zero-based core index; an unspecified initiator selects
core zero when num_cores=1. TCDM bandwidth is bytes per cycle;
the connected memory model supplies contention, latency and duration.
"""
import gvsoc.systree


class MXCore(gvsoc.systree.Component):
    """MXCore controller, input buffers, compute engine and result buffers.

    Parameters
    ----------
    parent: gvsoc.systree.Component
        Parent component in the system tree.
    name: str
        Instance name within the parent.
    vector_size: int
        Number of FP8 products per dot product (VS).
    npe: int
        Number of dot products issued in parallel.
    reuse: int
        Number of rows sharing B inputs and partial-sum storage.
    num_pipe_regs: int
        Dot-product latency in cycles without stalls.
    tcdm_bank_width: int
        Width of one TCDM bank in bytes.
    tcdm_bank_number: int
        Number of banks in a wide TCDM transfer.
    num_cores: int
        Number of core event destinations (the RTL default is nine).
    """

    def __init__(self, parent: gvsoc.systree.Component, name: str,
                 vector_size: int=32, npe: int=32, reuse: int=64,
                 num_pipe_regs: int=4, tcdm_bank_width: int=4,
                 tcdm_bank_number: int=16, num_cores: int=9):

        super().__init__(parent, name)

        self.add_sources([
            'pulp/mxcore/mxcore.cpp',
            'pulp/mxcore/mxdotp.cpp',
            'pulp/mxcore/fp8.cpp'
        ])
        self.add_c_flags(['-std=c++17'])

        properties = dict(vector_size=vector_size, npe=npe, reuse=reuse,
                          num_pipe_regs=num_pipe_regs, tcdm_bank_width=tcdm_bank_width,
                          tcdm_bank_number=tcdm_bank_number, num_cores=num_cores)
        
        for key, value in properties.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f'MXCore {key} must be an integer')
            
        for key, low, high in [('vector_size', 2, 32), ('npe', 32, 512),
                               ('reuse', 2, 512), ('tcdm_bank_width', 1, 512),
                               ('tcdm_bank_number', 1, 512)]:
            value = properties[key]
            if not low <= value <= high or value & (value-1):
                raise ValueError(f'MXCore {key} must be a power of two in [{low},{high}]')
        
        bandwidth = tcdm_bank_width * tcdm_bank_number
        
        if not max(4, vector_size) <= bandwidth <= 512:
            raise ValueError('MXCore TCDM bandwidth must cover one VS vector and be at most 512 bytes')
        
        if not 0 <= num_pipe_regs <= 64:
            raise ValueError('MXCore num_pipe_regs must be in [0,64]')
        
        if not 1 <= num_cores <= 16:
            raise ValueError('MXCore num_cores must be in [1,16]')
        
        self.add_properties(properties)

    def i_INPUT(self) -> gvsoc.systree.SlaveItf:
        """Return the peripheral register port (vp::IoSlave)."""
        return gvsoc.systree.SlaveItf(self, 'input', signature='io')

    def o_IRQ(self, itf: gvsoc.systree.SlaveItf):
        """Bind the aggregate completion pulse, as in LightRedmule."""
        self.itf_bind('done_irq', itf, signature='wire<bool>')

    def o_EVENT(self, itf: gvsoc.systree.SlaveItf, core: int=0, event: int=0):
        """Bind an RTL per-core event (event 0 is completion; event 1 is idle)."""
        if not 0 <= core < self.get_property('num_cores') or not 0 <= event < 2:
            raise ValueError('invalid MXCore core/event index')
        self.itf_bind('event_{}_{}'.format(core, event), itf, signature='wire<bool>')

    def o_TCDM(self, itf: gvsoc.systree.SlaveItf):
        """Bind the wide memory request port (vp::IoMaster).

        Reads and writes share this port. The connected memory system supplies
        contention, latency, duration and asynchronous responses.
        """
        self.itf_bind('tcdm', itf, signature='io')

    def gen_gtkw(self, tree, comp_traces):
        if tree.get_view() == 'overview':
            tree.add_trace(self, self.name, vcd_signal='busy', tag='overview')
