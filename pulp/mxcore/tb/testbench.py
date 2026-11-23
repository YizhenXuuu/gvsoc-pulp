from pathlib import Path
import math
import gvsoc.systree
from gvrun.parameter import TargetParameter


class Testbench(gvsoc.systree.Component):

    def __init__(self, parent, name, m, k, n, vs, reuse, npe):
        super().__init__(parent, name)

        self.add_sources(['pulp/mxcore/tb/testbench.cpp'])
        memory_file = TargetParameter(self, name='memory_file', value='input.hex',
                                      description='RTL $readmemh input: 32-bit hex words').get_value()
        golden_file = TargetParameter(self, name='golden_file', value='',
                                      description='PyGolden result hex; defaults to tb/golden/result_<input suffix>.txt').get_value()
        input_name = Path(memory_file).name
        golden_name = ('result_' + input_name[len('data_memory_'):]
                       if input_name.startswith('data_memory_') else f'result_{Path(input_name).stem}.txt')
        golden_file = (Path(golden_file).expanduser().resolve() if golden_file
                       else Path(__file__).resolve().parent / 'golden' / golden_name)
        input_format = TargetParameter(self, name='input', value='fp8',
                                       description='MXCore input format').get_value()    
        output_format = TargetParameter(self, name='output', value='fp8',
                                        description='MXCore output format').get_value()
        preload = TargetParameter(self, name='preload', value=1,
                                  description='Preload FP32 accumulator from the C region').get_value()
        standalone = TargetParameter(self, name='standalone', value=0,
                                     description='Quantize C without matrix multiplication').get_value()
        standalone_input = TargetParameter(self, name='standalone_input', value='fp32',
                                           description='Standalone C format: fp32 or bf16').get_value().lower()
        no_stalls = TargetParameter(self, name='no_stalls', value=1,
                                   description='Disable TCDM bank stalls, matching RTL NO_STALLS=1').get_value()
        stall_probability = float(TargetParameter(self, name='stall_probability', value=0.1,
                                  description='Independent stall probability for each 32-bit TCDM port').get_value())
        stall_seed = TargetParameter(self, name='stall_seed', value=1,
                                     description='Reproducible GVSOC bank-stall seed').get_value()
        grant_file = TargetParameter(self, name='grant_file', value='',
                                     description='Optional replay: decimal clock cycle, hex 16-bit grant mask; missing cycles grant all').get_value()
        
        if preload not in (0, 1):
            raise ValueError('MXCore preload must be 0 or 1')
        if standalone not in (0, 1):
            raise ValueError('MXCore standalone must be 0 or 1')
        if standalone_input not in ('fp32', 'bf16'):
            raise ValueError('MXCore standalone_input must be fp32 or bf16')
        if no_stalls not in (0, 1):
            raise ValueError('MXCore no_stalls must be 0 or 1')
        if not math.isfinite(stall_probability) or not 0 <= stall_probability <= 1:
            raise ValueError('MXCore stall_probability must be in [0,1]')
        if isinstance(stall_seed, bool) or not isinstance(stall_seed, int) or not 0 <= stall_seed <= 0x7fffffff:
            raise ValueError('MXCore stall_seed must be an integer in [0,2147483647]')
        
        output_file = TargetParameter(self, name='output_file',
                                      value=str(Path(__file__).resolve().parent / 'output_tb/output.bin'),
                                      description='MXCore output binary').get_value()
        
        self.add_properties(self.get_format_config(input_format, output_format))
        self.add_properties(dict(input_file=memory_file, input_format=input_format,
                                 output_format=output_format, output_file=output_file,
                                 golden_file=str(golden_file), preload=preload,
                                 standalone=standalone, standalone_input=standalone_input,
                                 standalone_input_bits=16 if standalone_input == 'bf16' else 32,
                                 no_stalls=no_stalls, stall_probability=stall_probability, stall_seed=stall_seed,
                                 grant_file=str(Path(grant_file).resolve()) if grant_file else '',
                                 m=m, k=k, n=n, vector_size=vs, reuse=reuse, npe=npe))

    @staticmethod
    def get_format_config(input_format: str, output_format: str) -> dict:
        inputs = {
            'fp8': (3 << 9, 8, 32),
            'e4m3': (5 << 9, 8, 32),
            'fp4': (8 << 9, 4, 64),
        }

        outputs = {
            'fp32': (0, 32),
            'bf16': (1 << 24, 16),
            'fp8': (1 << 23, 8),
            'e4m3': (1 << 22, 8),
            'fp4': (1 << 21, 4),
        }

        if output_format not in outputs:
            raise ValueError(f'MXCore unsupported output format: {output_format}')
        
        input_control, input_bits, input_block_size = inputs.get(input_format, (15 << 9, 8, 32))
        output_control, result_bits = outputs[output_format]

        return dict(input_control=input_control, input_bits=input_bits, input_block_size=input_block_size,
                    output_control=output_control, result_bits=result_bits)

    def o_DUT_INPUT(self, itf: gvsoc.systree.SlaveItf):
        self.itf_bind('mmio', itf, signature='io')

    def o_DUT_RESET(self, itf: gvsoc.systree.SlaveItf):
        self.itf_bind('dut_reset', itf, signature='wire<bool>')

    def i_COMPUTE_COMPLETION(self) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, 'compute_completion', signature='wire<bool>')

    def i_MEMORY(self) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, 'memory', signature='io')

    def i_COMPLETION(self) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, 'completion', signature='wire<bool>')

    def gen_gtkw(self, tree, comp_traces):
        if tree.get_view() == 'overview':
            tree.add_trace(self, 'clk', vcd_signal='timing/clk', tag='overview')
            tree.add_trace(self, 'reset', vcd_signal='timing/reset', tag='overview')
