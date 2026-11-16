from pathlib import Path
import gvsoc.systree
from gvrun.parameter import TargetParameter


class Testbench(gvsoc.systree.Component):

    def __init__(self, parent, name, m, k, n, vs, reuse, npe):
        super().__init__(parent, name)

        self.add_sources(['pulp/mxcore/tb/testbench.cpp'])
        binary = TargetParameter(self, name='binary', value='input.bin',
                                 description='MXCore input binary').get_value()        
        input_format = TargetParameter(self, name='input', value='fp8',
                                       description='MXCore input format').get_value()    
        output_format = TargetParameter(self, name='output', value='fp32',
                                        description='MXCore output format').get_value()
        preload = TargetParameter(self, name='preload', value=0,
                                  description='Preload FP32 accumulator from the C region').get_value()
        standalone = TargetParameter(self, name='standalone', value=0,
                                     description='Quantize C without matrix multiplication').get_value()
        standalone_input = TargetParameter(self, name='standalone_input', value='fp32',
                                           description='Standalone C format: fp32 or bf16').get_value().lower()
        
        if preload not in (0, 1):
            raise ValueError('MXCore preload must be 0 or 1')
        if standalone not in (0, 1):
            raise ValueError('MXCore standalone must be 0 or 1')
        if standalone_input not in ('fp32', 'bf16'):
            raise ValueError('MXCore standalone_input must be fp32 or bf16')
        
        output_file = TargetParameter(self, name='output_file',
                                      value=str(Path(__file__).resolve().parent / 'output_tb/output.bin'),
                                      description='MXCore output binary').get_value()
        
        self.add_properties(self.get_format_config(input_format, output_format))
        self.add_properties(dict(input_file=binary, input_format=input_format,
                                 output_format=output_format, output_file=output_file, preload=preload,
                                 standalone=standalone, standalone_input=standalone_input,
                                 standalone_input_bits=16 if standalone_input == 'bf16' else 32,
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

    def i_MEMORY(self) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, 'memory', signature='io')

    def i_COMPLETION(self) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, 'completion', signature='wire<bool>')
