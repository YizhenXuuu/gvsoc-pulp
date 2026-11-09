import gvsoc.systree
import gvsoc.runner
import vp.clock_domain
from gvrun.parameter import TargetParameter
from pulp.mxcore.mxcore import MXCore
from testbench import Testbench


class Test(gvsoc.systree.Component):

    def __init__(self, parent, name):
        super().__init__(parent, name)

        m = TargetParameter(self, name='m', value=64, description='Matrix A rows').get_value()
        k = TargetParameter(self, name='k', value=64, description='Matrix inner dimension').get_value()
        n = TargetParameter(self, name='n', value=32, description='Matrix B columns').get_value()
        vs = TargetParameter(self, name='vs', value=32, description='MXCore hardware byte lanes').get_value()
        reuse = TargetParameter(self, name='reuse', value=64, description='MXCore row reuse').get_value()
        npe = TargetParameter(self, name='npe', value=32, description='MXCore parallel dot products').get_value()

        dut = MXCore(self, 'dut', vector_size=vs, reuse=reuse, npe=npe, num_cores=1)

        testbench = Testbench(self, 'testbench', m=m, k=k, n=n, vs=vs, reuse=reuse, npe=npe)

        testbench.o_DUT_INPUT(dut.i_INPUT())
        dut.o_TCDM(testbench.i_MEMORY())
        dut.o_EVENT(testbench.i_COMPLETION(), event=1)


class Chip(gvsoc.systree.Component):

    def __init__(self, parent, name=None):

        super().__init__(parent, name)

        clock = vp.clock_domain.Clock_domain(self, 'clock', frequency=100000000)
        soc = Test(self, 'soc')
        clock.o_CLOCK(soc.i_CLOCK())


class Target(gvsoc.runner.Target):

    description = 'MXCore standalone testbench'
    model = Chip
    name = 'mxcore'
