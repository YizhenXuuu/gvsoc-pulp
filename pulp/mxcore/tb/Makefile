GVSOC_ROOT ?= $(abspath ../../../..)
GVSOC_WORKDIR ?= $(abspath $(CURDIR)/../build/test)
RTL_VCD ?= $(abspath $(GVSOC_ROOT)/../MXCore/mxcore-rtl/sim/build/wave_timing.vcd)
RTL_COMPILE_TCL ?= $(abspath $(GVSOC_ROOT)/../MXCore/mxcore-rtl/sim/build/compile.tcl)
memory_file ?= input.hex
# Empty selects tb/golden/result_<input suffix>.txt, derived from memory_file.
golden_file ?=
input ?= fp8
output ?= fp8
preload ?= 1
standalone ?= 0
standalone_input ?= fp32
no_stalls ?= 1
stall_probability ?= 0.1
stall_seed ?= 1
# Optional sparse replay: decimal absolute clock cycle, hex 16-bit grant mask.
# Bit i grants port i; omitted cycles grant all ports. no_stalls=1 overrides replay.
grant_file ?=
# Enable only the timing signals selected by the component's GTKWave overview.
vcd ?= 0
m ?= 128
k ?= 128
n ?= 128
vs ?= 32
reuse ?= 64
npe ?= 32
OUTPUT_DIR := $(CURDIR)/output_tb
RTL_REPORT_DIR ?= $(OUTPUT_DIR)/rtl
# Mirror tb/input's matrix folders inside the existing simulator work directory.
VCD_DIR ?= $(GVSOC_WORKDIR)/work/M=$(m) K=$(k) N=$(n)
VCD_FILE ?= $(VCD_DIR)/IN$(input)_OUT$(output)_VS$(vs)_NPE$(npe)_REUSE$(reuse)_preload$(preload)_standalone$(standalone)$(if $(filter 1,$(standalone)),_$(standalone_input),)_$(if $(filter 1,$(no_stalls)),no_stalls,stalls_p$(stall_probability)_seed$(stall_seed)).vcd
# GNU make's abspath splits on spaces; each input here is a single file path.
mxcore_abspath = $(if $(filter /%,$(firstword $(1))),$(1),$(CURDIR)/$(1))

export PATH := $(GVSOC_WORKDIR)/install/bin:$(PATH)
export USE_GVRUN = 1
export USE_GVRUN2 = 1
export PYTHONDONTWRITEBYTECODE = 1

.PHONY: all gvsoc run timing rtl-timing clean

all: run

gvsoc:
	$(MAKE) -C $(GVSOC_ROOT) TARGETS=my_system MODULES=$(CURDIR) BUILDDIR=$(GVSOC_WORKDIR)/build INSTALLDIR=$(GVSOC_WORKDIR)/install $(OPTIONS) build

run: gvsoc
	mkdir -p "$(OUTPUT_DIR)"
	$(if $(filter 1,$(vcd)),python3 "$(CURDIR)/report_timing.py" --vcd "$(GVSOC_WORKDIR)/work/all.vcd" --save-vcd "$(VCD_FILE)" --run,) \
		$(GVSOC_WORKDIR)/install/bin/gvrun --target-dir=$(CURDIR) --target=my_system --work-dir=$(GVSOC_WORKDIR)/work \
		--parameter="soc/m=$(m)" --parameter="soc/k=$(k)" --parameter="soc/n=$(n)" \
		--parameter="soc/vs=$(vs)" --parameter="soc/reuse=$(reuse)" --parameter="soc/npe=$(npe)" \
		--parameter="soc/testbench/memory_file=$(call mxcore_abspath,$(memory_file))" \
		--parameter="soc/testbench/golden_file=$(if $(golden_file),$(call mxcore_abspath,$(golden_file)),)" \
		--parameter="soc/testbench/input=$(input)" \
		--parameter="soc/testbench/output=$(output)" \
		--parameter="soc/testbench/preload=$(preload)" \
		--parameter="soc/testbench/standalone=$(standalone)" \
		--parameter="soc/testbench/standalone_input=$(standalone_input)" \
		--parameter="soc/testbench/no_stalls=$(no_stalls)" \
		--parameter="soc/testbench/stall_probability=$(stall_probability)" \
		--parameter="soc/testbench/stall_seed=$(stall_seed)" \
		--parameter="soc/testbench/grant_file=$(if $(grant_file),$(call mxcore_abspath,$(grant_file)),)" \
		--parameter="soc/testbench/output_file=$(OUTPUT_DIR)/output.bin" \
		$(if $(filter 1,$(vcd)),--vcd --event-format=vcd,) run $(runner_args)

timing:
	python3 "$(CURDIR)/report_timing.py" --vcd "$(VCD_FILE)"

rtl-timing:
	python3 "$(CURDIR)/report_rtl_timing.py" --vcd "$(RTL_VCD)" \
		--compile-tcl "$(RTL_COMPILE_TCL)" --report-dir "$(RTL_REPORT_DIR)"

clean:
	$(MAKE) -C $(GVSOC_ROOT) TARGETS=my_system MODULES=$(CURDIR) BUILDDIR=$(GVSOC_WORKDIR)/build INSTALLDIR=$(GVSOC_WORKDIR)/install clean
