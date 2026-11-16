GVSOC_ROOT ?= $(abspath ../../../..)
GVSOC_WORKDIR ?= $(abspath $(CURDIR)/../build/test)
memory_file ?= input.hex
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

export PATH := $(GVSOC_WORKDIR)/install/bin:$(PATH)
export USE_GVRUN = 1
export USE_GVRUN2 = 1
export PYTHONDONTWRITEBYTECODE = 1

.PHONY: all gvsoc run clean

all: run

gvsoc:
	$(MAKE) -C $(GVSOC_ROOT) TARGETS=my_system MODULES=$(CURDIR) BUILDDIR=$(GVSOC_WORKDIR)/build INSTALLDIR=$(GVSOC_WORKDIR)/install $(OPTIONS) build

run: gvsoc
	mkdir -p "$(OUTPUT_DIR)"
	$(GVSOC_WORKDIR)/install/bin/gvrun --target-dir=$(CURDIR) --target=my_system --work-dir=$(GVSOC_WORKDIR)/work \
		--parameter="soc/m=$(m)" --parameter="soc/k=$(k)" --parameter="soc/n=$(n)" \
		--parameter="soc/vs=$(vs)" --parameter="soc/reuse=$(reuse)" --parameter="soc/npe=$(npe)" \
		--parameter="soc/testbench/memory_file=$(abspath $(memory_file))" \
		--parameter="soc/testbench/input=$(input)" \
		--parameter="soc/testbench/output=$(output)" \
		--parameter="soc/testbench/preload=$(preload)" \
		--parameter="soc/testbench/standalone=$(standalone)" \
		--parameter="soc/testbench/standalone_input=$(standalone_input)" \
		--parameter="soc/testbench/no_stalls=$(no_stalls)" \
		--parameter="soc/testbench/stall_probability=$(stall_probability)" \
		--parameter="soc/testbench/stall_seed=$(stall_seed)" \
		--parameter="soc/testbench/grant_file=$(if $(grant_file),$(abspath $(grant_file)),)" \
		--parameter="soc/testbench/output_file=$(OUTPUT_DIR)/output.bin" \
		$(if $(filter 1,$(vcd)),--vcd --event-format=vcd,) run $(runner_args)

clean:
	$(MAKE) -C $(GVSOC_ROOT) TARGETS=my_system MODULES=$(CURDIR) BUILDDIR=$(GVSOC_WORKDIR)/build INSTALLDIR=$(GVSOC_WORKDIR)/install clean
