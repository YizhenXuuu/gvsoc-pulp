GVSOC_ROOT ?= $(abspath ../../../..)
GVSOC_WORKDIR ?= $(abspath $(CURDIR)/../build/test)
binary ?= input.bin
input ?= fp8
output ?= fp32
preload ?= 0
standalone ?= 0
standalone_input ?= fp32
no_stalls ?= 0
stall_probability ?= 0.1
stall_seed ?= 1
# Optional sparse replay: decimal absolute clock cycle, hex 16-bit grant mask.
# Bit i grants port i; omitted cycles grant all ports. no_stalls=1 overrides replay.
grant_file ?=
m ?= 64
k ?= 64
n ?= 32
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
		--parameter="soc/testbench/binary=$(abspath $(binary))" \
		--parameter="soc/testbench/input=$(input)" \
		--parameter="soc/testbench/output=$(output)" \
		--parameter="soc/testbench/preload=$(preload)" \
		--parameter="soc/testbench/standalone=$(standalone)" \
		--parameter="soc/testbench/standalone_input=$(standalone_input)" \
		--parameter="soc/testbench/no_stalls=$(no_stalls)" \
		--parameter="soc/testbench/stall_probability=$(stall_probability)" \
		--parameter="soc/testbench/stall_seed=$(stall_seed)" \
		--parameter="soc/testbench/grant_file=$(if $(grant_file),$(abspath $(grant_file)),)" \
		--parameter="soc/testbench/output_file=$(OUTPUT_DIR)/output.bin" run $(runner_args)

clean:
	$(MAKE) -C $(GVSOC_ROOT) TARGETS=my_system MODULES=$(CURDIR) BUILDDIR=$(GVSOC_WORKDIR)/build INSTALLDIR=$(GVSOC_WORKDIR)/install clean
