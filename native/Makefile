# Minimal build entry for the FastBF native backend.

include common.mk

ROOT ?= ..
BIN_DIR := $(ROOT)/bin

NATIVE_TARGET := fastbf_native
NATIVE_POWER_TARGET := fastbf_native_power
STAGED_NATIVE := $(BIN_DIR)/$(NATIVE_TARGET)
STAGED_NATIVE_POWER := $(BIN_DIR)/$(NATIVE_POWER_TARGET)
NATIVE_SRC := fastbf_native.cu
NATIVE_POWER_SRC := fastbf_native_power.cu

.PHONY: all banner clean help

all: banner $(NATIVE_TARGET) $(NATIVE_POWER_TARGET) $(STAGED_NATIVE) $(STAGED_NATIVE_POWER) ## Build the native backends in release mode by default

banner:
	@printf "\033[32m=== FastBF CUDA build: ARCH=%s ===\033[0m\n" "$(CUDA_SM)"

$(NATIVE_TARGET): $(NATIVE_SRC)
	$(NVCC) $(OPT_NVCCFLAGS) $(NVCC_STD) $(NVCC_ARCH_FLAGS) -o $@ $(NATIVE_SRC) \
		-L$(CUDA_LIBDIR) -lcufft -lcublas -lcudart $(STD_FILESYSTEM_LIBS)

$(NATIVE_POWER_TARGET): $(NATIVE_POWER_SRC)
	$(NVCC) $(OPT_NVCCFLAGS) $(NVCC_STD) $(NVCC_ARCH_FLAGS) -o $@ $(NATIVE_POWER_SRC) \
		-L$(CUDA_LIBDIR) -lcufft -lcudart $(STD_FILESYSTEM_LIBS)

$(STAGED_NATIVE): $(NATIVE_TARGET)
	mkdir -p "$(BIN_DIR)"
	cp "$(NATIVE_TARGET)" "$(STAGED_NATIVE)"
	chmod +x "$(STAGED_NATIVE)"

$(STAGED_NATIVE_POWER): $(NATIVE_POWER_TARGET)
	mkdir -p "$(BIN_DIR)"
	cp "$(NATIVE_POWER_TARGET)" "$(STAGED_NATIVE_POWER)"
	chmod +x "$(STAGED_NATIVE_POWER)"

clean: ## Remove native object files and local native binaries
	rm -f $(NATIVE_TARGET) $(NATIVE_POWER_TARGET)
	rm -rf ../bin

help: ## Show this help
	@awk 'BEGIN{FS=":.*##"; printf "\nUsage: make \033[36m<TARGET>\033[0m [ARCH=auto|sm_89]\n\nTargets:\n"} \
	     /^[a-zA-Z0-9_-]+:[^#]*##/{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}' \
	     $(MAKEFILE_LIST)
