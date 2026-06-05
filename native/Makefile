# Minimal build entry for the FastBF native backend.

include common.mk

ROOT ?= ..
BIN_DIR := $(ROOT)/bin

NATIVE_TARGET := fastbf_native
STAGED_NATIVE := $(BIN_DIR)/$(NATIVE_TARGET)
NATIVE_SRC := fastbf_native.cu

.PHONY: all banner clean help

all: banner $(NATIVE_TARGET) $(STAGED_NATIVE) ## Build the native backend in release mode by default

banner:
	@printf "\033[32m=== FastBF CUDA build: ARCH=%s ===\033[0m\n" "$(CUDA_SM)"

$(NATIVE_TARGET): $(NATIVE_SRC)
	$(NVCC) $(OPT_NVCCFLAGS) $(NVCC_STD) $(NVCC_ARCH_FLAGS) -o $@ $(NATIVE_SRC) \
		-L$(CUDA_LIBDIR) -lcufft -lcublas -lcudart

$(STAGED_NATIVE): $(NATIVE_TARGET)
	mkdir -p "$(BIN_DIR)"
	cp "$(NATIVE_TARGET)" "$(STAGED_NATIVE)"
	chmod +x "$(STAGED_NATIVE)"

clean: ## Remove native object files and local native binaries
	rm -f $(NATIVE_TARGET)
	rm -rf ../bin

help: ## Show this help
	@awk 'BEGIN{FS=":.*##"; printf "\nUsage: make \033[36m<TARGET>\033[0m [ARCH=auto|sm_89]\n\nTargets:\n"} \
	     /^[a-zA-Z0-9_-]+:[^#]*##/{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}' \
	     $(MAKEFILE_LIST)
