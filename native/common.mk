# Shared CUDA build settings for FastBF native backend.

NATIVE_SRC_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

NVCC ?= $(shell command -v nvcc 2>/dev/null || \
          command -v /usr/local/cuda/bin/nvcc 2>/dev/null || \
          command -v /usr/local/cuda-12.6/bin/nvcc 2>/dev/null)
CXX ?= g++

CUDA_HOME   ?= $(patsubst %/bin/nvcc,%,$(NVCC))
CUDA_LIBDIR ?= $(CUDA_HOME)/lib64

ARCH ?= auto
CUDA_ARCH_FALLBACK ?= sm_80

normalize_sm = $(if $(filter sm_%,$(1)),$(1),sm_$(1))

ifeq ($(strip $(ARCH)),auto)
  DETECTED_ARCHS := $(strip $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | awk 'NF {gsub(/\./, "", $$1); print "sm_" $$1}' | sort -u))
  CUDA_SM := $(if $(DETECTED_ARCHS),$(firstword $(DETECTED_ARCHS)),$(CUDA_ARCH_FALLBACK))
else
  DETECTED_ARCHS :=
  CUDA_SM := $(call normalize_sm,$(ARCH))
endif

sm_version = $(patsubst sm_%,%,$(1))
NVCC_ARCH_FLAGS := -gencode arch=compute_$(call sm_version,$(CUDA_SM)),code=$(CUDA_SM) \
                   -gencode arch=compute_$(call sm_version,$(CUDA_SM)),code=compute_$(call sm_version,$(CUDA_SM))

OPT_CXXFLAGS  ?= -O3
OPT_NVCCFLAGS ?= -O3 --use_fast_math --generate-line-info

WARN_CXXFLAGS ?= -Wall -Wextra -pedantic
CXX_STD ?= -std=c++17
NVCC_STD ?= -std=c++17
