# Small build entry for FastBF.
#
# Typical Linux/WSL workflow:
#   make install        # run the Python installer
#   make smoke          # run the self-contained example smoke check

PYTHON ?= python
ARCH ?= auto
CUDA_HOME ?=

INSTALL_ARGS :=
ifneq ($(ARCH),auto)
INSTALL_ARGS += --arch $(ARCH)
endif
ifneq ($(CUDA_HOME),)
INSTALL_ARGS += --cuda-home $(CUDA_HOME)
endif

.PHONY: all install build smoke guide clean help

all: install ## Build native binaries and install Python console commands

install: ## Full editable install through install_fastbf.py
	$(PYTHON) install_fastbf.py $(INSTALL_ARGS)

build: ## Build and stage native binaries without installing Python package
	$(PYTHON) install_fastbf.py --compile-only $(INSTALL_ARGS)

smoke: ## Run the self-contained example smoke check
	$(PYTHON) -m caponlib.cli doctor
	$(PYTHON) example/make_synthetic_sac.py
	$(PYTHON) -m caponlib.cli doctor example/config.ini
	$(PYTHON) -m caponlib.cli prepare example/config.ini
	$(PYTHON) -m caponlib.cli run example/config.ini --dry-run --serial --limit 1

guide: ## Print the short forget-proof usage reminder
	$(PYTHON) -m caponlib.cli guide

clean: ## Remove local build artifacts
	$(PYTHON) -c "import shutil; shutil.rmtree('build', ignore_errors=True)"
	$(MAKE) -C native clean
	rm -rf caponlib/bin

help: ## Show this help
	@awk 'BEGIN{FS=":.*##"; printf "\nUsage: make \033[36m<TARGET>\033[0m [ARCH=auto|sm_89] [CUDA_HOME=/usr/local/cuda]\n\nTargets:\n"} \
	     /^[a-zA-Z0-9_-]+:[^#]*##/{printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' \
	     $(MAKEFILE_LIST)
