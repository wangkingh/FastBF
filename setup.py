# -*- coding: utf-8 -*-
"""
setup.py — build CUDA-accelerated *cfk* extension (bindings + kernels).

Repo layout::

    FastBF/
    ├─ bindings/   fk_bind.cpp     (PyBind ↔ CUDA)
    ├─ cuda/       kernels.cu/.cuh
    └─ any_script.py  (imports cfk)

Adjust the three **PATH DEFINITIONS** below if you change folders.
"""
from __future__ import annotations
from pathlib import Path
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
import os, sysconfig
from pybind11 import get_include as pybind11_include
import shutil
from setuptools.command.install import install
try:
    from wheel.bdist_wheel import bdist_wheel
except ImportError:
    bdist_wheel = None

# -------------------------------------------------------------------------
# PATH DEFINITIONS
# -------------------------------------------------------------------------
ROOT_DIR   = Path(__file__).parent.resolve()
CUDA_DIR   = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")).resolve()
BIND_CPP   = ROOT_DIR / "bindings" / "fk_bind.cpp"
CUDA_SRC   = ROOT_DIR / "cuda"
KERNEL_CU  = CUDA_SRC / "kernels.cu"
KERNEL_SO  = "cfk_cuda.so"

# -------------------------------------------------------------------------
# Sanity checks
# -------------------------------------------------------------------------
if not (CUDA_DIR / "bin/nvcc").exists():
    raise RuntimeError("nvcc not found – set CUDA_HOME or install CUDA Toolkit")
if not KERNEL_CU.exists():
    raise FileNotFoundError(f"{KERNEL_CU} missing – check directory layout")

PYTHON_INC = sysconfig.get_path("include")
NVCC       = CUDA_DIR / "bin/nvcc"

# ------------------------------------------------------------------
# 1) 安装完成后清理 build/
# ------------------------------------------------------------------
class InstallClean(install):
    def run(self):
        super().run()                          # 先正常安装
        shutil.rmtree("build", ignore_errors=True)
        print("[clean] removed build/ after install")

# ------------------------------------------------------------------
# 2) 打 wheel 完成后也清理 build/（pip install . 时常走这里）
# ------------------------------------------------------------------
if bdist_wheel:
    class BDistWheelClean(bdist_wheel):       # type: ignore[misc]
        def run(self):
            super().run()                     # 先生成 .whl
            shutil.rmtree("build", ignore_errors=True)
            print("[clean] removed build/ after bdist_wheel")

# -------------------------------------------------------------------------
# 自定义 build_ext
# -------------------------------------------------------------------------
class BuildExtCUDA(build_ext):
    """
    自定义 build_ext 流程
    1. 编译 CUDA kernels → build/temp/.../cfk_cuda.so
    2. 编译 PyBind11 绑定 → cfk.cpython-*.so
    3. run() 阶段把 cfk_cuda.so 复制到每个扩展最终目录
    4. 编译开始前自动删除旧 *.so 和旧 build/ 目录，保持干净
    """

    # ---------- 编译前：先把旧产物清理掉 ----------
    def _pre_clean(self) -> None:
        # 1) 根目录残留的 .so 文件
        for so in ROOT_DIR.glob("*.so"):
            try:
                so.unlink()
                print(f"[clean] removed old {so.name}")
            except FileNotFoundError:
                pass

        # 2) 旧 build/ 目录
        shutil.rmtree(ROOT_DIR / "build", ignore_errors=True)
        print("[clean] removed build/")

    # ---------- 真正的编译入口 ----------
    def build_extensions(self) -> None:
        # 先做一次干净清理
        self._pre_clean()

        # nvcc 输出放在 build/temp/...
        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        self.cuda_so = build_temp / KERNEL_SO

        # CUDA → cfk_cuda.so
        nvcc_cmd = [
            str(NVCC), str(KERNEL_CU),
            "-shared", "-Xcompiler", "-fPIC",
            "-O3", "-std=c++14",
            "-gencode=arch=compute_120,code=sm_120",
            "-Xlinker", "-rpath,$ORIGIN",
            "-L", str(CUDA_DIR / "lib64"),
            "-lcublas", "-lcufft", "-lcudart",
            "-o", str(self.cuda_so),
        ]
        self.spawn(nvcc_cmd)
        if not self.cuda_so.exists():
            raise RuntimeError(f"nvcc 未生成 {self.cuda_so}")

        # 继续编译 pybind11 部分
        super().build_extensions()

    # ---------- 安装阶段：把 cfk_cuda.so 复制到最终目录 ----------
    def run(self) -> None:
        super().run()  # 触发 build_extensions 等内部流程

        # 将 cfk_cuda.so 拷贝到每个扩展 .so 所在目录
        for ext in self.extensions:
            final_path = Path(self.get_ext_fullpath(ext.name)).resolve()
            dst = final_path.parent / KERNEL_SO
            dst.parent.mkdir(parents=True, exist_ok=True)
            self.copy_file(str(self.cuda_so), str(dst))
            print(f"[copy] {self.cuda_so.name} → {dst}")
        
        if getattr(self, "inplace", False):
            shutil.rmtree(ROOT_DIR / "build", ignore_errors=True)
            print("[clean] removed build/ after build_ext --inplace")

# -------------------------------------------------------------------------
# Extension spec
# -------------------------------------------------------------------------
ext = Extension(
    "cfk",
    sources=[str(BIND_CPP)],
    include_dirs=[
        pybind11_include(),
        PYTHON_INC,
        str(CUDA_DIR / "include"),
        str(ROOT_DIR),
    ],
    library_dirs=[str(CUDA_DIR / "lib64")],
    runtime_library_dirs=["$ORIGIN", str(CUDA_DIR / "lib64")],
    libraries=["cufft", "cublas", "cudart"],
    language="c++",
    extra_compile_args=["-O3", "-std=c++17", "-Wall"],
    extra_link_args=["-Wl,-rpath,$ORIGIN"],   # 关键：同目录即可找到 cfk_cuda.so
)

cmds = {
    "build_ext": BuildExtCUDA,
    "install":   InstallClean,
}
if bdist_wheel:
    cmds["bdist_wheel"] = BDistWheelClean


setup(
    name="FastBF",
    version="0.1.0",
    description="CUDA-accelerated Capon kernels (cfk)",
    ext_modules=[ext],
    cmdclass=cmds,
    zip_safe=False,
    setup_requires=["pybind11>=2.11"],
)
