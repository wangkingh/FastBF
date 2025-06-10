// fk_bind.cpp  -----------------------------------------------------------
//  PyBind11 ↔ CUDA 动态库（cfk_cuda.so）绑定
//
//  主要改动：
//  1. 任何调用 ptr()/request() 的地方都保持 GIL
//  2. 仅在拿到全部裸指针之后才释放 GIL 去调 CUDA 侧函数
//  3. 对 dlsym 失败加上更友好的报错
//
//  ────────────────────────────────────────────────────────────────────────
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>

#include <dlfcn.h>
#include <stdexcept>
#include <string>
#include <sstream>
#include <iostream>

namespace py = pybind11;

// ---------- 常量 ----------
constexpr char kLibName[] = "cfk_cuda.so";

// ---------- 动态库加载 ----------
static void *cuda_lib()
{
    static void *handle = []() -> void *
    {
        // std::cerr << "[cuda_lib] try \"" << kLibName << "\"\n";

        // 1) cwd 直接 dlopen
        void *h = dlopen(kLibName, RTLD_NOW | RTLD_LOCAL);
        if (!h)
            std::cerr << "[cuda_lib] dlopen-1 failed: " << dlerror() << "\n";
        if (h)
            return h;

        // 2) 用 dladdr 找到绑定模块的目录
        Dl_info info{};
        if (dladdr(reinterpret_cast<void *>(cuda_lib), &info) && info.dli_fname)
        {
            std::string mod_path(info.dli_fname); // /abs/path/xxx.so
            auto slash = mod_path.find_last_of('/');
            if (slash != std::string::npos)
            {
                std::ostringstream oss;
                oss << mod_path.substr(0, slash + 1) << kLibName;
                h = dlopen(oss.str().c_str(), RTLD_NOW | RTLD_LOCAL);
                if (h)
                    return h;
            }
        }

        // 3) 仍失败 → 抛异常
        std::string err = "dlopen ";
        err += kLibName;
        err += " failed: ";
        err += dlerror();
        throw std::runtime_error(err);
    }();

    // std::cerr << "[cuda_lib] final handle=" << handle << "\n";
    return handle;
}

// --------- dlsym & 函数指针类型 ----------
#define LOAD_SYM(name, type) reinterpret_cast<type>(dlsym(cuda_lib(), name))

using fk_fn = void (*)(float *, int, float *, int, float *, int, int,
                       float *, int, float *, int,
                       float, float, int, int,
                       float, float, float, int, int);

using fk_multi_fn = fk_fn;

using fft_fn = void (*)(double *, int, int, double *);
using sm_fn = void (*)(float *, int, int, int, float *);
using cplxadd_fn = void (*)(float *, float *, std::complex<float> *, int, int);
using inv_fn = void (*)(std::complex<float> *, std::complex<float> *, int, int);

// --------- ndarray alias + helpers ----------
template <typename T>
using ndarray = py::array_t<T, py::array::c_style | py::array::forcecast>;

template <typename T>
inline T *ptr(ndarray<T> &a) { return static_cast<T *>(a.request().ptr); }

// =======================================================================
// 绑定函数实现
// =======================================================================

// ---- compute_fk (单 GPU) ----------------------------------------------
ndarray<float> py_compute_fk(
    ndarray<float> ismr,
    ndarray<float> ismi,
    ndarray<float> tfk,
    int nk,
    ndarray<float> rx,
    ndarray<float> ry,
    float kinc, float kmin,
    int nr, int nsamp,
    float delta, float smin, float sinc,
    int fave, int find)
{
    auto fn = LOAD_SYM("compute_fk_so", fk_fn);
    if (!fn)
        throw std::runtime_error("dlsym(\"compute_fk_so\") failed");

    /* ① 先拿裸指针 / size —— 持 GIL */
    float *p_ismr = ptr(ismr);
    float *p_ismi = ptr(ismi);
    float *p_tfk = ptr(tfk);
    float *p_rx = ptr(rx);
    float *p_ry = ptr(ry);

    int sz_ismr = ismr.size();
    int sz_ismi = ismi.size();
    int sz_tfk = tfk.size();
    int sz_rx = rx.size();
    int sz_ry = ry.size();

    /* ② 放 GIL 调 CUDA */
    {
        py::gil_scoped_release gil;
        fn(p_ismr, sz_ismr,
           p_ismi, sz_ismi,
           p_tfk, sz_tfk,
           nk,
           p_rx, sz_rx,
           p_ry, sz_ry,
           kinc, kmin, nr, nsamp,
           delta, smin, sinc,
           fave, find);
    }
    return tfk;
}

// ---- compute_fk_mulgpu -------------------------------------------------
ndarray<float> py_compute_fk_mulgpu(
    ndarray<float> ismr,
    ndarray<float> ismi,
    ndarray<float> tfk,
    int nk,
    ndarray<float> rx,
    ndarray<float> ry,
    float kinc, float kmin,
    int nr, int nsamp,
    float delta, float smin, float sinc,
    int fave, int find)
{
    auto fn = LOAD_SYM("compute_fk_so_mulgpu", fk_multi_fn);
    if (!fn)
        throw std::runtime_error("dlsym(\"compute_fk_so_mulgpu\") failed");

    float *p_ismr = ptr(ismr);
    float *p_ismi = ptr(ismi);
    float *p_tfk = ptr(tfk);
    float *p_rx = ptr(rx);
    float *p_ry = ptr(ry);

    int sz_ismr = ismr.size();
    int sz_ismi = ismi.size();
    int sz_tfk = tfk.size();
    int sz_rx = rx.size();
    int sz_ry = ry.size();

    py::gil_scoped_release gil;
    fn(p_ismr, sz_ismr,
       p_ismi, sz_ismi,
       p_tfk, sz_tfk,
       nk,
       p_rx, sz_rx,
       p_ry, sz_ry,
       kinc, kmin, nr, nsamp,
       delta, smin, sinc,
       fave, find);

    return tfk;
}

// ---- fft_cuda ----------------------------------------------------------
py::array py_fft_cuda(ndarray<double> xt, int nsamp, int nr)
{
    auto fn = LOAD_SYM("fft_cuda", fft_fn);
    if (!fn)
        throw std::runtime_error("dlsym(\"fft_cuda\") failed");

    std::size_t elems = static_cast<std::size_t>(nsamp / 2 + 1) * nr;
    ndarray<std::complex<double>> ft(elems);

    double *xt_ptr = ptr(xt);
    double *ft_ptr = reinterpret_cast<double *>(ft.mutable_data());

    {
        py::gil_scoped_release gil;
        fn(xt_ptr, nsamp, nr, ft_ptr);
    }
    return ft.attr("reshape")(py::make_tuple(nr, nsamp / 2 + 1));
}

// ---- compute_sm --------------------------------------------------------
ndarray<float> py_compute_sm(ndarray<float> ft, int nwin, int nr, int fave,
                             ndarray<float> sm)
{
    auto fn = LOAD_SYM("compute_sm_so", sm_fn);
    if (!fn)
        throw std::runtime_error("dlsym(\"compute_sm_so\") failed");

    float *ft_ptr = ptr(ft);
    float *sm_ptr = ptr(sm);

    {
        py::gil_scoped_release gil;
        fn(ft_ptr, nwin, nr, fave, sm_ptr);
    }
    return sm;
}

// ---- complex_add -------------------------------------------------------
py::array py_complex_add(ndarray<float> smr, ndarray<float> smi,
                         int nr, int fave)
{
    auto fn = LOAD_SYM("complex_add_so", cplxadd_fn);
    if (!fn)
        throw std::runtime_error("dlsym(\"complex_add_so\") failed");

    ndarray<std::complex<float>> out(static_cast<std::size_t>(nr) * nr *
                                     (2 * fave + 1));

    float *smr_ptr = ptr(smr);
    float *smi_ptr = ptr(smi);
    std::complex<float> *out_ptr =
        reinterpret_cast<std::complex<float> *>(out.mutable_data());

    {
        py::gil_scoped_release gil;
        fn(smr_ptr, smi_ptr, out_ptr, nr, fave);
    }
    return out.attr("reshape")(py::make_tuple(2 * fave + 1, nr, nr));
}

// ---- matrix_invert -----------------------------------------------------
py::array py_matrix_invert(ndarray<std::complex<float>> src,
                           int n, int batch)
{
    auto fn = LOAD_SYM("martix_invert_so", inv_fn);
    if (!fn)
        throw std::runtime_error("dlsym(\"martix_invert_so\") failed");

    ndarray<std::complex<float>> dst(static_cast<std::size_t>(batch) * n * n);

    auto *src_ptr = ptr(src);
    auto *dst_ptr = ptr(dst);

    {
        py::gil_scoped_release gil;
        fn(src_ptr, dst_ptr, n, batch);
    }
    return dst.attr("reshape")(py::make_tuple(batch, n, n));
}

// =======================================================================
// PYBIND11 模块导出
// =======================================================================
PYBIND11_MODULE(cfk, m)
{
    m.doc() = "CUDA-accelerated Capon kernels (compute_fk / fft_cuda / …)";

    m.def("compute_fk", &py_compute_fk, "单 GPU 频率-波数扫描");
    m.def("compute_fk_mulgpu", &py_compute_fk_mulgpu, "多 GPU 频率-波数扫描");
    m.def("fft_cuda", &py_fft_cuda, "GPU 实数 FFT (double→complex128)");
    m.def("compute_sm", &py_compute_sm, "互谱矩阵 & 权系数");
    m.def("complex_add", &py_complex_add, "实虚部 → complex");
    m.def("matrix_invert", &py_matrix_invert, "批量矩阵求逆 (cuBLAS)");

    // 旧名字向后兼容
    m.attr("complex_add_cuda") = m.attr("complex_add");
    m.attr("martix_invert") = m.attr("matrix_invert"); // 原拼写保留
}
