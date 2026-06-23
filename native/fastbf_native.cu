#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <ctime>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cublas_v2.h>
#include <cuComplex.h>
#include <cufft.h>
#include <cuda_runtime.h>

namespace {
namespace fs = std::filesystem;

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr int kSacHeaderBytes = 632;
constexpr int kSacFloatCount = 70;
constexpr int kSacIntOffset = kSacFloatCount * 4;
constexpr int kFDelta = 0;
constexpr int kINvhdr = 6;
constexpr int kINpts = 9;
constexpr int kIIftype = 15;
constexpr int kILeven = 35;
using Clock = std::chrono::steady_clock;

#define CUDA_CHECK(expr)                                                     \
    do {                                                                     \
        cudaError_t _err = (expr);                                           \
        if (_err != cudaSuccess) {                                           \
            std::ostringstream _oss;                                         \
            _oss << "CUDA error at " << __FILE__ << ":" << __LINE__         \
                 << " " << cudaGetErrorString(_err);                        \
            throw std::runtime_error(_oss.str());                            \
        }                                                                    \
    } while (0)

#define CUFFT_CHECK(expr)                                                    \
    do {                                                                     \
        cufftResult _err = (expr);                                           \
        if (_err != CUFFT_SUCCESS) {                                         \
            std::ostringstream _oss;                                         \
            _oss << "cuFFT error at " << __FILE__ << ":" << __LINE__        \
                 << " code=" << static_cast<int>(_err);                     \
            throw std::runtime_error(_oss.str());                            \
        }                                                                    \
    } while (0)

#define CUBLAS_CHECK(expr)                                                   \
    do {                                                                     \
        cublasStatus_t _err = (expr);                                        \
        if (_err != CUBLAS_STATUS_SUCCESS) {                                 \
            std::ostringstream _oss;                                         \
            _oss << "cuBLAS error at " << __FILE__ << ":" << __LINE__       \
                 << " code=" << static_cast<int>(_err);                     \
            throw std::runtime_error(_oss.str());                            \
        }                                                                    \
    } while (0)

__global__ void accumulate_sm(float *ft, int nwin, int nr, int fave, float *sm)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int n_bins = 2 * fave + 1;
    if (i < nr && j < nr)
    {
        for (int n = 0; n < nwin; ++n)
        {
            for (int l = 0; l < n_bins; ++l)
            {
                sm[l * 2 * nr * nr + i * nr + j] +=
                    ft[n * 2 * nr * n_bins + i * n_bins + l] *
                        ft[n * 2 * nr * n_bins + j * n_bins + l] +
                    ft[n * 2 * nr * n_bins + (i + nr) * n_bins + l] *
                        ft[n * 2 * nr * n_bins + (j + nr) * n_bins + l];
                sm[l * 2 * nr * nr + (i + nr) * nr + j] +=
                    ft[n * 2 * nr * n_bins + j * n_bins + l] *
                        ft[n * 2 * nr * n_bins + (i + nr) * n_bins + l] -
                    ft[n * 2 * nr * n_bins + i * n_bins + l] *
                        ft[n * 2 * nr * n_bins + (j + nr) * n_bins + l];
            }
        }
    }
}

__global__ void scale_sm(float *sm, int total, float scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total)
        sm[idx] *= scale;
}

__global__ void compute_w(float *sm, int fave, int nr, float *w)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int l = blockIdx.y * blockDim.y + threadIdx.y;
    int n_bins = 2 * fave + 1;
    if (i < nr && l < n_bins)
    {
        float re = sm[l * 2 * nr * nr + i * nr + i];
        float im = sm[l * 2 * nr * nr + (i + nr) * nr + i];
        float mag_sq = re * re + im * im;
        float safe = fmaxf(mag_sq, 1.0e-12f);
        float mask = (mag_sq > 0.0f) ? 1.0f : 0.0f;
        w[l * nr + i] = powf(safe, -0.25f) * mask;
    }
}

__global__ void compute_sm_weighted(float *sm, float *w, int fave, int nr)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int n_bins = 2 * fave + 1;
    if (i < nr && j < nr)
    {
        for (int l = 0; l < n_bins; ++l)
        {
            float factor = w[l * nr + i] * w[l * nr + j];
            sm[l * 2 * nr * nr + i * nr + j] *= factor;
            sm[l * 2 * nr * nr + (i + nr) * nr + j] *= factor;
        }
    }
}

__global__ void apply_diagonal_loading(float *sm, int n_bins, int nr,
                                       int nsamp, float dl)
{
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= n_bins)
        return;
    float trace = 0.0f;
    for (int i = 0; i < nr; ++i)
        trace += sm[l * 2 * nr * nr + i * nr + i];
    float load = trace / nsamp * dl;
    for (int i = 0; i < nr; ++i)
        sm[l * 2 * nr * nr + i * nr + i] += load;
}

__global__ void sm_interleaved_to_complex(float *sm, cuComplex *dst,
                                          int n_bins, int nr)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_bins * nr * nr;
    if (idx >= total)
        return;
    int l = idx / (nr * nr);
    int rem = idx % (nr * nr);
    int row = rem / nr;
    int col = rem % nr;
    float re = sm[l * 2 * nr * nr + row * nr + col];
    float im = sm[l * 2 * nr * nr + (row + nr) * nr + col];
    dst[idx] = make_cuComplex(re, im);
}

__global__ void complex_to_split_real_imag(cuComplex *src, float *dst_re,
                                           float *dst_im, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total)
    {
        dst_re[idx] = cuCrealf(src[idx]);
        dst_im[idx] = cuCimagf(src[idx]);
    }
}

__global__ void precompute_tau_grid(float *tau, float *rx, float *ry,
                                    int nr, int nk, float smin, float sinc)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nr * nk * nk;
    if (idx >= total)
        return;
    int receiver = idx % nr;
    int grid_id = idx / nr;
    int i = grid_id / nk;
    int j = grid_id % nk;
    float sx = smin + i * sinc;
    float sy = smin + j * sinc;
    tau[idx] = rx[receiver] * sx + ry[receiver] * sy;
}

__global__ void gather_tau_subset(float *dst, const float *master_tau,
                                  const int *master_indices,
                                  int nr, int master_nr, int nk)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nr * nk * nk;
    if (idx >= total)
        return;
    int receiver = idx % nr;
    int grid_id = idx / nr;
    int master_receiver = master_indices[receiver];
    dst[idx] = master_tau[grid_id * master_nr + master_receiver];
}

__global__ void precompute_steering_chunk(float *steer_re, float *steer_im,
                                          float *tau, int nr, int nk,
                                          int n_bins, int nsamp, float delta,
                                          int fave, int find)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_bins * nk * nk * nr;
    if (idx >= total)
        return;
    int receiver = idx % nr;
    int grid_id = (idx / nr) % (nk * nk);
    int local_bin = idx / (nr * nk * nk);
    int freq_bin = find - fave + local_bin;
    float freq = freq_bin / (nsamp * delta);
    float phase = 2.0f * static_cast<float>(kPi) * freq *
                  tau[grid_id * nr + receiver];
    steer_re[idx] = cosf(phase);
    steer_im[idx] = sinf(phase);
}

__global__ void compute_fk_steering(float *ismr, float *ismi, float *tfk,
                                    int nk, float *steer_re, float *steer_im,
                                    int nr, int n_bins)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nk || j >= nk)
        return;

    int grid_id = i * nk + j;
    int grid_count = nk * nk;
    for (int l = 0; l < n_bins; ++l)
    {
        int steer_base = (l * grid_count + grid_id) * nr;
        float tmp_fk = 0.0f;
        for (int m = 0; m < nr; ++m)
        {
            tmp_fk += ismr[l * nr * nr + m * nr + m];
            float mr = steer_re[steer_base + m];
            float mi = steer_im[steer_base + m];
            for (int n = m + 1; n < nr; ++n)
            {
                float nr_re = steer_re[steer_base + n];
                float nr_im = steer_im[steer_base + n];
                float factor_re = mr * nr_re + mi * nr_im;
                float factor_im = mr * nr_im - mi * nr_re;
                tmp_fk += 2.0f * (
                    ismr[l * nr * nr + m * nr + n] * factor_re -
                    ismi[l * nr * nr + m * nr + n] * factor_im);
            }
        }
        tfk[i * nk + j] += 1.0f / tmp_fk;
    }
}

__global__ void compute_bartlett_fk_steering(float *sm, float *tfk, int nk,
                                             float *steer_re, float *steer_im,
                                             int nr, int n_bins)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nk || j >= nk)
        return;

    int grid_id = i * nk + j;
    int grid_count = nk * nk;
    float acc = 0.0f;
    for (int l = 0; l < n_bins; ++l)
    {
        int steer_base = (l * grid_count + grid_id) * nr;
        float tmp_fk = 0.0f;
        for (int m = 0; m < nr; ++m)
        {
            tmp_fk += sm[l * 2 * nr * nr + m * nr + m];
            float mr = steer_re[steer_base + m];
            float mi = steer_im[steer_base + m];
            for (int n = m + 1; n < nr; ++n)
            {
                float nr_re = steer_re[steer_base + n];
                float nr_im = steer_im[steer_base + n];
                float factor_re = mr * nr_re + mi * nr_im;
                float factor_im = mr * nr_im - mi * nr_re;
                float re = sm[l * 2 * nr * nr + m * nr + n];
                float im = sm[l * 2 * nr * nr + (m + nr) * nr + n];
                tmp_fk += 2.0f * (re * factor_re - im * factor_im);
            }
        }
        acc += tmp_fk;
    }
    tfk[i * nk + j] += acc;
}

struct Options {
    std::string saclist;
    std::string saclist_list;
    std::string geometry;
    std::string output;
    bool plan_only = false;
    float fmin = std::numeric_limits<float>::quiet_NaN();
    float fmax = std::numeric_limits<float>::quiet_NaN();
    double win_len = 0.0;
    double shift_len = 0.0;
    std::string taper = "tukey";
    double taper_alpha = 0.05;
    int record_samples = 0;
    double record_seconds = 0.0;
    float s_range = 0.0f;
    float s_res = 0.0f;
    int freq_chunk_bins = 33;
    int gpu = 0;
    float dl = 1.0f;
    double host_steering_max_gb = 32.0;
    std::string steering_cache = "none";
    std::string method = "capon";
    std::string mode = "stack";
    int group_windows = 0;
    std::vector<int> skip_steps;
    bool profile = false;
};

struct ProfileStats {
    double master_tau_ms = 0.0;
    double select_tau_ms = 0.0;
    double read_ms = 0.0;
    double window_ms = 0.0;
    double fft_ms = 0.0;
    double chunk_total_ms = 0.0;
    double chunk_setup_ms = 0.0;
    double steering_ms = 0.0;
    double pack_ms = 0.0;
    double h2d_ms = 0.0;
    double covariance_ms = 0.0;
    double inverse_ms = 0.0;
    double fk_ms = 0.0;
    double d2h_ms = 0.0;
    double write_ms = 0.0;
    int chunks = 0;
};

struct SacRow {
    std::string nsl_id;
    std::string path;
};

struct GeometryTable {
    std::vector<std::string> nsl_ids;
    std::vector<float> rx;
    std::vector<float> ry;
    std::unordered_map<std::string, int> index_by_nsl;
};

struct SacHeader {
    std::string path;
    char endian = '<';
    float delta = 0.0f;
    int npts = 0;
};

struct ListPlan {
    std::string saclist_path;
    std::vector<SacRow> rows;
    std::vector<SacHeader> headers;
    std::vector<std::pair<int, int>> windows;
    int common_npts = 0;
    int nr = 0;
    double dt = 0.0;
    int nwin = 0;
    int nsamp = 0;
    int nfreq = 0;
};

struct BandChunk {
    int start_bin = 0;
    int end_bin = 0;
    int n_bins() const { return end_bin - start_bin + 1; }
    int fave() const { return (n_bins() - 1) / 2; }
    int find() const { return (start_bin + end_bin) / 2; }
};

struct SteeringPlan {
    std::string mode = "none";
    bool reusable = false;
    std::string reason;
    std::vector<BandChunk> chunks;
    int selected_start = 0;
    int selected_end = 0;
    int selected_bins = 0;
    std::size_t values = 0;
    std::size_t bytes = 0;
};

std::size_t align_up(std::size_t value, std::size_t alignment = 256)
{
    return (value + alignment - 1) / alignment * alignment;
}

double elapsed_ms(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void add_profile(ProfileStats &dst, const ProfileStats &src)
{
    dst.master_tau_ms += src.master_tau_ms;
    dst.select_tau_ms += src.select_tau_ms;
    dst.read_ms += src.read_ms;
    dst.window_ms += src.window_ms;
    dst.fft_ms += src.fft_ms;
    dst.chunk_total_ms += src.chunk_total_ms;
    dst.chunk_setup_ms += src.chunk_setup_ms;
    dst.steering_ms += src.steering_ms;
    dst.pack_ms += src.pack_ms;
    dst.h2d_ms += src.h2d_ms;
    dst.covariance_ms += src.covariance_ms;
    dst.inverse_ms += src.inverse_ms;
    dst.fk_ms += src.fk_ms;
    dst.d2h_ms += src.d2h_ms;
    dst.write_ms += src.write_ms;
    dst.chunks += src.chunks;
}

void print_profile(const std::string &label, const ProfileStats &p)
{
    std::ios old_state(nullptr);
    old_state.copyfmt(std::cerr);
    std::cerr << std::fixed << std::setprecision(3)
              << "[fastbf_native][profile] " << label
              << " master_tau_ms=" << p.master_tau_ms
              << " select_tau_ms=" << p.select_tau_ms
              << " read_ms=" << p.read_ms
              << " window_ms=" << p.window_ms
              << " fft_ms=" << p.fft_ms
              << " chunk_total_ms=" << p.chunk_total_ms
              << " chunk_setup_ms=" << p.chunk_setup_ms
              << " steering_ms=" << p.steering_ms
              << " pack_ms=" << p.pack_ms
              << " h2d_ms=" << p.h2d_ms
              << " covariance_ms=" << p.covariance_ms
              << " inverse_ms=" << p.inverse_ms
              << " fk_ms=" << p.fk_ms
              << " d2h_ms=" << p.d2h_ms
              << " write_ms=" << p.write_ms
              << " chunks=" << p.chunks
              << "\n";
    std::cerr.copyfmt(old_state);
}

struct MemoryPlan {
    std::size_t permanent = 0;
    std::size_t master_tau = 0;
    std::size_t active_tau = 0;
    std::size_t receiver_map = 0;
    std::size_t fft = 0;
    std::size_t cufft_work = 0;
    std::size_t steering = 0;
    std::size_t covariance = 0;
    std::size_t inverse = 0;
    std::size_t beamforming = 0;
    std::size_t chunk_combined = 0;
    std::size_t arena = 0;
};

struct WorkspacePlan {
    MemoryPlan memory;
    int max_nr = 0;
    int max_nwin = 0;
    int max_nsamp = 0;
    int max_nfreq = 0;
    int max_selected_bins = 0;
    int max_chunk_bins = 0;
    int master_nr = 0;
    std::size_t max_cufft_work = 0;
    std::string source_saclist;
};

class DeviceArena {
public:
    explicit DeviceArena(std::size_t bytes) : size_(bytes)
    {
        CUDA_CHECK(cudaMalloc(&base_, size_));
    }
    ~DeviceArena()
    {
        if (base_)
            cudaFree(base_);
    }
    DeviceArena(const DeviceArena &) = delete;
    DeviceArena &operator=(const DeviceArena &) = delete;

    void reset(std::size_t cursor = 0) { cursor_ = cursor; }
    std::size_t cursor() const { return cursor_; }

    template <typename T>
    T *alloc(std::size_t count, const char *name)
    {
        std::size_t bytes = sizeof(T) * count;
        std::size_t start = align_up(cursor_);
        std::size_t end = start + bytes;
        if (end > size_)
        {
            std::ostringstream oss;
            oss << "GPU arena overflow at " << name << ": need " << end
                << " bytes, arena has " << size_;
            throw std::runtime_error(oss.str());
        }
        cursor_ = end;
        return reinterpret_cast<T *>(base_ + start);
    }

    void *alloc_bytes(std::size_t bytes, const char *name)
    {
        std::size_t start = align_up(cursor_);
        std::size_t end = start + bytes;
        if (end > size_)
        {
            std::ostringstream oss;
            oss << "GPU arena overflow at " << name << ": need " << end
                << " bytes, arena has " << size_;
            throw std::runtime_error(oss.str());
        }
        cursor_ = end;
        return base_ + start;
    }

private:
    unsigned char *base_ = nullptr;
    std::size_t size_ = 0;
    std::size_t cursor_ = 0;
};

struct PersistentDevice {
    float *master_rx = nullptr;
    float *master_ry = nullptr;
    float *master_tau = nullptr;
    int *receiver_master_index = nullptr;
    float *tau = nullptr;
    std::size_t scratch_start = 0;
};

struct CufftPlan {
    cufftHandle handle = 0;
    ~CufftPlan()
    {
        if (handle)
            cufftDestroy(handle);
    }
};

struct CublasHandle {
    cublasHandle_t handle = nullptr;
    CublasHandle() { CUBLAS_CHECK(cublasCreate_v2(&handle)); }
    ~CublasHandle()
    {
        if (handle)
            cublasDestroy_v2(handle);
    }
};

struct GlobalSteeringCache {
    std::string mode = "none";
    int selected_start = 0;
    std::size_t values = 0;
    std::vector<float> host_re;
    std::vector<float> host_im;
    float *device_re = nullptr;
    float *device_im = nullptr;

    GlobalSteeringCache() = default;
    GlobalSteeringCache(const GlobalSteeringCache &) = delete;
    GlobalSteeringCache &operator=(const GlobalSteeringCache &) = delete;
    GlobalSteeringCache(GlobalSteeringCache &&other) noexcept
        : mode(std::move(other.mode)),
          selected_start(other.selected_start),
          values(other.values),
          host_re(std::move(other.host_re)),
          host_im(std::move(other.host_im)),
          device_re(other.device_re),
          device_im(other.device_im)
    {
        other.device_re = nullptr;
        other.device_im = nullptr;
    }
    GlobalSteeringCache &operator=(GlobalSteeringCache &&other) noexcept
    {
        if (this != &other)
        {
            if (device_re)
                cudaFree(device_re);
            if (device_im)
                cudaFree(device_im);
            mode = std::move(other.mode);
            selected_start = other.selected_start;
            values = other.values;
            host_re = std::move(other.host_re);
            host_im = std::move(other.host_im);
            device_re = other.device_re;
            device_im = other.device_im;
            other.device_re = nullptr;
            other.device_im = nullptr;
        }
        return *this;
    }

    ~GlobalSteeringCache()
    {
        if (device_re)
            cudaFree(device_re);
        if (device_im)
            cudaFree(device_im);
    }

    const float *host_re_ptr() const
    {
        return mode == "host" ? host_re.data() : nullptr;
    }
    const float *host_im_ptr() const
    {
        return mode == "host" ? host_im.data() : nullptr;
    }
    const float *device_re_ptr() const
    {
        return mode == "device" ? device_re : nullptr;
    }
    const float *device_im_ptr() const
    {
        return mode == "device" ? device_im : nullptr;
    }
};

std::vector<std::string> split_tab(const std::string &line)
{
    std::vector<std::string> out;
    std::string item;
    std::stringstream ss(line);
    while (std::getline(ss, item, '\t'))
        out.push_back(item);
    if (!line.empty() && line.back() == '\t')
        out.emplace_back();
    return out;
}

std::string trim(std::string text)
{
    const char *spaces = " \t\r\n";
    auto left = text.find_first_not_of(spaces);
    if (left == std::string::npos)
        return "";
    auto right = text.find_last_not_of(spaces);
    return text.substr(left, right - left + 1);
}

int require_column(const std::vector<std::string> &header,
                   const std::string &name, const std::string &path)
{
    for (std::size_t i = 0; i < header.size(); ++i)
        if (trim(header[i]) == name)
            return static_cast<int>(i);
    throw std::runtime_error(path + " missing column: " + name);
}

std::vector<SacRow> read_saclist(const std::string &path)
{
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("cannot open saclist: " + path);
    std::string line;
    if (!std::getline(in, line))
        throw std::runtime_error("empty saclist: " + path);
    auto header = split_tab(line);
    int nsl_col = require_column(header, "nsl_id", path);
    int path_col = require_column(header, "sac_path", path);
    std::vector<SacRow> rows;
    while (std::getline(in, line))
    {
        if (trim(line).empty())
            continue;
        auto fields = split_tab(line);
        if (static_cast<int>(fields.size()) <= std::max(nsl_col, path_col))
            throw std::runtime_error("malformed saclist row: " + line);
        SacRow row{trim(fields[nsl_col]), trim(fields[path_col])};
        if (!row.nsl_id.empty() && !row.path.empty())
            rows.push_back(std::move(row));
    }
    if (rows.empty())
        throw std::runtime_error("saclist contains no rows: " + path);
    return rows;
}

std::string resolve_manifest_path(const fs::path &manifest,
                                  const std::string &text)
{
    fs::path item(trim(text));
    if (item.empty())
        return "";
    if (item.is_absolute())
        return item.string();
    return (manifest.parent_path() / item).lexically_normal().string();
}

std::vector<std::string> read_saclist_manifest(const std::string &path)
{
    fs::path manifest(path);
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("cannot open saclist manifest: " + path);

    std::vector<std::string> raw_lines;
    std::string line;
    while (std::getline(in, line))
    {
        std::string text = trim(line);
        if (text.empty() || text[0] == '#')
            continue;
        raw_lines.push_back(text);
    }
    if (raw_lines.empty())
        throw std::runtime_error("saclist manifest contains no paths: " + path);

    std::vector<std::string> paths;
    auto first_fields = split_tab(raw_lines.front());
    int path_col = -1;
    for (std::size_t i = 0; i < first_fields.size(); ++i)
    {
        std::string field = trim(first_fields[i]);
        if (field == "saclist" || field == "sac_list" || field == "path")
            path_col = static_cast<int>(i);
    }

    if (path_col >= 0)
    {
        for (std::size_t row = 1; row < raw_lines.size(); ++row)
        {
            auto fields = split_tab(raw_lines[row]);
            if (static_cast<int>(fields.size()) <= path_col)
                throw std::runtime_error("malformed saclist manifest row: " + raw_lines[row]);
            auto resolved = resolve_manifest_path(manifest, fields[path_col]);
            if (!resolved.empty())
                paths.push_back(resolved);
        }
    }
    else
    {
        for (const auto &raw : raw_lines)
        {
            auto fields = split_tab(raw);
            auto resolved = resolve_manifest_path(manifest, fields.front());
            if (!resolved.empty())
                paths.push_back(resolved);
        }
    }
    if (paths.empty())
        throw std::runtime_error("saclist manifest contains no usable paths: " + path);
    return paths;
}

GeometryTable read_geometry_table(const std::string &path)
{
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("cannot open geometry TSV: " + path);
    std::string line;
    if (!std::getline(in, line))
        throw std::runtime_error("empty geometry TSV: " + path);
    auto header = split_tab(line);
    int nsl_col = require_column(header, "nsl_id", path);
    int rx_col = require_column(header, "rx", path);
    int ry_col = require_column(header, "ry", path);
    GeometryTable table;
    while (std::getline(in, line))
    {
        if (trim(line).empty())
            continue;
        auto fields = split_tab(line);
        if (static_cast<int>(fields.size()) <= std::max({nsl_col, rx_col, ry_col}))
            throw std::runtime_error("malformed geometry row: " + line);
        std::string nsl_id = trim(fields[nsl_col]);
        if (nsl_id.empty())
            throw std::runtime_error("geometry row has empty nsl_id: " + line);
        if (table.index_by_nsl.find(nsl_id) != table.index_by_nsl.end())
            throw std::runtime_error("duplicate geometry nsl_id: " + nsl_id);
        int index = static_cast<int>(table.nsl_ids.size());
        table.index_by_nsl[nsl_id] = index;
        table.nsl_ids.push_back(std::move(nsl_id));
        table.rx.push_back(std::stof(trim(fields[rx_col])));
        table.ry.push_back(std::stof(trim(fields[ry_col])));
    }
    if (table.nsl_ids.empty())
        throw std::runtime_error("geometry TSV contains no rows: " + path);
    return table;
}

std::vector<int> receiver_master_indices(const GeometryTable &geometry,
                                         const std::vector<SacRow> &rows)
{
    std::vector<int> indices(rows.size());
    for (std::size_t i = 0; i < rows.size(); ++i)
    {
        auto it = geometry.index_by_nsl.find(rows[i].nsl_id);
        if (it == geometry.index_by_nsl.end())
            throw std::runtime_error("geometry missing nsl_id: " + rows[i].nsl_id);
        indices[i] = it->second;
    }
    return indices;
}

template <typename T>
T bswap(T value)
{
    T out{};
    auto *src = reinterpret_cast<unsigned char *>(&value);
    auto *dst = reinterpret_cast<unsigned char *>(&out);
    for (std::size_t i = 0; i < sizeof(T); ++i)
        dst[i] = src[sizeof(T) - 1 - i];
    return out;
}

template <typename T>
T read_scalar(const std::vector<char> &raw, int offset, char endian)
{
    T value{};
    std::memcpy(&value, raw.data() + offset, sizeof(T));
    if (endian == '>')
        value = bswap(value);
    return value;
}

char detect_endian(const std::vector<char> &raw)
{
    for (char endian : {'<', '>'})
    {
        int nvhdr = read_scalar<int>(raw, kSacIntOffset + kINvhdr * 4, endian);
        int npts = read_scalar<int>(raw, kSacIntOffset + kINpts * 4, endian);
        float delta = read_scalar<float>(raw, kFDelta * 4, endian);
        if (nvhdr > 0 && nvhdr <= 20 && npts > 0 &&
            npts < 2000000000 && std::isfinite(delta))
            return endian;
    }
    throw std::runtime_error("could not detect SAC byte order");
}

SacHeader parse_sac_header(const std::string &path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("cannot open SAC file: " + path);
    std::vector<char> raw(kSacHeaderBytes);
    in.read(raw.data(), raw.size());
    if (in.gcount() != kSacHeaderBytes)
        throw std::runtime_error("SAC file too small: " + path);
    char endian = detect_endian(raw);
    float delta = read_scalar<float>(raw, kFDelta * 4, endian);
    int npts = read_scalar<int>(raw, kSacIntOffset + kINpts * 4, endian);
    int iftype = read_scalar<int>(raw, kSacIntOffset + kIIftype * 4, endian);
    int leven = read_scalar<int>(raw, kSacIntOffset + kILeven * 4, endian);
    if (!std::isfinite(delta) || delta <= 0.0f)
        throw std::runtime_error("invalid SAC delta: " + path);
    if (npts <= 0)
        throw std::runtime_error("invalid SAC npts: " + path);
    if (iftype != -12345 && iftype != 1)
        throw std::runtime_error("unsupported SAC iftype: " + path);
    if (leven == 0)
        throw std::runtime_error("uneven SAC file: " + path);
    return SacHeader{path, endian, delta, npts};
}

std::vector<SacHeader> read_headers(const std::vector<SacRow> &rows)
{
    std::vector<SacHeader> headers;
    headers.reserve(rows.size());
    for (const auto &row : rows)
        headers.push_back(parse_sac_header(row.path));
    float ref_delta = headers.front().delta;
    for (const auto &header : headers)
    {
        float tol = std::max(1.0e-9f, std::abs(ref_delta) * 1.0e-6f);
        if (std::abs(header.delta - ref_delta) > tol)
            throw std::runtime_error("SAC delta mismatch: " + header.path);
    }
    return headers;
}

std::vector<float> read_sac_row(const SacHeader &header, int npts)
{
    std::ifstream in(header.path, std::ios::binary);
    if (!in)
        throw std::runtime_error("cannot open SAC file: " + header.path);
    in.seekg(kSacHeaderBytes);
    std::vector<float> row(npts);
    in.read(reinterpret_cast<char *>(row.data()), npts * sizeof(float));
    if (in.gcount() != static_cast<std::streamsize>(npts * sizeof(float)))
        throw std::runtime_error("SAC data shorter than expected: " + header.path);
    if (header.endian == '>')
        for (float &value : row)
            value = bswap(value);
    return row;
}

std::vector<float> read_data_matrix(const std::vector<SacHeader> &headers,
                                    int common_npts)
{
    int nr = static_cast<int>(headers.size());
    std::vector<float> matrix(static_cast<std::size_t>(nr) * common_npts);
    for (int r = 0; r < nr; ++r)
    {
        auto row = read_sac_row(headers[r], common_npts);
        std::copy(row.begin(), row.end(),
                  matrix.begin() + static_cast<std::size_t>(r) * common_npts);
    }
    return matrix;
}

int py_round(double value)
{
    return static_cast<int>(std::nearbyint(value));
}

std::vector<std::pair<int, int>> build_windows(int npts, double dt,
                                               double win_len,
                                               double shift_len,
                                               const std::vector<int> &skip_steps)
{
    int nsamp = py_round(win_len / dt);
    int hop = py_round(shift_len / dt);
    if (nsamp <= 0 || hop <= 0)
        throw std::runtime_error("invalid window/hop samples");
    if (nsamp > npts)
        throw std::runtime_error("window length exceeds npts");
    std::vector<std::pair<int, int>> windows;
    int total_steps = 0;
    for (int start = 0; start + nsamp <= npts; start += hop, ++total_steps)
    {
        if (std::find(skip_steps.begin(), skip_steps.end(), total_steps) != skip_steps.end())
            continue;
        windows.emplace_back(start, start + nsamp);
    }
    for (int step : skip_steps)
    {
        if (step < 0 || step >= total_steps)
            throw std::runtime_error("--skip-step index out of range for this record");
    }
    if (windows.empty())
        throw std::runtime_error("all windows were skipped by --skip-step");
    return windows;
}

ListPlan build_list_plan(const std::string &saclist_path,
                         const Options &opt)
{
    ListPlan plan;
    plan.saclist_path = saclist_path;
    plan.rows = read_saclist(saclist_path);
    if (opt.plan_only)
    {
        plan.headers.push_back(parse_sac_header(plan.rows.front().path));
    }
    else
    {
        plan.headers = read_headers(plan.rows);
    }
    const SacHeader &shape_header = plan.headers.front();
    if (opt.record_samples > 0)
    {
        plan.common_npts = opt.record_samples;
    }
    else if (opt.record_seconds > 0.0)
    {
        plan.common_npts = py_round(opt.record_seconds / shape_header.delta);
    }
    else
    {
        plan.common_npts = shape_header.npts;
        if (!opt.plan_only)
        {
            for (const auto &header : plan.headers)
                plan.common_npts = std::min(plan.common_npts, header.npts);
        }
    }
    for (const auto &header : plan.headers)
    {
        if (header.npts < plan.common_npts)
            throw std::runtime_error("SAC file shorter than requested record length: " +
                                     header.path);
    }
    plan.nr = static_cast<int>(plan.rows.size());
    plan.dt = shape_header.delta;
    plan.windows = build_windows(plan.common_npts, plan.dt,
                                 opt.win_len, opt.shift_len,
                                 opt.skip_steps);
    plan.nwin = static_cast<int>(plan.windows.size());
    plan.nsamp = plan.windows.front().second - plan.windows.front().first;
    plan.nfreq = plan.nsamp / 2 + 1;
    return plan;
}

std::vector<double> build_taper(int nsamp, const std::string &mode, double alpha)
{
    std::vector<double> taper(nsamp);
    if (nsamp == 1)
    {
        taper[0] = 1.0;
        return taper;
    }
    if (mode == "none")
    {
        std::fill(taper.begin(), taper.end(), 1.0);
        return taper;
    }
    if (mode == "hann")
    {
        for (int i = 0; i < nsamp; ++i)
            taper[i] = 0.5 - 0.5 * std::cos(2.0 * kPi * i / (nsamp - 1));
        return taper;
    }
    if (mode == "tukey")
    {
        if (alpha <= 0.0)
        {
            std::fill(taper.begin(), taper.end(), 1.0);
            return taper;
        }
        if (alpha >= 1.0)
        {
            for (int i = 0; i < nsamp; ++i)
                taper[i] = 0.5 - 0.5 * std::cos(2.0 * kPi * i / (nsamp - 1));
            return taper;
        }
        for (int i = 0; i < nsamp; ++i)
        {
            double x = static_cast<double>(i) / static_cast<double>(nsamp - 1);
            if (x < alpha / 2.0)
                taper[i] = 0.5 * (1.0 + std::cos(kPi * (2.0 * x / alpha - 1.0)));
            else if (x <= 1.0 - alpha / 2.0)
                taper[i] = 1.0;
            else
                taper[i] = 0.5 * (1.0 + std::cos(kPi * (2.0 * x / alpha - 2.0 / alpha + 1.0)));
        }
        return taper;
    }
    throw std::runtime_error("unsupported taper mode: " + mode);
}

std::string taper_description(const Options &opt)
{
    std::ostringstream oss;
    oss << opt.taper;
    if (opt.taper == "tukey")
        oss << "(alpha=" << opt.taper_alpha << ")";
    return oss.str();
}

double taper_power_seconds(int nsamp, double dt, const std::string &mode, double alpha)
{
    auto taper = build_taper(nsamp, mode, alpha);
    double sum_sq = 0.0;
    for (double value : taper)
        sum_sq += value * value;
    return sum_sq * dt;
}

double taper_amplitude_seconds(int nsamp, double dt, const std::string &mode, double alpha)
{
    auto taper = build_taper(nsamp, mode, alpha);
    double sum = 0.0;
    for (double value : taper)
        sum += value;
    return sum * dt;
}

std::vector<double> window_taper(int nsamp, const Options &opt)
{
    return build_taper(nsamp, opt.taper, opt.taper_alpha);
}

void normalize_taper_options(Options &opt)
{
    std::transform(opt.taper.begin(), opt.taper.end(), opt.taper.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (opt.taper == "hanning")
        opt.taper = "hann";
    else if (opt.taper == "rect" || opt.taper == "rectangular" || opt.taper == "boxcar")
        opt.taper = "none";
    if (opt.taper != "hann" && opt.taper != "tukey" && opt.taper != "none")
        throw std::runtime_error("--taper must be hann, tukey, or none");
    if (opt.taper == "tukey" && !(opt.taper_alpha > 0.0 && opt.taper_alpha <= 1.0))
        throw std::runtime_error("--tukey-alpha must be in (0, 1]");
}

std::vector<double> build_window_stack(const std::vector<float> &matrix,
                                       int nr, int npts,
                                       const std::vector<std::pair<int, int>> &windows,
                                       const Options &opt)
{
    int nwin = static_cast<int>(windows.size());
    int nsamp = windows.front().second - windows.front().first;
    auto taper = window_taper(nsamp, opt);
    std::vector<double> out(static_cast<std::size_t>(nwin) * nr * nsamp);
    for (int w = 0; w < nwin; ++w)
    {
        int start = windows[w].first;
        for (int r = 0; r < nr; ++r)
        {
            std::size_t row_base = static_cast<std::size_t>(r) * npts;
            double mean = 0.0;
            for (int s = 0; s < nsamp; ++s)
                mean += matrix[row_base + start + s];
            mean /= nsamp;
            std::size_t out_base = (static_cast<std::size_t>(w) * nr + r) * nsamp;
            for (int s = 0; s < nsamp; ++s)
                out[out_base + s] =
                    (static_cast<double>(matrix[row_base + start + s]) - mean) *
                    taper[s];
        }
    }
    return out;
}

std::pair<int, int> band_bin_range(float fmin, float fmax,
                                   int nsamp, double dt)
{
    double fmid = 0.5 * (static_cast<double>(fmin) + fmax);
    double df = 0.5 * (static_cast<double>(fmax) - fmin);
    int find = py_round(fmid * nsamp * dt);
    int fave = py_round(df * nsamp * dt);
    if (fave < 1)
        throw std::runtime_error("bandwidth too narrow");
    return {find - fave, find + fave};
}

int normalize_chunk_bins(int max_bins)
{
    if (max_bins <= 0)
        throw std::runtime_error("freq chunk bins must be positive");
    if (max_bins % 2 == 0)
        --max_bins;
    return std::max(max_bins, 1);
}

std::vector<int> odd_chunk_sizes(int total_bins, int max_bins)
{
    std::vector<int> sizes;
    int remaining = total_bins;
    while (remaining > max_bins)
    {
        sizes.push_back(max_bins);
        remaining -= max_bins;
    }
    if (remaining % 2 == 1)
        sizes.push_back(remaining);
    else
    {
        sizes.push_back(remaining - 1);
        sizes.push_back(1);
    }
    sizes.erase(std::remove_if(sizes.begin(), sizes.end(),
                               [](int v) { return v <= 0; }),
                sizes.end());
    return sizes;
}

std::vector<BandChunk> plan_chunks(float fmin, float fmax, int nsamp,
                                   double dt, int nfreq, int max_bins)
{
    auto [start_bin, end_bin] = band_bin_range(fmin, fmax, nsamp, dt);
    if (start_bin < 0 || end_bin >= nfreq)
        throw std::runtime_error("band maps outside FFT bins");
    auto sizes = odd_chunk_sizes(end_bin - start_bin + 1,
                                 normalize_chunk_bins(max_bins));
    std::vector<BandChunk> chunks;
    int cursor = start_bin;
    for (int size : sizes)
    {
        chunks.push_back(BandChunk{cursor, cursor + size - 1});
        cursor += size;
    }
    return chunks;
}

std::vector<float> pack_ft_chunk_window_batch(const std::vector<double> &ft,
                                              int first_window,
                                              int batch_windows,
                                              int nr,
                                              int selected_bins,
                                              int selected_start_bin,
                                              const BandChunk &chunk)
{
    int n_bins = chunk.n_bins();
    int local_start = chunk.start_bin - selected_start_bin;
    std::vector<float> packed(static_cast<std::size_t>(batch_windows) *
                              2 * nr * n_bins);
    for (int bw = 0; bw < batch_windows; ++bw)
    {
        int w = first_window + bw;
        for (int r = 0; r < nr; ++r)
        {
            for (int b = 0; b < n_bins; ++b)
            {
                std::size_t src = ((static_cast<std::size_t>(w) * nr + r) *
                                       selected_bins +
                                   local_start + b) *
                                  2;
                std::size_t re = (static_cast<std::size_t>(bw) * 2 * nr + r) *
                                     n_bins +
                                 b;
                std::size_t im = (static_cast<std::size_t>(bw) * 2 * nr + nr + r) *
                                     n_bins +
                                 b;
                packed[re] = static_cast<float>(ft[src]);
                packed[im] = static_cast<float>(ft[src + 1]);
            }
        }
    }
    return packed;
}

void add_inplace(std::vector<float> &dst, const std::vector<float> &src)
{
    for (std::size_t i = 0; i < dst.size(); ++i)
        dst[i] += src[i];
}

void write_npy_f4_2d(const std::string &path, const std::vector<float> &data,
                     int rows, int cols)
{
    std::ofstream out(path, std::ios::binary);
    if (!out)
        throw std::runtime_error("cannot open output: " + path);
    std::ostringstream dict;
    dict << "{'descr': '<f4', 'fortran_order': False, 'shape': ("
         << rows << ", " << cols << "), }";
    std::string header = dict.str();
    std::size_t padding = (16 - ((10 + header.size() + 1) % 16)) % 16;
    header.append(padding, ' ');
    header.push_back('\n');
    std::uint16_t hlen = static_cast<std::uint16_t>(header.size());
    const unsigned char magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0,
                                   static_cast<unsigned char>(hlen & 0xff),
                                   static_cast<unsigned char>((hlen >> 8) & 0xff)};
    out.write(reinterpret_cast<const char *>(magic), sizeof(magic));
    out.write(header.data(), header.size());
    out.write(reinterpret_cast<const char *>(data.data()),
              data.size() * sizeof(float));
}

std::string saclist_stem(const std::string &path)
{
    return fs::path(path).stem().string();
}

bool parse_timestamp_stem(const std::string &stem, std::tm &out)
{
    if (stem.size() != 16 ||
        stem[4] != '-' || stem[7] != '-' || stem[10] != '-' || stem[13] != '-')
        return false;
    try
    {
        int year = std::stoi(stem.substr(0, 4));
        int month = std::stoi(stem.substr(5, 2));
        int day = std::stoi(stem.substr(8, 2));
        int hour = std::stoi(stem.substr(11, 2));
        int minute = std::stoi(stem.substr(14, 2));
        out = {};
        out.tm_year = year - 1900;
        out.tm_mon = month - 1;
        out.tm_mday = day;
        out.tm_hour = hour;
        out.tm_min = minute;
        out.tm_sec = 0;
        out.tm_isdst = 0;
    }
    catch (...)
    {
        return false;
    }
    return true;
}

std::string format_timestamp_utc(std::time_t value, bool include_seconds)
{
    std::tm tm_out{};
    gmtime_r(&value, &tm_out);
    std::ostringstream oss;
    oss << std::setfill('0')
        << std::setw(4) << tm_out.tm_year + 1900 << "-"
        << std::setw(2) << tm_out.tm_mon + 1 << "-"
        << std::setw(2) << tm_out.tm_mday << "-"
        << std::setw(2) << tm_out.tm_hour << "-"
        << std::setw(2) << tm_out.tm_min;
    if (include_seconds)
        oss << "-" << std::setw(2) << tm_out.tm_sec;
    return oss.str();
}

std::string frame_time_label(const ListPlan &plan, int first_window)
{
    std::string stem = saclist_stem(plan.saclist_path);
    if (first_window < 0 || first_window >= static_cast<int>(plan.windows.size()))
        return stem;

    std::tm base_tm{};
    if (!parse_timestamp_stem(stem, base_tm))
        return stem;

    std::time_t base = timegm(&base_tm);
    if (base == static_cast<std::time_t>(-1))
        return stem;

    long long offset_sec =
        std::llround(static_cast<double>(plan.windows[first_window].first) *
                     plan.dt);
    bool include_seconds = (offset_sec % 60) != 0;
    return format_timestamp_utc(base + static_cast<std::time_t>(offset_sec),
                                include_seconds);
}

std::string group_suffix(int group_index)
{
    std::ostringstream oss;
    oss << "g" << std::setfill('0') << std::setw(4) << group_index;
    return oss.str();
}

int output_group_windows(const Options &opt, int nwin)
{
    if (opt.mode == "stack")
        return nwin;
    if (opt.mode == "window")
        return 1;
    return opt.group_windows;
}

std::string output_path_for_list(const Options &opt,
                                 const std::string &saclist_path,
                                 bool multi)
{
    if (!multi)
        return opt.output;
    fs::path out_dir(opt.output);
    fs::create_directories(out_dir);
    return (out_dir / (saclist_stem(saclist_path) + ".npy")).string();
}

std::string output_path_for_method(const Options &opt,
                                   const std::string &saclist_path,
                                   bool multi,
                                   const std::string &method)
{
    if (opt.method != "both")
        return output_path_for_list(opt, saclist_path, multi);

    fs::path output(opt.output);
    if (!multi && output.has_extension())
    {
        fs::path parent = output.parent_path();
        if (parent.empty())
            parent = ".";
        fs::create_directories(parent);
        return (parent / (output.stem().string() + "_" + method +
                          output.extension().string())).string();
    }

    fs::create_directories(output);
    return (output / (saclist_stem(saclist_path) + "_" + method + ".npy")).string();
}

std::string output_path_for_frame_method(const Options &opt,
                                         const ListPlan &plan,
                                         bool multi,
                                         int group_index,
                                         int first_window,
                                         const std::string &method)
{
    bool frame_multi = opt.mode != "stack";
    if (!frame_multi)
        return output_path_for_method(opt, plan.saclist_path, multi, method);

    fs::path out_dir(opt.output);
    fs::create_directories(out_dir);
    std::string stem = frame_time_label(plan, first_window) + "_" +
                       group_suffix(group_index) + "_" + method;
    return (out_dir / (stem + ".npy")).string();
}

std::string bytes_text(std::size_t bytes)
{
    const char *units[] = {"B", "KiB", "MiB", "GiB"};
    double value = static_cast<double>(bytes);
    int unit = 0;
    while (value >= 1024.0 && unit < 3)
    {
        value /= 1024.0;
        ++unit;
    }
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(unit == 0 ? 0 : 2);
    oss << value << " " << units[unit];
    return oss.str();
}

std::size_t estimate_cufft_work(int nsamp, int batch)
{
    cufftHandle plan = 0;
    std::size_t work = 0;
    CUFFT_CHECK(cufftCreate(&plan));
    CUFFT_CHECK(cufftSetAutoAllocation(plan, 0));
    cufftResult r = cufftMakePlan1d(plan, nsamp, CUFFT_D2Z, batch, &work);
    cufftDestroy(plan);
    if (r != CUFFT_SUCCESS)
    {
        std::ostringstream oss;
        oss << "cufftMakePlan1d failed during estimate, code=" << static_cast<int>(r);
        throw std::runtime_error(oss.str());
    }
    return work;
}

MemoryPlan estimate_memory(int nr, int master_nr, int nk,
                           int nwin, int nsamp, int nfreq,
                           int selected_bins, int max_chunk_bins,
                           int cov_window_batch,
                           std::size_t cufft_work)
{
    int batch = nr * nwin;
    std::size_t master_tau_size = static_cast<std::size_t>(master_nr) * nk * nk;
    std::size_t active_tau_size = static_cast<std::size_t>(nr) * nk * nk;
    std::size_t grid_receiver = static_cast<std::size_t>(nk) * nk * nr;
    std::size_t mat = static_cast<std::size_t>(max_chunk_bins) * nr * nr;
    std::size_t n_bins = max_chunk_bins;

    MemoryPlan p;
    p.master_tau = align_up(sizeof(float) * master_tau_size);
    p.active_tau = align_up(sizeof(float) * active_tau_size);
    p.receiver_map = align_up(sizeof(int) * nr);
    p.permanent =
        align_up(sizeof(float) * master_nr) +
        align_up(sizeof(float) * master_nr) +
        p.master_tau +
        p.receiver_map +
        p.active_tau;
    p.cufft_work = cufft_work;
    p.fft =
        align_up(sizeof(cufftDoubleReal) * static_cast<std::size_t>(batch) * nsamp) +
        align_up(sizeof(cufftDoubleComplex) * static_cast<std::size_t>(batch) * nfreq) +
        align_up(cufft_work);
    p.steering =
        align_up(sizeof(float) * n_bins * grid_receiver) * 2;
    p.covariance =
        align_up(sizeof(float) * static_cast<std::size_t>(cov_window_batch) * 2 * nr * n_bins) +
        align_up(sizeof(float) * 2 * static_cast<std::size_t>(nr) * nr * n_bins) +
        align_up(sizeof(float) * n_bins * nr);
    p.inverse =
        align_up(sizeof(float) * 2 * static_cast<std::size_t>(nr) * nr * n_bins) +
        align_up(sizeof(cuComplex) * mat) +
        align_up(sizeof(cuComplex) * mat) +
        align_up(sizeof(float) * mat) +
        align_up(sizeof(float) * mat) +
        align_up(sizeof(int) * n_bins * nr) +
        align_up(sizeof(int) * n_bins) +
        align_up(sizeof(cuComplex *) * n_bins) * 2;
    p.beamforming =
        align_up(sizeof(float) * mat) * 2 +
        align_up(sizeof(float) * static_cast<std::size_t>(nk) * nk) +
        p.steering;
    p.chunk_combined =
        align_up(sizeof(float) * static_cast<std::size_t>(cov_window_batch) * 2 * nr * n_bins) +
        align_up(sizeof(float) * 2 * static_cast<std::size_t>(nr) * nr * n_bins) +
        align_up(sizeof(float) * n_bins * nr) +
        align_up(sizeof(cuComplex) * mat) +
        align_up(sizeof(cuComplex) * mat) +
        align_up(sizeof(float) * mat) +
        align_up(sizeof(float) * mat) +
        align_up(sizeof(float) * static_cast<std::size_t>(nk) * nk) +
        p.steering +
        align_up(sizeof(int) * n_bins * nr) +
        align_up(sizeof(int) * n_bins) +
        align_up(sizeof(cuComplex *) * n_bins) * 2;
    p.arena = p.permanent + std::max({p.fft, p.steering, p.chunk_combined});
    (void)selected_bins;
    return p;
}

std::pair<float, float> parse_pair_float(const std::string &text,
                                         const std::string &name)
{
    auto slash = text.find('/');
    if (slash == std::string::npos)
        throw std::runtime_error(name + " must use a/b");
    return {std::stof(text.substr(0, slash)), std::stof(text.substr(slash + 1))};
}

std::pair<double, double> parse_pair_double(const std::string &text,
                                            const std::string &name)
{
    auto slash = text.find('/');
    if (slash == std::string::npos)
        throw std::runtime_error(name + " must use a/b");
    return {std::stod(text.substr(0, slash)), std::stod(text.substr(slash + 1))};
}

std::vector<int> parse_int_list(const std::string &text,
                                const std::string &name)
{
    std::vector<int> values;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ','))
    {
        item = trim(item);
        if (item.empty())
            continue;
        std::size_t used = 0;
        int value = 0;
        try
        {
            value = std::stoi(item, &used);
        }
        catch (const std::exception &err)
        {
            throw std::runtime_error(name + " must be comma-separated integers: " + err.what());
        }
        if (used != item.size())
            throw std::runtime_error(name + " must be comma-separated integers");
        values.push_back(value);
    }
    if (values.empty())
        throw std::runtime_error(name + " requires at least one integer");
    return values;
}

void lowercase_inplace(std::string &text)
{
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
}

Options parse_args(int argc, char **argv)
{
    Options opt;
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        auto need = [&](const std::string &flag) -> std::string {
            if (i + 1 >= argc)
                throw std::runtime_error(flag + " requires a value");
            return argv[++i];
        };
        if (arg == "-I" || arg == "--saclist")
            opt.saclist = need(arg);
        else if (arg == "--saclist-list")
            opt.saclist_list = need(arg);
        else if (arg == "--geometry")
            opt.geometry = need(arg);
        else if (arg == "-O" || arg == "--output")
            opt.output = need(arg);
        else if (arg == "--plan-only")
            opt.plan_only = true;
        else if (arg == "-B" || arg == "--band")
        {
            auto p = parse_pair_float(need(arg), arg);
            opt.fmin = p.first;
            opt.fmax = p.second;
        }
        else if (arg == "-W" || arg == "--window")
        {
            auto p = parse_pair_double(need(arg), arg);
            opt.win_len = p.first;
            opt.shift_len = p.second;
        }
        else if (arg == "--taper")
            opt.taper = trim(need(arg));
        else if (arg == "--tukey-alpha")
            opt.taper_alpha = std::stod(need(arg));
        else if (arg == "--record-samples")
            opt.record_samples = std::stoi(need(arg));
        else if (arg == "--record-seconds")
            opt.record_seconds = std::stod(need(arg));
        else if (arg == "-S" || arg == "--slowness")
        {
            auto p = parse_pair_float(need(arg), arg);
            opt.s_range = p.first;
            opt.s_res = p.second;
        }
        else if (arg == "--freq-chunk-bins")
            opt.freq_chunk_bins = std::stoi(need(arg));
        else if (arg == "--method")
            opt.method = trim(need(arg));
        else if (arg == "--mode")
            opt.mode = trim(need(arg));
        else if (arg == "--group-windows")
            opt.group_windows = std::stoi(need(arg));
        else if (arg == "--skip-step")
            opt.skip_steps = parse_int_list(need(arg), arg);
        else if (arg == "-G" || arg == "--gpu")
            opt.gpu = std::stoi(need(arg));
        else if (arg == "--dl")
            opt.dl = std::stof(need(arg));
        else if (arg == "--steering-cache")
            opt.steering_cache = trim(need(arg));
        else if (arg == "--host-steering-max-gb")
            opt.host_steering_max_gb = std::stod(need(arg));
        else if (arg == "--profile")
            opt.profile = true;
        else if (arg == "-h" || arg == "--help")
        {
            std::cout << "fastbf_native (-I saclist.tsv | --saclist-list lists.txt) "
                         "[--geometry geometry.tsv] [-O out.npy_or_dir] "
                         "-B fmin/fmax -W win/shift -S range/res "
                         "[--mode stack|grouped|window] "
                         "[--group-windows N] "
                         "[--skip-step i,j] "
                         "[--taper hann|tukey|none] [--tukey-alpha A] "
                         "[--method capon|bartlett|both] "
                         "[--steering-cache auto|none|host|device] "
                         "[--plan-only] [--profile]\n";
            std::exit(0);
        }
        else
            throw std::runtime_error("unknown argument: " + arg);
    }
    if (opt.saclist.empty() == opt.saclist_list.empty())
        throw std::runtime_error("provide exactly one of --saclist or --saclist-list");
    if (!opt.plan_only && (opt.geometry.empty() || opt.output.empty()))
        throw std::runtime_error("missing required input/output paths");
    if (!std::isfinite(opt.fmin) || !std::isfinite(opt.fmax) || opt.fmin >= opt.fmax)
        throw std::runtime_error("invalid band");
    lowercase_inplace(opt.method);
    lowercase_inplace(opt.mode);
    lowercase_inplace(opt.steering_cache);
    normalize_taper_options(opt);
    if (opt.method != "capon" && opt.method != "bartlett" && opt.method != "both")
        throw std::runtime_error("fastbf_native supports method capon, bartlett, or both");
    if (opt.mode != "stack" && opt.mode != "grouped" && opt.mode != "window")
        throw std::runtime_error("--mode must be stack, grouped, or window");
    if (opt.mode == "window")
        opt.group_windows = 1;
    if (opt.mode == "grouped" && opt.group_windows <= 0)
        throw std::runtime_error("--mode grouped requires --group-windows N");
    if (opt.mode == "stack" && opt.group_windows < 0)
        throw std::runtime_error("--group-windows must be non-negative");
    for (int step : opt.skip_steps)
    {
        if (step < 0)
            throw std::runtime_error("--skip-step values must be non-negative");
    }
    if (opt.record_samples < 0 || opt.record_seconds < 0.0)
        throw std::runtime_error("record length overrides must be non-negative");
    if (opt.record_samples > 0 && opt.record_seconds > 0.0)
        throw std::runtime_error("use only one of --record-samples or --record-seconds");
    if (opt.steering_cache != "auto" && opt.steering_cache != "host" &&
        opt.steering_cache != "device" && opt.steering_cache != "none")
        throw std::runtime_error("--steering-cache must be auto, host, device, or none");
    return opt;
}

int slowness_count(float s_range, float s_res)
{
    int count = 0;
    for (float s = -s_range; s <= s_range + 1.0e-9f; s += s_res)
        ++count;
    return count;
}

void print_memory_plan(const MemoryPlan &p)
{
    std::cerr << "[fastbf_native] memory estimate\n";
    std::cerr << "  permanent          " << bytes_text(p.permanent) << "\n";
    std::cerr << "    master tau       " << bytes_text(p.master_tau) << "\n";
    std::cerr << "    active tau       " << bytes_text(p.active_tau) << "\n";
    std::cerr << "    receiver map     " << bytes_text(p.receiver_map) << "\n";
    std::cerr << "  fft scratch         " << bytes_text(p.fft)
              << " (cuFFT work " << bytes_text(p.cufft_work) << ")\n";
    std::cerr << "  steering scratch    " << bytes_text(p.steering) << "\n";
    std::cerr << "  covariance stage    " << bytes_text(p.covariance) << "\n";
    std::cerr << "  inverse stage       " << bytes_text(p.inverse) << "\n";
    std::cerr << "  beamforming stage   " << bytes_text(p.beamforming) << "\n";
    std::cerr << "  chunk combined      " << bytes_text(p.chunk_combined) << "\n";
    std::cerr << "  arena allocation    " << bytes_text(p.arena) << "\n";
}

void print_machine_plan(const WorkspacePlan &workspace,
                        const SteeringPlan &steering,
                        int cov_window_batch,
                        int nk,
                        std::size_t free_bytes,
                        std::size_t total_bytes)
{
    std::cout
        << "FASTBF_PLAN"
        << "\tarena_bytes=" << workspace.memory.arena
        << "\tpermanent_bytes=" << workspace.memory.permanent
        << "\tmaster_tau_bytes=" << workspace.memory.master_tau
        << "\tactive_tau_bytes=" << workspace.memory.active_tau
        << "\treceiver_map_bytes=" << workspace.memory.receiver_map
        << "\tfft_bytes=" << workspace.memory.fft
        << "\tcufft_work_bytes=" << workspace.memory.cufft_work
        << "\tsteering_bytes=" << workspace.memory.steering
        << "\tcovariance_bytes=" << workspace.memory.covariance
        << "\tinverse_bytes=" << workspace.memory.inverse
        << "\tbeamforming_bytes=" << workspace.memory.beamforming
        << "\tchunk_combined_bytes=" << workspace.memory.chunk_combined
        << "\tglobal_steering_bytes=" << steering.bytes
        << "\tglobal_steering_mode=" << steering.mode
        << "\tfree_bytes=" << free_bytes
        << "\ttotal_bytes=" << total_bytes
        << "\tcov_window_batch=" << cov_window_batch
        << "\tmax_nr=" << workspace.max_nr
        << "\tmax_nwin=" << workspace.max_nwin
        << "\tmax_nsamp=" << workspace.max_nsamp
        << "\tmax_nfreq=" << workspace.max_nfreq
        << "\tmax_selected_bins=" << workspace.max_selected_bins
        << "\tmax_chunk_bins=" << workspace.max_chunk_bins
        << "\tmaster_nr=" << workspace.master_nr
        << "\tnk=" << nk
        << "\n";
}

std::vector<std::string> collect_saclist_paths(const Options &opt)
{
    if (!opt.saclist.empty())
        return {opt.saclist};
    return read_saclist_manifest(opt.saclist_list);
}

std::vector<ListPlan> build_all_list_plans(const std::vector<std::string> &paths,
                                           const Options &opt)
{
    std::vector<ListPlan> plans;
    plans.reserve(paths.size());
    for (const auto &path : paths)
        plans.push_back(build_list_plan(path, opt));
    return plans;
}

void validate_single_fft_shape(const std::vector<ListPlan> &plans)
{
    if (plans.empty())
        throw std::runtime_error("no list plans were built");
    int nsamp = plans.front().nsamp;
    for (const auto &plan : plans)
    {
        if (plan.nsamp != nsamp)
        {
            throw std::runtime_error(
                "all saclists in one native workspace must share nsamp; "
                "use --record-samples/--record-seconds or split commands");
        }
    }
}

WorkspacePlan plan_workspace(const std::vector<ListPlan> &plans,
                             int nk,
                             const Options &opt,
                             int cov_window_batch,
                             int master_nr)
{
    WorkspacePlan result;
    result.master_nr = master_nr;

    for (const auto &plan : plans)
    {
        auto chunks = plan_chunks(opt.fmin, opt.fmax, plan.nsamp, plan.dt,
                                  plan.nfreq, opt.freq_chunk_bins);
        int selected_bins = chunks.back().end_bin - chunks.front().start_bin + 1;
        int max_chunk_bins = std::max_element(
            chunks.begin(), chunks.end(),
            [](const BandChunk &a, const BandChunk &b) {
                return a.n_bins() < b.n_bins();
            })->n_bins();
        result.max_nr = std::max(result.max_nr, plan.nr);
        result.max_nwin = std::max(result.max_nwin, plan.nwin);
        result.max_nsamp = std::max(result.max_nsamp, plan.nsamp);
        result.max_nfreq = std::max(result.max_nfreq, plan.nfreq);
        result.max_selected_bins = std::max(result.max_selected_bins, selected_bins);
        result.max_chunk_bins = std::max(result.max_chunk_bins, max_chunk_bins);
    }
    result.max_cufft_work = estimate_cufft_work(
        result.max_nsamp, result.max_nr * result.max_nwin);
    result.memory = estimate_memory(
        result.max_nr, master_nr, nk, result.max_nwin, result.max_nsamp,
        result.max_nfreq, result.max_selected_bins, result.max_chunk_bins,
        std::min(cov_window_batch, result.max_nwin),
        result.max_cufft_work);
    result.source_saclist = "max-envelope";
    return result;
}

int choose_cov_window_batch(const std::vector<ListPlan> &plans,
                            int nk,
                            const Options &opt,
                            std::size_t free_bytes,
                            int master_nr,
                            WorkspacePlan &workspace)
{
    for (int candidate : {8, 4, 1})
    {
        WorkspacePlan trial = plan_workspace(plans, nk, opt, candidate, master_nr);
        if (trial.memory.arena <= free_bytes)
        {
            workspace = std::move(trial);
            return candidate;
        }
    }
    workspace = plan_workspace(plans, nk, opt, 1, master_nr);
    return 1;
}

bool same_receiver_order(const ListPlan &a, const ListPlan &b)
{
    if (a.rows.size() != b.rows.size())
        return false;
    for (std::size_t i = 0; i < a.rows.size(); ++i)
        if (a.rows[i].nsl_id != b.rows[i].nsl_id)
            return false;
    return true;
}

bool steering_reusable_across_lists(const std::vector<ListPlan> &plans,
                                    std::string &reason)
{
    if (plans.empty())
    {
        reason = "no lists";
        return false;
    }
    const auto &ref = plans.front();
    for (const auto &plan : plans)
    {
        if (plan.nr != ref.nr)
        {
            reason = "receiver count differs";
            return false;
        }
        if (plan.nsamp != ref.nsamp || plan.nfreq != ref.nfreq)
        {
            reason = "FFT shape differs";
            return false;
        }
        double tol = std::max(1.0e-12, std::abs(ref.dt) * 1.0e-9);
        if (std::abs(plan.dt - ref.dt) > tol)
        {
            reason = "sample interval differs";
            return false;
        }
        if (!same_receiver_order(ref, plan))
        {
            reason = "receiver order differs";
            return false;
        }
    }
    reason = "shared receiver order and FFT shape";
    return true;
}

SteeringPlan plan_global_steering_cache(const std::vector<ListPlan> &plans,
                                        int nk,
                                        const Options &opt,
                                        const WorkspacePlan &workspace,
                                        std::size_t free_bytes)
{
    SteeringPlan plan;
    std::string reason;
    bool reusable = steering_reusable_across_lists(plans, reason);
    plan.reusable = reusable;
    plan.reason = reason;
    if (!reusable)
    {
        if (opt.steering_cache == "host" || opt.steering_cache == "device")
            throw std::runtime_error("global steering cache requested but " + reason);
        return plan;
    }

    const auto &ref = plans.front();
    plan.chunks = plan_chunks(opt.fmin, opt.fmax, ref.nsamp, ref.dt,
                              ref.nfreq, opt.freq_chunk_bins);
    plan.selected_start = plan.chunks.front().start_bin;
    plan.selected_end = plan.chunks.back().end_bin;
    plan.selected_bins = plan.selected_end - plan.selected_start + 1;
    plan.values = static_cast<std::size_t>(plan.selected_bins) *
                  nk * nk * ref.nr;
    plan.bytes = plan.values * sizeof(float) * 2;

    if (opt.steering_cache == "none")
    {
        plan.mode = "none";
        plan.bytes = 0;
        return plan;
    }
    if (opt.steering_cache == "device")
    {
        if (workspace.memory.arena + plan.bytes >= free_bytes)
            throw std::runtime_error("device steering cache does not fit current GPU memory");
        plan.mode = "device";
        return plan;
    }
    if (opt.steering_cache == "host")
    {
        plan.mode = "host";
        return plan;
    }

    if (plans.size() <= 1)
    {
        plan.mode = "none";
        plan.bytes = 0;
        plan.reason = "single list has no cross-list steering reuse";
        return plan;
    }

    constexpr std::size_t kDeviceCacheMargin =
        static_cast<std::size_t>(1024) * 1024 * 1024;
    if (workspace.memory.arena + plan.bytes + kDeviceCacheMargin < free_bytes)
    {
        plan.mode = "device";
        return plan;
    }

    std::size_t host_limit =
        static_cast<std::size_t>(opt.host_steering_max_gb *
                                 1024.0 * 1024.0 * 1024.0);
    if (plan.bytes <= host_limit)
    {
        plan.mode = "host";
        return plan;
    }

    plan.mode = "none";
    plan.bytes = 0;
    plan.reason = "global steering cache exceeds device memory and host threshold";
    return plan;
}

CufftPlan make_fft_plan(int nsamp, int batch, void *work_area)
{
    CufftPlan plan;
    std::size_t work = 0;
    CUFFT_CHECK(cufftCreate(&plan.handle));
    CUFFT_CHECK(cufftSetAutoAllocation(plan.handle, 0));
    CUFFT_CHECK(cufftMakePlan1d(plan.handle, nsamp, CUFFT_D2Z, batch, &work));
    CUFFT_CHECK(cufftSetWorkArea(plan.handle, work_area));
    return plan;
}

PersistentDevice allocate_persistent(DeviceArena &arena, int max_nr,
                                     int master_nr, int nk)
{
    PersistentDevice p;
    p.master_rx = arena.alloc<float>(master_nr, "master.rx");
    p.master_ry = arena.alloc<float>(master_nr, "master.ry");
    p.master_tau = arena.alloc<float>(
        static_cast<std::size_t>(master_nr) * nk * nk, "master.tau");
    p.receiver_master_index = arena.alloc<int>(max_nr, "receiver.master_index");
    p.tau = arena.alloc<float>(
        static_cast<std::size_t>(max_nr) * nk * nk, "active.tau");
    p.scratch_start = arena.cursor();
    return p;
}

std::vector<double> run_fft_selected(DeviceArena &arena,
                                     const PersistentDevice &persistent,
                                     CufftPlan &fft_plan,
                                     const std::vector<double> &window_stack,
                                     int nr, int nwin, int nsamp,
                                     int max_batch, int max_nsamp, int max_nfreq,
                                     int start_bin, int end_bin,
                                     std::size_t cufft_work,
                                     ProfileStats *profile)
{
    auto profile_start = Clock::now();
    int batch = nr * nwin;
    if (nsamp != max_nsamp)
        throw std::runtime_error("current list nsamp does not match workspace FFT plan");
    if (batch > max_batch)
        throw std::runtime_error("current list batch exceeds workspace FFT plan");
    int nfreq = max_nfreq;
    int selected_bins = end_bin - start_bin + 1;
    arena.reset(persistent.scratch_start);
    auto *d_xt = arena.alloc<cufftDoubleReal>(
        static_cast<std::size_t>(max_batch) * max_nsamp, "fft.xt");
    auto *d_ft = arena.alloc<cufftDoubleComplex>(
        static_cast<std::size_t>(max_batch) * max_nfreq, "fft.ft_full");
    arena.alloc_bytes(cufft_work, "fft.work");

    CUDA_CHECK(cudaMemset(d_xt, 0,
                          sizeof(cufftDoubleReal) *
                              static_cast<std::size_t>(max_batch) * max_nsamp));
    CUDA_CHECK(cudaMemcpy(d_xt, window_stack.data(),
                          sizeof(double) * window_stack.size(),
                          cudaMemcpyHostToDevice));
    CUFFT_CHECK(cufftExecD2Z(fft_plan.handle, d_xt, d_ft));

    std::vector<double> selected(static_cast<std::size_t>(batch) *
                                 selected_bins * 2);
    CUDA_CHECK(cudaMemcpy2D(
        selected.data(),
        sizeof(cufftDoubleComplex) * static_cast<std::size_t>(selected_bins),
        d_ft + start_bin,
        sizeof(cufftDoubleComplex) * static_cast<std::size_t>(nfreq),
        sizeof(cufftDoubleComplex) * static_cast<std::size_t>(selected_bins),
        batch,
        cudaMemcpyDeviceToHost));
    if (profile)
        profile->fft_ms += elapsed_ms(profile_start);
    return selected;
}

void precompute_host_steering(DeviceArena &arena,
                              const PersistentDevice &persistent,
                              std::vector<float> &host_re,
                              std::vector<float> &host_im,
                              const std::vector<BandChunk> &chunks,
                              int selected_start_bin,
                              int nr, int nk, int nsamp, float delta)
{
    std::size_t grid_receiver = static_cast<std::size_t>(nk) * nk * nr;
    for (const auto &chunk : chunks)
    {
        arena.reset(persistent.scratch_start);
        auto *d_re = arena.alloc<float>(static_cast<std::size_t>(chunk.n_bins()) *
                                            grid_receiver,
                                        "steering.re");
        auto *d_im = arena.alloc<float>(static_cast<std::size_t>(chunk.n_bins()) *
                                            grid_receiver,
                                        "steering.im");
        std::size_t total = static_cast<std::size_t>(chunk.n_bins()) * grid_receiver;
        precompute_steering_chunk<<<(total + 255) / 256, 256>>>(
            d_re, d_im, persistent.tau, nr, nk, chunk.n_bins(),
            nsamp, delta, chunk.fave(), chunk.find());
        CUDA_CHECK(cudaGetLastError());
        std::size_t dst = static_cast<std::size_t>(chunk.start_bin - selected_start_bin) *
                          grid_receiver;
        CUDA_CHECK(cudaMemcpy(host_re.data() + dst, d_re,
                              sizeof(float) * total, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host_im.data() + dst, d_im,
                              sizeof(float) * total, cudaMemcpyDeviceToHost));
    }
}

void upload_master_geometry_and_tau(const GeometryTable &geometry,
                                    const PersistentDevice &persistent,
                                    int nk, float s_range, float s_res)
{
    int master_nr = static_cast<int>(geometry.nsl_ids.size());
    CUDA_CHECK(cudaMemcpy(persistent.master_rx, geometry.rx.data(),
                          sizeof(float) * master_nr, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(persistent.master_ry, geometry.ry.data(),
                          sizeof(float) * master_nr, cudaMemcpyHostToDevice));
    precompute_tau_grid<<<(static_cast<std::size_t>(master_nr) * nk * nk + 255) / 256, 256>>>(
        persistent.master_tau, persistent.master_rx, persistent.master_ry,
        master_nr, nk,
        -s_range, s_res);
    CUDA_CHECK(cudaGetLastError());
}

void select_tau_for_rows(const GeometryTable &geometry,
                         const std::vector<SacRow> &rows,
                         const PersistentDevice &persistent,
                         int nr, int nk,
                         ProfileStats *profile)
{
    auto profile_start = Clock::now();
    auto indices = receiver_master_indices(geometry, rows);
    int master_nr = static_cast<int>(geometry.nsl_ids.size());
    CUDA_CHECK(cudaMemcpy(persistent.receiver_master_index, indices.data(),
                          sizeof(int) * indices.size(), cudaMemcpyHostToDevice));
    gather_tau_subset<<<(static_cast<std::size_t>(nr) * nk * nk + 255) / 256, 256>>>(
        persistent.tau, persistent.master_tau, persistent.receiver_master_index,
        nr, master_nr, nk);
    CUDA_CHECK(cudaGetLastError());
    if (profile)
    {
        CUDA_CHECK(cudaDeviceSynchronize());
        profile->select_tau_ms += elapsed_ms(profile_start);
    }
}

void precompute_device_steering(GlobalSteeringCache &cache,
                                const PersistentDevice &persistent,
                                const std::vector<BandChunk> &chunks,
                                int selected_start_bin,
                                int nr, int nk, int nsamp, float delta)
{
    std::size_t grid_receiver = static_cast<std::size_t>(nk) * nk * nr;
    CUDA_CHECK(cudaMalloc(&cache.device_re, sizeof(float) * cache.values));
    CUDA_CHECK(cudaMalloc(&cache.device_im, sizeof(float) * cache.values));
    for (const auto &chunk : chunks)
    {
        std::size_t total = static_cast<std::size_t>(chunk.n_bins()) * grid_receiver;
        std::size_t dst = static_cast<std::size_t>(chunk.start_bin - selected_start_bin) *
                          grid_receiver;
        precompute_steering_chunk<<<(total + 255) / 256, 256>>>(
            cache.device_re + dst, cache.device_im + dst,
            persistent.tau, nr, nk, chunk.n_bins(),
            nsamp, delta, chunk.fave(), chunk.find());
        CUDA_CHECK(cudaGetLastError());
    }
}

GlobalSteeringCache build_global_steering_cache(DeviceArena &arena,
                                                const PersistentDevice &persistent,
                                                const SteeringPlan &plan,
                                                int nr, int nk, int nsamp,
                                                float delta)
{
    GlobalSteeringCache cache;
    cache.mode = plan.mode;
    cache.selected_start = plan.selected_start;
    cache.values = plan.values;
    if (plan.mode == "host")
    {
        cache.host_re.resize(plan.values);
        cache.host_im.resize(plan.values);
        precompute_host_steering(arena, persistent, cache.host_re, cache.host_im,
                                 plan.chunks, plan.selected_start,
                                 nr, nk, nsamp, delta);
    }
    else if (plan.mode == "device")
    {
        precompute_device_steering(cache, persistent, plan.chunks,
                                   plan.selected_start, nr, nk, nsamp, delta);
    }
    return cache;
}

void run_beamforming_chunk(DeviceArena &arena, const PersistentDevice &persistent,
                           CublasHandle &blas, const std::vector<double> &ft_selected,
                           int selected_bins, int selected_start_bin,
                           int cov_window_batch,
                           const float *host_steer_re, const float *host_steer_im,
                           const float *device_steer_re, const float *device_steer_im,
                           int host_steer_start_bin,
                           const BandChunk &chunk, int nwin, int nr, int nk,
                           int first_window, int nsamp, float delta, float dl,
                           bool do_capon, bool do_bartlett,
                           std::vector<float> *capon_total,
                           std::vector<float> *bartlett_total,
                           ProfileStats *profile)
{
    if (!do_capon && !do_bartlett)
        throw std::runtime_error("beamforming chunk has no requested outputs");
    auto chunk_start = Clock::now();
    auto setup_start = Clock::now();
    int n_bins = chunk.n_bins();
    std::size_t mat = static_cast<std::size_t>(n_bins) * nr * nr;
    std::size_t grid_receiver = static_cast<std::size_t>(nk) * nk * nr;
    arena.reset(persistent.scratch_start);

    int max_cov_batch = std::min(cov_window_batch, nwin);
    auto *d_ft = arena.alloc<float>(
        static_cast<std::size_t>(max_cov_batch) * 2 * nr * n_bins, "chunk.ft");
    auto *d_sm = arena.alloc<float>(
        2 * static_cast<std::size_t>(nr) * nr * n_bins, "chunk.sm");
    auto *d_w = arena.alloc<float>(static_cast<std::size_t>(n_bins) * nr, "chunk.w");
    auto *d_tfk = arena.alloc<float>(static_cast<std::size_t>(nk) * nk, "chunk.tfk");
    cuComplex *d_tx = nullptr;
    cuComplex *d_inv = nullptr;
    float *d_ismr = nullptr;
    float *d_ismi = nullptr;
    int *d_piv = nullptr;
    int *d_info = nullptr;
    cuComplex **d_A = nullptr;
    cuComplex **d_C = nullptr;
    float *d_steer_re = nullptr;
    float *d_steer_im = nullptr;
    if (do_capon)
    {
        d_tx = arena.alloc<cuComplex>(mat, "chunk.tx");
        d_inv = arena.alloc<cuComplex>(mat, "chunk.inv");
        d_ismr = arena.alloc<float>(mat, "chunk.ismr");
        d_ismi = arena.alloc<float>(mat, "chunk.ismi");
        d_piv = arena.alloc<int>(static_cast<std::size_t>(n_bins) * nr, "chunk.piv");
        d_info = arena.alloc<int>(n_bins, "chunk.info");
        d_A = arena.alloc<cuComplex *>(n_bins, "chunk.Aptr");
        d_C = arena.alloc<cuComplex *>(n_bins, "chunk.Cptr");
    }

    CUDA_CHECK(cudaMemset(d_sm, 0, sizeof(float) * 2 * nr * nr * n_bins));
    CUDA_CHECK(cudaMemset(d_w, 0, sizeof(float) * n_bins * nr));
    CUDA_CHECK(cudaMemset(d_tfk, 0, sizeof(float) * nk * nk));
    if (profile)
    {
        CUDA_CHECK(cudaDeviceSynchronize());
        profile->chunk_setup_ms += elapsed_ms(setup_start);
    }

    auto steering_start = Clock::now();
    std::size_t steer_total = static_cast<std::size_t>(n_bins) * grid_receiver;
    std::size_t steer_src = static_cast<std::size_t>(chunk.start_bin - host_steer_start_bin) *
                            grid_receiver;
    if (device_steer_re && device_steer_im)
    {
        d_steer_re = const_cast<float *>(device_steer_re + steer_src);
        d_steer_im = const_cast<float *>(device_steer_im + steer_src);
    }
    else
    {
        d_steer_re = arena.alloc<float>(
            static_cast<std::size_t>(n_bins) * grid_receiver, "chunk.steer_re");
        d_steer_im = arena.alloc<float>(
            static_cast<std::size_t>(n_bins) * grid_receiver, "chunk.steer_im");
    }

    if (host_steer_re && host_steer_im)
    {
        CUDA_CHECK(cudaMemcpy(d_steer_re, host_steer_re + steer_src,
                              sizeof(float) * steer_total, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_steer_im, host_steer_im + steer_src,
                              sizeof(float) * steer_total, cudaMemcpyHostToDevice));
    }
    else if (!device_steer_re || !device_steer_im)
    {
        precompute_steering_chunk<<<(steer_total + 255) / 256, 256>>>(
            d_steer_re, d_steer_im, persistent.tau, nr, nk, n_bins,
            nsamp, delta, chunk.fave(), chunk.find());
        CUDA_CHECK(cudaGetLastError());
    }
    if (profile)
    {
        CUDA_CHECK(cudaDeviceSynchronize());
        profile->steering_ms += elapsed_ms(steering_start);
    }

    dim3 block(8, 8, 1);
    dim3 grid((nr + block.x - 1) / block.x, (nr + block.y - 1) / block.y, 1);
    for (int first = 0; first < nwin; first += max_cov_batch)
    {
        int batch_windows = std::min(max_cov_batch, nwin - first);
        auto pack_start = Clock::now();
        auto packed = pack_ft_chunk_window_batch(
            ft_selected, first_window + first, batch_windows, nr,
            selected_bins, selected_start_bin, chunk);
        if (profile)
            profile->pack_ms += elapsed_ms(pack_start);
        auto h2d_start = Clock::now();
        CUDA_CHECK(cudaMemcpy(d_ft, packed.data(),
                              sizeof(float) * packed.size(),
                              cudaMemcpyHostToDevice));
        if (profile)
            profile->h2d_ms += elapsed_ms(h2d_start);
        auto cov_start = Clock::now();
        accumulate_sm<<<grid, block>>>(d_ft, batch_windows, nr,
                                       chunk.fave(), d_sm);
        CUDA_CHECK(cudaGetLastError());
        if (profile)
        {
            CUDA_CHECK(cudaDeviceSynchronize());
            profile->covariance_ms += elapsed_ms(cov_start);
        }
    }
    auto cov_post_start = Clock::now();
    int sm_total = 2 * nr * nr * n_bins;
    scale_sm<<<(sm_total + 255) / 256, 256>>>(
        d_sm, sm_total, 1.0f / static_cast<float>(nwin));
    CUDA_CHECK(cudaGetLastError());

    dim3 block_fk(8, 8, 1);
    dim3 grid_fk((nk + block_fk.x - 1) / block_fk.x,
                 (nk + block_fk.y - 1) / block_fk.y, 1);
    auto copy_tfk = [&](std::vector<float> *target) {
        auto d2h_start = Clock::now();
        std::vector<float> chunk_tfk(static_cast<std::size_t>(nk) * nk);
        CUDA_CHECK(cudaMemcpy(chunk_tfk.data(), d_tfk,
                              sizeof(float) * chunk_tfk.size(),
                              cudaMemcpyDeviceToHost));
        if (profile)
            profile->d2h_ms += elapsed_ms(d2h_start);
        add_inplace(*target, chunk_tfk);
    };

    dim3 grid_w((nr + block.x - 1) / block.x, (n_bins + block.y - 1) / block.y, 1);
    compute_w<<<grid_w, block>>>(d_sm, chunk.fave(), nr, d_w);
    compute_sm_weighted<<<grid, block>>>(d_sm, d_w, chunk.fave(), nr);
    CUDA_CHECK(cudaGetLastError());
    if (profile)
    {
        CUDA_CHECK(cudaDeviceSynchronize());
        profile->covariance_ms += elapsed_ms(cov_post_start);
    }

    if (do_bartlett)
    {
        auto fk_start = Clock::now();
        compute_bartlett_fk_steering<<<grid_fk, block_fk>>>(
            d_sm, d_tfk, nk, d_steer_re, d_steer_im, nr, n_bins);
        CUDA_CHECK(cudaGetLastError());
        if (profile)
        {
            CUDA_CHECK(cudaDeviceSynchronize());
            profile->fk_ms += elapsed_ms(fk_start);
        }
        copy_tfk(bartlett_total);
        if (do_capon)
            CUDA_CHECK(cudaMemset(d_tfk, 0, sizeof(float) * nk * nk));
    }

    if (do_capon)
    {
        auto inverse_start = Clock::now();
        apply_diagonal_loading<<<(n_bins + 255) / 256, 256>>>(
            d_sm, n_bins, nr, nsamp, dl);
        sm_interleaved_to_complex<<<(mat + 255) / 256, 256>>>(
            d_sm, d_tx, n_bins, nr);
        CUDA_CHECK(cudaGetLastError());

        std::vector<cuComplex *> h_A(n_bins);
        std::vector<cuComplex *> h_C(n_bins);
        for (int b = 0; b < n_bins; ++b)
        {
            h_A[b] = d_tx + static_cast<std::size_t>(b) * nr * nr;
            h_C[b] = d_inv + static_cast<std::size_t>(b) * nr * nr;
        }
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), sizeof(cuComplex *) * n_bins,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), sizeof(cuComplex *) * n_bins,
                              cudaMemcpyHostToDevice));
        CUBLAS_CHECK(cublasCgetrfBatched(blas.handle, nr, d_A, nr,
                                         d_piv, d_info, n_bins));
        CUBLAS_CHECK(cublasCgetriBatched(blas.handle, nr,
                                         const_cast<const cuComplex **>(d_A), nr,
                                         d_piv, d_C, nr, d_info, n_bins));

        complex_to_split_real_imag<<<(mat + 255) / 256, 256>>>(
            d_inv, d_ismr, d_ismi, static_cast<int>(mat));
        CUDA_CHECK(cudaGetLastError());
        if (profile)
        {
            CUDA_CHECK(cudaDeviceSynchronize());
            profile->inverse_ms += elapsed_ms(inverse_start);
        }

        auto fk_start = Clock::now();
        compute_fk_steering<<<grid_fk, block_fk>>>(
            d_ismr, d_ismi, d_tfk, nk, d_steer_re, d_steer_im, nr, n_bins);
        CUDA_CHECK(cudaGetLastError());
        if (profile)
        {
            CUDA_CHECK(cudaDeviceSynchronize());
            profile->fk_ms += elapsed_ms(fk_start);
        }
        copy_tfk(capon_total);
    }

    if (profile)
    {
        profile->chunk_total_ms += elapsed_ms(chunk_start);
        profile->chunks += 1;
    }
}

void run(const Options &opt)
{
    CUDA_CHECK(cudaSetDevice(opt.gpu));
    auto saclist_paths = collect_saclist_paths(opt);
    auto list_plans = build_all_list_plans(saclist_paths, opt);
    validate_single_fft_shape(list_plans);
    int nk = slowness_count(opt.s_range, opt.s_res);
    int active_max_nr = 0;
    for (const auto &plan : list_plans)
        active_max_nr = std::max(active_max_nr, plan.nr);
    GeometryTable geometry;
    bool geometry_loaded = !opt.geometry.empty();
    if (geometry_loaded)
    {
        geometry = read_geometry_table(opt.geometry);
        for (const auto &plan : list_plans)
            (void)receiver_master_indices(geometry, plan.rows);
    }
    int master_nr = geometry_loaded
                        ? static_cast<int>(geometry.nsl_ids.size())
                        : active_max_nr;
    std::size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    WorkspacePlan workspace;
    int cov_window_batch = choose_cov_window_batch(
        list_plans, nk, opt, free_bytes, master_nr, workspace);
    bool multi = !opt.saclist_list.empty() || list_plans.size() > 1;
    SteeringPlan steering_plan = plan_global_steering_cache(
        list_plans, nk, opt, workspace, free_bytes);

    std::cerr << "[fastbf_native] list-manifest native CUDA path\n";
    std::cerr << "[fastbf_native] lists=" << list_plans.size()
              << " mode=" << opt.mode
              << " max_nr=" << workspace.max_nr
              << " max_nwin=" << workspace.max_nwin
              << " max_nsamp=" << workspace.max_nsamp
              << " max_selected_bins=" << workspace.max_selected_bins
              << " max_chunk_bins=" << workspace.max_chunk_bins
              << " master_nr=" << workspace.master_nr
              << " cov_window_batch(auto)=" << cov_window_batch
              << " nk=" << nk
              << " gpu=" << opt.gpu << "\n";

    print_memory_plan(workspace.memory);
    std::cerr << "[fastbf_native] workspace source "
              << workspace.source_saclist << "\n";
    std::cerr << "[fastbf_native] cuda free/total "
              << bytes_text(free_bytes) << " / " << bytes_text(total_bytes) << "\n";
    std::cerr << "[fastbf_native] global steering cache "
              << bytes_text(steering_plan.bytes)
              << " mode=" << steering_plan.mode
              << " reason=" << steering_plan.reason << "\n";
    if (workspace.memory.arena > free_bytes)
        throw std::runtime_error("estimated arena is larger than current free GPU memory");
    print_machine_plan(workspace, steering_plan, cov_window_batch,
                       nk, free_bytes, total_bytes);
    if (opt.plan_only)
        return;

    ProfileStats total_profile;
    DeviceArena arena(workspace.memory.arena);
    PersistentDevice persistent = allocate_persistent(
        arena, workspace.max_nr, workspace.master_nr, nk);
    auto master_tau_start = Clock::now();
    upload_master_geometry_and_tau(geometry, persistent, nk,
                                   opt.s_range, opt.s_res);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (opt.profile)
        total_profile.master_tau_ms += elapsed_ms(master_tau_start);
    std::cerr << "[fastbf_native] master tau ready"
              << " master_nr=" << workspace.master_nr
              << " active_max_nr=" << workspace.max_nr << "\n";
    int max_fft_batch = workspace.max_nr * workspace.max_nwin;
    arena.reset(persistent.scratch_start);
    arena.alloc<cufftDoubleReal>(
        static_cast<std::size_t>(max_fft_batch) * workspace.max_nsamp,
        "fft.plan_xt");
    arena.alloc<cufftDoubleComplex>(
        static_cast<std::size_t>(max_fft_batch) * workspace.max_nfreq,
        "fft.plan_ft");
    void *fft_work_area = arena.alloc_bytes(workspace.max_cufft_work, "fft.plan_work");
    CufftPlan fft_plan = make_fft_plan(
        workspace.max_nsamp, max_fft_batch, fft_work_area);
    CublasHandle blas;
    GlobalSteeringCache global_steering;
    if (steering_plan.mode != "none")
    {
        const ListPlan &ref = list_plans.front();
        select_tau_for_rows(geometry, ref.rows, persistent, ref.nr, nk,
                            opt.profile ? &total_profile : nullptr);
        std::cerr << "[fastbf_native] precomputing global steering "
                  << bytes_text(steering_plan.bytes)
                  << " mode=" << steering_plan.mode << "\n";
        global_steering = build_global_steering_cache(
            arena, persistent, steering_plan,
            ref.nr, nk, ref.nsamp, static_cast<float>(ref.dt));
        CUDA_CHECK(cudaDeviceSynchronize());
        std::cerr << "[fastbf_native] global steering ready\n";
    }

    for (std::size_t list_idx = 0; list_idx < list_plans.size(); ++list_idx)
    {
        ProfileStats list_profile;
        const ListPlan &plan = list_plans[list_idx];
        std::vector<BandChunk> chunks =
            global_steering.mode == "none"
                ? plan_chunks(opt.fmin, opt.fmax, plan.nsamp, plan.dt,
                              plan.nfreq, opt.freq_chunk_bins)
                : steering_plan.chunks;
        int selected_start = chunks.front().start_bin;
        int selected_end = chunks.back().end_bin;
        int selected_bins = selected_end - selected_start + 1;
        std::cerr << "[fastbf_native] list " << (list_idx + 1) << "/"
                  << list_plans.size()
                  << " " << plan.saclist_path
                  << " nr=" << plan.nr
                  << " npts=" << plan.common_npts
                  << " dt=" << plan.dt
                  << " nwin=" << plan.nwin
                  << " nsamp=" << plan.nsamp
                  << " bins=" << selected_start << "-" << selected_end
                  << " chunks=" << chunks.size() << "\n";

        if (global_steering.mode == "none")
        {
            select_tau_for_rows(geometry, plan.rows, persistent, plan.nr, nk,
                                opt.profile ? &list_profile : nullptr);
        }
        std::cerr << "[fastbf_native] taper=" << taper_description(opt)
                  << " amplitude_seconds="
                  << taper_amplitude_seconds(plan.nsamp, plan.dt, opt.taper, opt.taper_alpha)
                  << " power_seconds="
                  << taper_power_seconds(plan.nsamp, plan.dt, opt.taper, opt.taper_alpha)
                  << "\n";

        auto read_start = Clock::now();
        auto matrix = read_data_matrix(plan.headers, plan.common_npts);
        if (opt.profile)
            list_profile.read_ms += elapsed_ms(read_start);
        auto window_start = Clock::now();
        auto window_stack = build_window_stack(matrix, plan.nr,
                                               plan.common_npts, plan.windows, opt);
        if (opt.profile)
            list_profile.window_ms += elapsed_ms(window_start);
        auto ft_selected = run_fft_selected(
            arena, persistent, fft_plan, window_stack,
            plan.nr, plan.nwin, plan.nsamp,
            max_fft_batch, workspace.max_nsamp, workspace.max_nfreq,
            selected_start, selected_end,
            workspace.max_cufft_work,
            opt.profile ? &list_profile : nullptr);

        bool want_capon = opt.method == "capon" || opt.method == "both";
        bool want_bartlett = opt.method == "bartlett" || opt.method == "both";
        int group_size = output_group_windows(opt, plan.nwin);
        if (group_size <= 0)
            throw std::runtime_error("invalid beamforming output group size");
        int group_count = (plan.nwin + group_size - 1) / group_size;
        std::cerr << "[fastbf_native] output groups=" << group_count
                  << " group_windows=" << group_size
                  << " method=" << opt.method << "\n";
        for (int group_idx = 0; group_idx < group_count; ++group_idx)
        {
            int first_window = group_idx * group_size;
            int frame_nwin = std::min(group_size, plan.nwin - first_window);
            std::vector<float> capon_total;
            std::vector<float> bartlett_total;
            if (want_capon)
                capon_total.assign(static_cast<std::size_t>(nk) * nk, 0.0f);
            if (want_bartlett)
                bartlett_total.assign(static_cast<std::size_t>(nk) * nk, 0.0f);
            std::cerr << "[fastbf_native] group " << (group_idx + 1) << "/"
                      << group_count
                      << " first_window=" << first_window
                      << " nwin=" << frame_nwin
                      << " label=" << frame_time_label(plan, first_window)
                      << "\n";
            for (std::size_t idx = 0; idx < chunks.size(); ++idx)
            {
                const auto &chunk = chunks[idx];
                run_beamforming_chunk(arena, persistent, blas, ft_selected,
                                      selected_bins, selected_start,
                                      cov_window_batch,
                                      global_steering.host_re_ptr(),
                                      global_steering.host_im_ptr(),
                                      global_steering.device_re_ptr(),
                                      global_steering.device_im_ptr(),
                                      global_steering.mode == "none"
                                          ? selected_start
                                          : global_steering.selected_start,
                                      chunk, frame_nwin, plan.nr, nk,
                                      first_window, plan.nsamp,
                                      static_cast<float>(plan.dt), opt.dl,
                                      want_capon, want_bartlett,
                                      want_capon ? &capon_total : nullptr,
                                      want_bartlett ? &bartlett_total : nullptr,
                                      opt.profile ? &list_profile : nullptr);
                std::cerr << "[fastbf_native] chunk " << (idx + 1) << "/"
                          << chunks.size() << " bins=" << chunk.start_bin << "-"
                          << chunk.end_bin << "\n";
            }
            auto write_start = Clock::now();
            if (want_capon)
            {
                std::string out_path =
                    output_path_for_frame_method(opt, plan, multi, group_idx,
                                                 first_window, "capon");
                write_npy_f4_2d(out_path, capon_total, nk, nk);
                std::cerr << "[fastbf_native] wrote " << out_path << "\n";
            }
            if (want_bartlett)
            {
                std::string out_path =
                    output_path_for_frame_method(opt, plan, multi, group_idx,
                                                 first_window, "bartlett");
                write_npy_f4_2d(out_path, bartlett_total, nk, nk);
                std::cerr << "[fastbf_native] wrote " << out_path << "\n";
            }
            if (opt.profile)
                list_profile.write_ms += elapsed_ms(write_start);
        }
        if (opt.profile)
        {
            add_profile(total_profile, list_profile);
            print_profile("list=" + saclist_stem(plan.saclist_path), list_profile);
        }
    }
    if (opt.profile)
        print_profile("total", total_profile);
}

}  // namespace

int main(int argc, char **argv)
{
    try
    {
        Options opt = parse_args(argc, argv);
        run(opt);
        return 0;
    }
    catch (const std::exception &exc)
    {
        std::cerr << "fastbf_native: " << exc.what() << "\n";
        return 2;
    }
}
