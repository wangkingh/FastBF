#include "cufft.h"
#include "cuComplex.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernels.cuh"
#include <cmath>
#include <stdio.h>
#include <cublas_v2.h>
#define NUM_DEVICE 4
#define DEVICE_TO_USE 3

/* --------------------------------------------------------------------------
   compute_fk  ── Single-GPU Capon FK kernel
   ---------------------------------------------------------------------------
   Purpose :   accumulate 1/(a† S⁻¹ a) over the selected freq-bins and write
               the FK power into tfk (one 2-D slowness grid per call).
FFf
   Grid   :    blockDim{8,8}, gridDim{ceil(nk/8), ceil(nk/8)}
   Each thread (i,j) handles one (kx,ky) point.

   in/out :
       ismr, ismi  [in]   real & imag parts of S⁻¹,   size = (2*fave+1)*nr*nr
       tfk         [in/out]  pre-zeroed nk*nk power plane, += per-freq term
       rx, ry      [in]   array geometry, length = nr
       other scalars = slowness, freq index info, etc.
   Assumes:   tfk already zero on entry; atomic not required because each
              (i,j) is unique per thread.
-------------------------------------------------------------------------- */
__global__ void compute_fk(float *ismr, int ismr_size, float *ismi, int ismi_size, float *tfk, int tfk_size, int nk, float *rx, int rx_size, float *ry, int ry_size, float kinc, float kmin, int nr, int nsamp, float delta, float smin, float sinc, int fave, int find)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int g, m, n;
    double freq, kx, ky, arg;
    if (i < nk && j < nk)
    {
        for (g = find - fave; g <= find + fave; g++)
        {
            freq = g / (nsamp * delta);
            kmin = 2 * M_PI * freq * smin;
            kinc = 2 * M_PI * freq * sinc;
            kx = -(kmin + i * kinc);
            ky = -(kmin + j * kinc);
            float tmp_fk = 0.0;
            for (m = 0; m < nr; m++)
            {
                tmp_fk += ismr[(g - find + fave) * nr * nr + m * nr + m];
                for (n = m + 1; n < nr; n++)
                {
                    arg = kx * (rx[m] - rx[n]) + ky * (ry[m] - ry[n]);
                    tmp_fk += 2 * (ismr[(g - find + fave) * nr * nr + m * nr + n] * cosf(arg) - ismi[(g - find + fave) * nr * nr + m * nr + n] * sinf(arg));
                }
            }
            tmp_fk = 1.0 / tmp_fk;
            tfk[i * nk + j] += tmp_fk;
        }
    }
}

/* --------------------------------------------------------------------------
   compute_fk_so  ── Thin C-wrapper that alloc / copy / launch compute_fk
-------------------------------------------------------------------------- */
void compute_fk_so(float ismr[], int ismr_size, float ismi[], int ismi_size, float tfk[], int tfk_size, int nk, float rx[], int rx_size, float ry[], int ry_size, float kinc, float kmin, int nr, int nsamp, float delta, float smin, float sinc, int fave, int find)
{
    cudaSetDevice(DEVICE_TO_USE);
    float *d_ismr, *d_ismi, *d_tfk, *d_rx, *d_ry;
    cudaMalloc((void **)&d_ismr, sizeof(float) * ismr_size);
    cudaMalloc((void **)&d_ismi, sizeof(float) * ismi_size);
    cudaMalloc((void **)&d_tfk, sizeof(float) * tfk_size);
    cudaMalloc((void **)&d_rx, sizeof(float) * rx_size);
    cudaMalloc((void **)&d_ry, sizeof(float) * ry_size);
    cudaMemcpy(d_ismr, ismr, sizeof(float) * ismr_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ismi, ismi, sizeof(float) * ismi_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tfk, tfk, sizeof(float) * tfk_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rx, rx, sizeof(float) * rx_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ry, ry, sizeof(float) * ry_size, cudaMemcpyHostToDevice);

    dim3 blockSize(8, 8, 1);
    dim3 gridSize((nk + blockSize.x - 1) / blockSize.x, (nk + blockSize.y - 1) / blockSize.y, 1);
    compute_fk<<<gridSize, blockSize>>>(d_ismr, ismr_size, d_ismi, ismi_size, d_tfk, tfk_size, nk, d_rx, rx_size, d_ry, ry_size, kinc, kmin, nr, nsamp, delta, smin, sinc, fave, find);

    cudaMemcpy(tfk, d_tfk, sizeof(float) * tfk_size, cudaMemcpyDeviceToHost);

    cudaFree(d_ismr);
    cudaFree(d_ismi);
    cudaFree(d_tfk);
    cudaFree(d_rx);
    cudaFree(d_ry);

    return;
}

/* --------------------------------------------------------------------------
   fft_cuda  ── batched cufftExecD2Z wrapper for nr traces
   -------------------------------------------------------------------------
   xt_real : [in]  nr×nsamp  host-side double
   ft      : [out] nr×(nsamp/2+1) complex spectrum (host)
--------------------------------------------------------------------------- */
extern "C" void fft_cuda(double *xt_real, int nsamp, int nr, double *ft)
{
    cudaError_t e;
    cufftResult r;
    // fprintf(stderr, ">> enter fft_cuda  nsamp=%d  nr=%d\n", nsamp, nr);
    fflush(stderr); // ★ 立刻刷屏

    cufftHandle plan;
    r = cufftCreate(&plan);
    if (r != CUFFT_SUCCESS)
    {
        fprintf(stderr, "cufftCreate: %d\n", r);
        return;
    }

    r = cufftPlan1d(&plan, nsamp, CUFFT_D2Z, nr);
    if (r != CUFFT_SUCCESS)
    {
        fprintf(stderr, "cufftPlan1d: %d\n", r);
        cufftDestroy(plan);
        return;
    }

    cufftDoubleReal *d_xt = nullptr;
    cufftDoubleComplex *d_ft = nullptr;
    size_t bytes_xt = sizeof(cufftDoubleReal) * nsamp * nr;
    size_t bytes_ft = sizeof(cufftDoubleComplex) * (nsamp / 2 + 1) * nr;

    e = cudaMalloc(&d_xt, bytes_xt);
    if (e != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc d_xt: %s\n", cudaGetErrorString(e));
        goto cleanup;
    }

    e = cudaMalloc(&d_ft, bytes_ft);
    if (e != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc d_ft: %s\n", cudaGetErrorString(e));
        goto cleanup;
    }

    e = cudaMemcpy(d_xt, xt_real, bytes_xt, cudaMemcpyHostToDevice);
    if (e != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy H2D: %s\n", cudaGetErrorString(e));
        goto cleanup;
    }

    r = cufftExecD2Z(plan, d_xt, d_ft);
    if (r != CUFFT_SUCCESS)
    {
        fprintf(stderr, "cufftExecD2Z: %d\n", r);
        goto cleanup;
    }

    e = cudaMemcpy(ft, d_ft, bytes_ft, cudaMemcpyDeviceToHost);
    if (e != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy D2H: %s\n", cudaGetErrorString(e));
    }

cleanup:
    if (d_xt)
        cudaFree(d_xt);
    if (d_ft)
        cudaFree(d_ft);
    cufftDestroy(plan);
}

/* --------------------------------------------------------------------------
   compute_fk_mulgpu  ── Same math as compute_fk but runs only a sub-range
                         of g-bins (g_begin … g_end); used for multi-GPU.
-------------------------------------------------------------------------- */
__global__ void compute_fk_mulgpu(float *ismr, int ismr_size, float *ismi, int ismi_size, float *tfk, int tfk_size, int nk, float *rx, int rx_size, float *ry, int ry_size, float kinc, float kmin, int nr, int nsamp, float delta, float smin, float sinc, int fave, int find, int g_begin, int g_end)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int g, m, n;
    double freq, kx, ky, arg;
    if (i < nk && j < nk)
    {
        for (g = g_begin; g <= g_end; g++)
        {
            freq = (g + find - fave) / (nsamp * delta);
            kmin = 2 * M_PI * freq * smin;
            kinc = 2 * M_PI * freq * sinc;
            kx = -(kmin + i * kinc);
            ky = -(kmin + j * kinc);
            float tmp_fk = 0.0;
            for (m = 0; m < nr; m++)
            {
                tmp_fk += ismr[(g - g_begin) * nr * nr + m * nr + m];
                for (n = m + 1; n < nr; n++)
                {
                    arg = kx * (rx[m] - rx[n]) + ky * (ry[m] - ry[n]);
                    tmp_fk += 2.0 * (ismr[(g - g_begin) * nr * nr + m * nr + n] * cosf(arg) - ismi[(g - g_begin) * nr * nr + m * nr + n] * sinf(arg));
                }
            }
            tmp_fk = 1.0 / tmp_fk;
            tfk[i * nk + j] += tmp_fk;
        }
    }
}

/* --------------------------------------------------------------------------
   addPartTFK  ── Sum NUM_DEVICE partial tfk arrays element-wise on one GPU
-------------------------------------------------------------------------- */
__global__ void addPartTFK(float *tfk, float *part_tfk, int tfk_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < tfk_size)
    {
        for (int i = 0; i < NUM_DEVICE; ++i)
        {
            tfk[tid] += part_tfk[i * tfk_size + tid];
        }
    }
}

/* --------------------------------------------------------------------------
   compute_fk_so_mulgpu  ── Host coordinator for multi-GPU FK computation.
   Cuts freq-bins across GPUs, launches compute_fk_mulgpu on each,
   gathers partial tfk with addPartTFK.
-------------------------------------------------------------------------- */
void compute_fk_so_mulgpu(float ismr[], int ismr_size, float ismi[], int ismi_size, float tfk[], int tfk_size, int nk, float rx[], int rx_size, float ry[], int ry_size, float kinc, float kmin, int nr, int nsamp, float delta, float smin, float sinc, int fave, int find)
{
    int g_begin[NUM_DEVICE];
    int g_end[NUM_DEVICE];
    int g_size[NUM_DEVICE];
    for (int i = 0; i < NUM_DEVICE; ++i)
    {
        g_begin[i] = i * (2 * fave + 1) / NUM_DEVICE;
        g_end[i] = (i + 1) * (2 * fave + 1) / NUM_DEVICE - 1;
        g_size[i] = g_end[i] - g_begin[i] + 1;
    }

    float *d_ismr[NUM_DEVICE], *d_ismi[NUM_DEVICE], *d_tfk[NUM_DEVICE], *d_rx[NUM_DEVICE], *d_ry[NUM_DEVICE], *part_tfk[NUM_DEVICE];
    for (int i = 0; i < NUM_DEVICE; i++)
    {
        cudaSetDevice(i);
        cudaMalloc((void **)&d_ismr[i], sizeof(float) * nr * nr * g_size[i]);
        cudaMalloc((void **)&d_ismi[i], sizeof(float) * nr * nr * g_size[i]);
        cudaMalloc((void **)&d_tfk[i], sizeof(float) * tfk_size);
        cudaMalloc((void **)&d_rx[i], sizeof(float) * rx_size);
        cudaMalloc((void **)&d_ry[i], sizeof(float) * ry_size);
        part_tfk[i] = (float *)malloc(sizeof(float) * tfk_size);
    }
    for (int i = 0; i < NUM_DEVICE; i++)
    {
        cudaSetDevice(i);
        cudaMemcpy(d_ismr[i], ismr + nr * nr * g_begin[i], sizeof(float) * nr * nr * g_size[i], cudaMemcpyHostToDevice);
        cudaMemcpy(d_ismi[i], ismi + nr * nr * g_begin[i], sizeof(float) * nr * nr * g_size[i], cudaMemcpyHostToDevice);
        cudaMemcpy(d_tfk[i], tfk, sizeof(float) * tfk_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_rx[i], rx, sizeof(float) * rx_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_ry[i], ry, sizeof(float) * ry_size, cudaMemcpyHostToDevice);
    }
    for (int i = 0; i < NUM_DEVICE; ++i)
    {
        cudaSetDevice(i);
        dim3 blockSize(8, 8, 1);
        dim3 gridSize((nk + blockSize.x - 1) / blockSize.x, (nk + blockSize.y - 1) / blockSize.y, 1);
        compute_fk_mulgpu<<<gridSize, blockSize>>>(d_ismr[i], ismr_size, d_ismi[i], ismi_size, d_tfk[i], tfk_size, nk, d_rx[i], rx_size, d_ry[i], ry_size, kinc, kmin, nr, nsamp, delta, smin, sinc, fave, find, g_begin[i], g_end[i]);
    }
    for (int i = 0; i < NUM_DEVICE; ++i)
    {
        cudaSetDevice(i);
        cudaMemcpy(part_tfk[i], d_tfk[i], sizeof(float) * tfk_size, cudaMemcpyDeviceToHost);
    }
    for (int i = 0; i < NUM_DEVICE; ++i)
    {
        cudaSetDevice(i);
        cudaFree(d_ismr[i]);
        cudaFree(d_ismi[i]);
        cudaFree(d_tfk[i]);
        cudaFree(d_rx[i]);
        cudaFree(d_ry[i]);
    }

    // 合并part_tfk
    float *d_part_tfk;
    cudaMalloc((void **)&d_part_tfk, sizeof(float *) * NUM_DEVICE * tfk_size);
    for (int i = 0; i < NUM_DEVICE; i++)
    {
        cudaMemcpy(d_part_tfk + i * tfk_size, part_tfk[i], sizeof(float) * tfk_size, cudaMemcpyHostToDevice);
    }
    float *d_final_tfk;
    cudaMalloc((void **)&d_final_tfk, sizeof(float) * tfk_size);
    addPartTFK<<<(tfk_size + 255) / 256, 256>>>(d_final_tfk, d_part_tfk, tfk_size);
    cudaMemcpy(tfk, d_final_tfk, sizeof(float) * tfk_size, cudaMemcpyDeviceToHost);
    cudaFree(d_final_tfk);
    cudaFree(d_part_tfk);

    return;
}

/* --------------------------------------------------------------------------
   compute_sm  ── Kernel to build (and average) covariance matrices S(f)
   -------------------------------------------------------------------------
   ft   : [in]  (nwin × 2*nr × (2*fave+1))  real array, interleaved Re/Im
   nwin : #windows packed in ft
   sm   : [in/out] (2*fave+1) × 2*nr × nr   ← on exit already ÷ nwin
--------------------------------------------------------------------------- */
__global__ void compute_sm(float *ft, int nwin, int nr, int fave, float *sm)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int n;
    int tmp = 2 * fave + 1;
    if (i < nr && j < nr)
    {
        for (n = 0; n < nwin; n++)
        {
            for (int l = 0; l < 2 * fave + 1; l++)
            {
                // sm[l][i][j] += ft[n][i][l]*ft[n][j][l]+ft[n][i+nr][l]*ft[n][j+nr][l];
                sm[l * 2 * nr * nr + i * nr + j] += ft[n * 2 * nr * tmp + i * tmp + l] * ft[n * 2 * nr * tmp + j * tmp + l] + ft[n * 2 * nr * tmp + (i + nr) * tmp + l] * ft[n * 2 * nr * tmp + (j + nr) * tmp + l];
                // sm[l][i+nr][j] += ft[n][j][l]*ft[n][i+nr][l]-ft[n][i][l]*ft[n][j+nr][l];
                sm[l * 2 * nr * nr + (i + nr) * nr + j] += ft[n * 2 * nr * tmp + j * tmp + l] * ft[n * 2 * nr * tmp + (i + nr) * tmp + l] - ft[n * 2 * nr * tmp + i * tmp + l] * ft[n * 2 * nr * tmp + (j + nr) * tmp + l];
            }
        }
        for (int l = 0; l < 2 * fave + 1; l++)
        {
            sm[l * 2 * nr * nr + i * nr + j] /= nwin;
            sm[l * 2 * nr * nr + (i + nr) * nr + j] /= nwin;
        }
    }
    return;
}

/* --------------------------------------------------------------------------
   compute_w  ── Per-frequency diagonal weighting  |S_ii|^{-1/2}
-------------------------------------------------------------------------- */
__global__ void compute_w(float *sm, int fave, int nr, float *w)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nr && j < 2 * fave + 1)
    {
        // 1. read the real and imaginary part
        float re = sm[j * 2 * nr * nr + i * nr + i];
        float im = sm[j * 2 * nr * nr + (i + nr) * nr + i];

        // 2. calculate the square of amplitude
        float mag_sq = re * re + im * im;

        // 3. use famxf to avoid do nagetive power on zero value
        float eps = 1.0e-12f;
        float mag_sq_safe = fmaxf(mag_sq, eps);

        // 4.  wtemp = mag_sq_safe^(-0.25)
        //    也就是幅度的 -1/2 次方（1/sqrt(amplitude)）
        float wtemp = powf(mag_sq_safe, -0.25f);

        // 5. 生成掩码：若原本幅度为 0，则 mask=0，否则 mask=1
        //    这样真正幅度为 0 的地方就会被乘零
        float mask = (mag_sq > 0.0f) ? 1.0f : 0.0f;

        // 6. 用掩码将真正 0 幅度的点置为 0，
        //    非 0 幅度点保留 wtemp
        w[j * nr + i] = wtemp * mask;
    }
}

/* --------------------------------------------------------------------------
   compute_sm_weighted  ── Apply weight matrix W(f) to every entry of S(f)
-------------------------------------------------------------------------- */
__global__ void compute_sm_weighted(float *sm, float *w, int fave, int nr)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int m;
    if (i < nr && j < nr)
    {
        for (m = 0; m < 2 * fave + 1; m++)
        {
            sm[m * 2 * nr * nr + i * nr + j] *= w[m * nr + i] * w[m * nr + j];
            sm[m * 2 * nr * nr + (i + nr) * nr + j] *= w[m * nr + i] * w[m * nr + j];
        }
    }
    return;
}

void compute_sm_so(float ft[], int nwin, int nr, int fave, float sm[])
{
    cudaSetDevice(DEVICE_TO_USE);
    float *d_ft, *d_sm, *d_w;
    cudaMalloc((void **)&d_ft, sizeof(float) * nwin * 2 * nr * (2 * fave + 1));
    cudaMalloc((void **)&d_sm, sizeof(float) * 2 * nr * nr * (2 * fave + 1));
    cudaMalloc((void **)&d_w, sizeof(float) * (2 * fave + 1) * nr);
    cudaMemcpy(d_ft, ft, sizeof(float) * nwin * 2 * nr * (2 * fave + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sm, sm, sizeof(float) * 2 * nr * nr * (2 * fave + 1), cudaMemcpyHostToDevice);
    cudaMemset(d_w, 0, sizeof(float) * (2 * fave + 1) * nr);
    dim3 blockSize(8, 8, 1);
    dim3 gridSize((nr + blockSize.x - 1) / blockSize.x, (nr + blockSize.y - 1) / blockSize.y, 1);
    compute_sm<<<gridSize, blockSize>>>(d_ft, nwin, nr, fave, d_sm);
    dim3 blockSize_1(8, 8, 1);
    dim3 gridSize_1((nr + blockSize.x - 1) / blockSize.x, (2 * fave + blockSize.y) / blockSize.y, 1);
    compute_w<<<gridSize_1, blockSize_1>>>(d_sm, fave, nr, d_w);
    compute_sm_weighted<<<gridSize, blockSize>>>(d_sm, d_w, fave, nr);
    cudaMemcpy(sm, d_sm, sizeof(float) * 2 * nr * nr * (2 * fave + 1), cudaMemcpyDeviceToHost);
    cudaFree(d_w);
    cudaFree(d_sm);
    cudaFree(d_ft);
    return;
}

__global__ void complex_add_my(float *smr, float *smi, cuComplex *tx, int nr, int fave)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int l = blockIdx.z * blockDim.z + threadIdx.z;
    if (i < nr && j < nr && l < 2 * fave + 1)
    {
        tx[l * nr * nr + j * nr + i] = make_cuComplex(smr[l * nr * nr + j * nr + i], smi[l * nr * nr + j * nr + i]);
    }
}

void complex_add_so(float smr[], float smi[], cuComplex tx[], int nr, int fave)
{
    cudaSetDevice(DEVICE_TO_USE);
    float *d_smr, *d_smi;
    cuComplex *d_tx;
    cudaMalloc((void **)&d_smr, sizeof(float) * nr * nr * (2 * fave + 1));
    cudaMalloc((void **)&d_smi, sizeof(float) * nr * nr * (2 * fave + 1));
    cudaMalloc((void **)&d_tx, sizeof(cuComplex) * nr * nr * (2 * fave + 1));
    cudaMemcpy(d_smr, smr, sizeof(float) * nr * nr * (2 * fave + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_smi, smi, sizeof(float) * nr * nr * (2 * fave + 1), cudaMemcpyHostToDevice);
    cudaMemset(d_tx, 0, sizeof(cuComplex) * nr * nr * (2 * fave + 1));
    dim3 blockSize(8, 8, 8);
    dim3 gridSize((nr + blockSize.x - 1) / blockSize.x, (nr + blockSize.y - 1) / blockSize.y, (2 * fave + blockSize.z) / blockSize.z);
    complex_add_my<<<gridSize, blockSize>>>(d_smr, d_smi, d_tx, nr, fave);
    cudaMemcpy(tx, d_tx, sizeof(cuComplex) * nr * nr * (2 * fave + 1), cudaMemcpyDeviceToHost);
    cudaFree(d_tx);
    cudaFree(d_smi);
    cudaFree(d_smr);
    return;
}

/* --------------------------------------------------------------------------
   martix_invert_so  ── cuBLAS batched LU + inv  (complex single precision)
   -------------------------------------------------------------------------
   src  : [in]   batchSize × n × n  matrices (host)
   dst  : [out]  inverses (host)
   P/INFO buffers live on device.
-------------------------------------------------------------------------- */
void martix_invert_so(cuComplex *src, cuComplex *dst, int n, int batchSize)
{
    cudaSetDevice(DEVICE_TO_USE);
    cublasHandle_t handle;
    cublasCreate_v2(&handle);

    int *P, *INFO;
    cudaMalloc((void **)&P, sizeof(int) * batchSize * n);
    cudaMalloc((void **)&INFO, sizeof(int) * batchSize);
    int lda = n;
    cuComplex **A = (cuComplex **)malloc(batchSize * sizeof(cuComplex *));
    cuComplex **d_A, *A_dflat;
    cudaMalloc((void **)&d_A, batchSize * sizeof(cuComplex *));
    cudaMalloc((void **)&A_dflat, batchSize * n * n * sizeof(cuComplex));
    A[0] = A_dflat;
    for (int i = 1; i < batchSize; i++)
    {
        A[i] = A[i - 1] + n * n;
    }
    // step1: LU decomposition
    cudaMemcpy(d_A, A, batchSize * sizeof(cuComplex *), cudaMemcpyHostToDevice);
    cudaMemcpy(A_dflat, src, batchSize * n * n * sizeof(cuComplex), cudaMemcpyHostToDevice);
    cublasCgetrfBatched(handle, n, d_A, lda, P, INFO, batchSize);

    // step2: invert
    cuComplex **C = (cuComplex **)malloc(batchSize * sizeof(cuComplex *));
    cuComplex **d_C, *C_dflat;
    cudaMalloc((void **)&d_C, batchSize * sizeof(cuComplex *));
    cudaMalloc((void **)&C_dflat, batchSize * n * n * sizeof(cuComplex));
    C[0] = C_dflat;
    for (int i = 1; i < batchSize; i++)
    {
        C[i] = C[i - 1] + n * n;
    }
    cudaMemcpy(d_C, C, batchSize * sizeof(cuComplex *), cudaMemcpyHostToDevice);
    cublasCgetriBatched(handle, n, (const cuComplex **)d_A, lda, P, d_C, lda, INFO, batchSize);
    cudaMemcpy(dst, C_dflat, batchSize * n * n * sizeof(cuComplex), cudaMemcpyDeviceToHost);
    cudaFree(P);
    cudaFree(INFO);
    cudaFree(d_A);
    cudaFree(A_dflat);
    cudaFree(d_C);
    cudaFree(C_dflat);
    free(A);
    free(C);
    cublasDestroy_v2(handle);
}
