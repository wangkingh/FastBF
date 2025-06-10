#ifndef MYCFK_H
#define MYCFK_H

#include "cuComplex.h"

extern "C" void compute_fk_so(float ismr[], int ismr_size, float ismi[], int ismi_size, float tfk[], int tfk_size, int nk, float rx[], int rx_size, float ry[], int ry_size, float kinc, float kmin, int nr, int nsamp, float delta, float smin, float sinc, int fave, int find);

extern "C" void fft_cuda(double xt_real[],int nsamp, int nr,double ft[]);

extern "C" void compute_fk_so_mulgpu(float ismr[], int ismr_size, float ismi[], int ismi_size, float tfk[], int tfk_size, int nk, float rx[], int rx_size, float ry[], int ry_size, float kinc, float kmin, int nr, int nsamp, float delta, float smin, float sinc, int fave, int find);

extern "C" void compute_sm_so(float ft[], int nwin, int nr, int fave, float sm[]);

extern "C" void complex_add_so(float smr[], float smi[], cuComplex tx[], int nr, int fave);

extern "C" void martix_invert_so(cuComplex* src, cuComplex* dst, int n, int batchSize);
#endif


