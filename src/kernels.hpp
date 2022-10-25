#ifndef KERNELS_HPP
#define KERNELS_HPP

namespace kernels {
using DRIVER = enum { TILED, UNTILED };

void stokeslet_direct_gpu_tiled(const double *r_src, const double *f_src, int n_src, const double *r_trg, double *u_trg,
                                int n_trg);

void stokeslet_direct_gpu_tiled(const float *r_src, const float *f_src, int n_src, const float *r_trg, float *u_trg,
                                int n_trg);

void stokeslet_direct_gpu_untiled(const double *r_src, const double *f_src, int n_src, const double *r_trg,
                                  double *u_trg, int n_trg);

void stokeslet_direct_gpu_untiled(const float *r_src, const float *f_src, int n_src, const float *r_trg, float *u_trg,
                                  int n_trg);
} // namespace kernels
#endif
