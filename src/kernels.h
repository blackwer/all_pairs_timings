#ifndef KERNELS_H
#define KERNELS_H

namespace kernels {
void stokeslet_direct_gpu(const double *r_src, const double *f_src, int n_src,
                          const double *r_trg, double *u_trg, int n_trg);

void stokeslet_direct_gpu(const float *r_src, const float *f_src, int n_src,
                          const float *r_trg, float *u_trg, int n_trg);
}
#endif
