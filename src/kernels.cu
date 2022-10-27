#include <cuda_runtime.h>
#include <iostream>

#include "kernels.hpp"

namespace kernels {

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

template <typename T>
void check(T result, char const *const func, const char *const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(result),
                cudaGetErrorName(result), func);
        exit(EXIT_FAILURE);
    }
}

template <typename T, int srcdim_, int trgdim_>
struct GenericCudaKernel {
    constexpr static int srcdim = srcdim_;
    constexpr static int trgdim = trgdim_;
    using floattype = T;
};

template <typename T>
struct StokesCuda : GenericCudaKernel<T, 3, 3> {
    constexpr static T scale_factor = 1.0 / 8.0 / M_PI;
    constexpr static char name[] = "Stokes";

    __device__ static void uKernel(const T *ri, const T *rj, const T *f, T *u) {
        T dr[3];

        dr[0] = rj[0] - ri[0];
        dr[1] = rj[1] - ri[1];
        dr[2] = rj[2] - ri[2];

        T r2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
        const T rinv = r2 == 0.0 ? 0.0 : rsqrt(r2);
        T rinv2 = rinv * rinv;
        T inner_prod = (f[0] * dr[0] + f[1] * dr[1] + f[2] * dr[2]) * rinv2;
        u[0] += rinv * (f[0] + dr[0] * inner_prod);
        u[1] += rinv * (f[1] + dr[1] * inner_prod);
        u[2] += rinv * (f[2] + dr[2] * inner_prod);
    }
};

template <typename kernel>
__global__ void tiled_driver(const typename kernel::floattype *r_src, const typename kernel::floattype *r_trg,
                             typename kernel::floattype *__restrict__ u_trg, const typename kernel::floattype *f_src,
                             int n_src, int n_trg, int n_tiles) {

    using T = typename kernel::floattype;
    int i_trg = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ char shared_char[];
    T *shared = (T *)shared_char;
    const int shared_row_size = 3 + kernel::srcdim;
    const int toffset = shared_row_size * threadIdx.x;
    T *r_shared = (T *)(&shared[toffset]);
    T *f_shared = (T *)(&shared[toffset] + 3);
    if (i_trg < n_trg) {
        for (int i = 0; i < kernel::trgdim; ++i)
            u_trg[i_trg * kernel::trgdim + i] = 0.0;
    }
    for (int tile = 0; tile < n_tiles; tile++) {
        const int i_src = (tile * blockDim.x + threadIdx.x);
        for (int i = 0; i < 3; ++i)
            r_shared[i] = r_src[i_src * 3 + i];

        for (int i = 0; i < kernel::srcdim; ++i)
            f_shared[i] = f_src[i_src * kernel::srcdim + i];

        __syncthreads();

        // Loop over particles in our tile. But if tile contains i_src >= n_src, don't include those
        const int n_local_max = ((tile + 1) * (blockDim.x) > n_src) ? n_src - tile * blockDim.x : blockDim.x;
        if (i_trg < n_trg) {
            for (int i_local = 0; i_local < n_local_max; i_local++) {
                kernel::uKernel(&shared[shared_row_size * i_local], &r_trg[i_trg * 3],
                                &shared[shared_row_size * i_local + 3], &u_trg[i_trg * kernel::trgdim]);
            }
        }
        __syncthreads();
    }

    if (i_trg >= n_trg)
        return;

    for (int i = 0; i < kernel::trgdim; ++i)
        u_trg[i_trg * kernel::trgdim + i] *= kernel::scale_factor;
}

template <typename kernel>
__global__ void untiled_driver(const typename kernel::floattype *r_src, const typename kernel::floattype *r_trg,
                               typename kernel::floattype *__restrict__ u_trg, const typename kernel::floattype *f_src,
                               int n_src, int n_trg) {
    const int threadId = threadIdx.x;
    const int blkIdx = blockIdx.x;
    const int blkDim = blockDim.x;
    const int i_trg = threadId + blkIdx * blkDim;

    if (i_trg >= n_trg)
        return;

    for (int i = 0; i < kernel::trgdim; ++i)
        u_trg[i_trg * kernel::trgdim + i] = 0.0;

    for (int i_src = 0; i_src < n_src; ++i_src)
        kernel::uKernel(&r_src[i_src * 3], &r_trg[3 * i_trg], &f_src[i_src * kernel::srcdim],
                        &u_trg[i_trg * kernel::trgdim]);

    for (int i = 0; i < kernel::trgdim; ++i)
        u_trg[i_trg * kernel::trgdim + i] *= kernel::scale_factor;
}

template <typename kernel, Driver driver>
void kernel_direct_gpu(const typename kernel::floattype *r_src, const typename kernel::floattype *f_src, int n_src,
                       const typename kernel::floattype *r_trg, typename kernel::floattype *u_trg, int n_trg) {
    const int block_size = 32;
    const int n_blocks = (n_trg + block_size - 1) / block_size;
    using T = typename kernel::floattype;

    T *r_src_device, *r_trg_device, *f_src_device, *u_trg_device;

    checkCudaErrors(cudaMalloc((void **)&r_src_device, 3 * n_src * sizeof(T)));
    checkCudaErrors(cudaMalloc((void **)&f_src_device, kernel::srcdim * n_src * sizeof(T)));
    checkCudaErrors(cudaMalloc((void **)&r_trg_device, 3 * n_trg * sizeof(T)));
    checkCudaErrors(cudaMalloc((void **)&u_trg_device, kernel::trgdim * n_trg * sizeof(T)));

    checkCudaErrors(cudaMemcpy(r_src_device, r_src, 3 * n_src * sizeof(T), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(r_trg_device, r_trg, 3 * n_trg * sizeof(T), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(f_src_device, f_src, kernel::srcdim * n_src * sizeof(T), cudaMemcpyHostToDevice));

    if constexpr (driver == TILED) {
        int n_tiles = (n_src + block_size - 1) / block_size;
        int shared_mem_size = block_size * (3 + kernel::srcdim) * sizeof(T);
        tiled_driver<kernel><<<n_blocks, block_size, shared_mem_size>>>(r_src_device, r_trg_device, u_trg_device,
                                                                        f_src_device, n_src, n_trg, n_tiles);
    }
    else
        untiled_driver<kernel>
            <<<n_blocks, block_size>>>(r_src_device, r_trg_device, u_trg_device, f_src_device, n_src, n_trg);

    checkCudaErrors(cudaMemcpy(u_trg, u_trg_device, sizeof(T) * n_trg * kernel::trgdim, cudaMemcpyDeviceToHost));

    cudaFree(r_src_device);
    cudaFree(f_src_device);
    cudaFree(u_trg_device);
    cudaFree(r_trg_device);
}

void stokeslet_direct_gpu_tiled(const double *r_src, const double *f_src, int n_src, const double *r_trg, double *u_trg,
                                int n_trg) {
    kernel_direct_gpu<StokesCuda<double>, TILED>(r_src, f_src, n_src, r_trg, u_trg, n_trg);
}

void stokeslet_direct_gpu_tiled(const float *r_src, const float *f_src, int n_src, const float *r_trg, float *u_trg,
                                int n_trg) {
    kernel_direct_gpu<StokesCuda<float>, TILED>(r_src, f_src, n_src, r_trg, u_trg, n_trg);
}

void stokeslet_direct_gpu_untiled(const double *r_src, const double *f_src, int n_src, const double *r_trg,
                                  double *u_trg, int n_trg) {
    kernel_direct_gpu<StokesCuda<double>, UNTILED>(r_src, f_src, n_src, r_trg, u_trg, n_trg);
}

void stokeslet_direct_gpu_untiled(const float *r_src, const float *f_src, int n_src, const float *r_trg, float *u_trg,
                                  int n_trg) {
    kernel_direct_gpu<StokesCuda<float>, UNTILED>(r_src, f_src, n_src, r_trg, u_trg, n_trg);
}

} // namespace kernels
