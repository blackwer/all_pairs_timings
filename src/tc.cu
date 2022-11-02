#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <mma.h>

#include "timer.hpp"
#include <vector>

constexpr int warp_size = 32; // number of threads per warp
constexpr int n_warps_per_block = 1;
constexpr int n_threads_per_block = warp_size * n_warps_per_block;

template <typename T>
void check(T result, char const *const func, const char *const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(result),
                cudaGetErrorName(result), func);
        exit(EXIT_FAILURE);
    }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

__global__ void warmup() {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;
    ib += ia + tid;
}

__global__ void driver_bulk(const double *r_src, int n_src, const double *r_trg, int n_trg, double *__restrict__ u) {
    constexpr int K = 4; // TC K dimension: holds x,y,z,0 of coords
    constexpr int M = 8; // TC M dimension: holds M trg coords
    constexpr int N = 8; // TC N dimension: holds N src coords
    constexpr int n_trg_tiles_per_warp = warp_size / N;
    constexpr int n_trg_tiles_per_block = n_warps_per_block * n_trg_tiles_per_warp;

    constexpr int block_size = warp_size * n_warps_per_block;
    constexpr int tile_shmem_size = M * N;
    constexpr int SHMEM_SIZE = n_trg_tiles_per_block * tile_shmem_size + 2 * block_size;

    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_id = thread_id / warp_size;
    const int n_src_tiles = n_src / N;

    __shared__ double buffer[SHMEM_SIZE];
    double *rmagtrg = buffer + SHMEM_SIZE - 2 * block_size;
    double *rmagsrc = rmagtrg + block_size;

    using namespace nvcuda;
    wmma::fragment<wmma::matrix_a, M, N, K, double, wmma::row_major> r_trg_tile[n_trg_tiles_per_block];

    u[thread_id] = 0.0;
    rmagtrg[threadIdx.x] = 0.0;

#pragma unroll
    for (int i = 0; i < 3; ++i)
        rmagtrg[threadIdx.x] += r_trg[thread_id * 4 + i] * r_trg[thread_id * 4 + i];

#pragma unroll
    for (int trg_tile = 0; trg_tile < n_trg_tiles_per_block; ++trg_tile)
        wmma::load_matrix_sync(r_trg_tile[trg_tile], r_trg + (n_trg_tiles_per_warp * warp_id + trg_tile) * M * 4, K);

    for (int src_tile = 0; src_tile < n_src_tiles; ++src_tile) {
        int subtile = src_tile % 4;
        if (subtile == 0) {
            rmagsrc[threadIdx.x] = 0.0;
#pragma unroll
            for (int i = 0; i < 3; ++i)
                rmagsrc[threadIdx.x] +=
                    r_src[src_tile * 4 * N + threadIdx.x * 4 + i] * r_src[src_tile * 4 * N + threadIdx.x * 4 + i];
        }

        wmma::fragment<wmma::matrix_b, M, N, K, double, wmma::col_major> r_src_tile;
        wmma::fragment<wmma::accumulator, M, N, K, double> r_trg_src_outer[4];

        // Load source fragment for current source tile
        wmma::load_matrix_sync(r_src_tile, r_src + (src_tile * N) * 4, K);

#pragma unroll
        for (int trg_tile = 0; trg_tile < n_trg_tiles_per_block; ++trg_tile) {
            // Initialize the output to zero
            wmma::fill_fragment(r_trg_src_outer[trg_tile], 0.0f);

            // Perform the matrix multiplication
            wmma::mma_sync(r_trg_src_outer[trg_tile], r_trg_tile[trg_tile], r_src_tile, r_trg_src_outer[trg_tile]);

            // Store the output
            wmma::store_matrix_sync(buffer + tile_shmem_size * trg_tile, r_trg_src_outer[trg_tile], N,
                                    wmma::mem_row_major);
        }

        __syncthreads();

#pragma unroll
        for (int i_src = 0; i_src < 8; ++i_src) {
            double dr2 = rmagsrc[i_src + subtile * 8] + rmagtrg[threadIdx.x] - 2.0 * buffer[threadIdx.x * N + i_src];
            u[thread_id] += dr2 == 0.0 ? 0.0 : rsqrt(dr2);
        }
    }
}

template <typename T>
T driver_host(const T &r_src, const T &r_trg) {
    int n_trg = r_trg.size() / 4;
    int n_src = r_src.size() / 4;
    T sol(n_trg);

    for (int i = 0; i < n_src; ++i) {
        for (int j = 0; j < n_trg; ++j) {
            float x = r_trg[j * 4 + 0] - r_src[i * 4 + 0];
            float y = r_trg[j * 4 + 1] - r_src[i * 4 + 1];
            float z = r_trg[j * 4 + 2] - r_src[i * 4 + 2];
            double dr2 = x * x + y * y + z * z;
            sol[j] += dr2 == 0.0 ? 0.0 : rsqrt(dr2);
        }
    }

    return sol;
}

int main(int argc, char *argv[]) {
    using prec_t = double;
    constexpr int n_trg = 1024 * 128;
    constexpr int n_src = 1024 * 128;
    std::vector<prec_t> r_src(4 * n_src);
    std::vector<prec_t> r_trg(4 * n_trg);
    std::vector<prec_t> sol(n_trg);

    for (int i = 0; i < n_src; ++i)
        for (int j = 0; j < 1; ++j)
            r_src[i * 4 + j] = drand48();

    for (int i = 0; i < n_trg; ++i)
        for (int j = 0; j < 1; ++j)
            r_trg[i * 4 + j] = drand48();

    prec_t *r_src_d, *r_trg_d, *sol_d;
    warmup<<<2048, 128>>>();
    cudaDeviceSynchronize();

    checkCudaErrors(cudaMalloc((void **)&r_src_d, r_src.size() * sizeof(prec_t)));
    checkCudaErrors(cudaMalloc((void **)&r_trg_d, r_trg.size() * sizeof(prec_t)));
    checkCudaErrors(cudaMalloc((void **)&sol_d, sol.size() * sizeof(prec_t)));

    checkCudaErrors(cudaMemcpy(r_src_d, r_src.data(), r_src.size() * sizeof(prec_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(r_trg_d, r_trg.data(), r_trg.size() * sizeof(prec_t), cudaMemcpyHostToDevice));

    Timer timer;
    timer.start();
    driver_bulk<<<n_trg / n_threads_per_block, n_threads_per_block>>>(r_src_d, n_src, r_trg_d, n_trg, sol_d);
    cudaDeviceSynchronize();
    timer.stop();

    checkCudaErrors(cudaMemcpy(sol.data(), sol_d, sol.size() * sizeof(prec_t), cudaMemcpyDeviceToHost));

    cudaFree(r_src_d);
    cudaFree(r_trg_d);

    timer.stop();

    if (n_trg <= 128) {
        auto sol_h = driver_host(r_src, r_trg);
        for (int j = 0; j < n_trg; ++j) {
            prec_t err = fabs(1.0 - sol_h[j] / sol[j]);
            printf("%d %0.10g %0.10g %.10g %.10g\n", j, err, sol[j], sol_h[j], sol_h[j] - sol[j]);
        }
    }

    printf("timing: %g\n", timer.mSec());
    return 0;
}
