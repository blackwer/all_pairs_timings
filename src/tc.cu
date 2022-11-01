#include <cstdlib>
#include <cuda.h>
#include <mma.h>
#include <cstdio>

#include <vector>

template <typename T>
void check(T result, char const *const func, const char *const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(result),
                cudaGetErrorName(result), func);
        exit(EXIT_FAILURE);
    }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

__global__ void dr2_driver(const float *r_src, const float *r_trg, float *__restrict__ dr_mat) {
    constexpr int warpsize = 32;
    constexpr int K = 8;
    constexpr int M = 16;
    constexpr int N = 16;

    const unsigned int warpId = threadIdx.x / warpsize;
    const unsigned int laneId = threadIdx.x % warpsize;

    using nvcuda::wmma::precision::tf32;
    __shared__ float buffer[256 + 32];
    const int i_trg = blockIdx.x * blockDim.x + threadIdx.x;
    float *A = buffer;
    float *B = buffer + 128;
    float *rmagsrc = buffer + 256;
    float *rmagtrg = buffer + 256 + 16;

    if (i_trg < 16) {
        for (int i = 0; i < 3; ++i)
            A[i_trg * K + i] = r_src[i_trg * 3 + i];

        for (int i = 0; i < 3; ++i)
            B[i * N + i_trg] = r_trg[i_trg * 3 + i];

        for (int i = 0; i < 3; ++i)
            rmagsrc[i_trg] += r_src[i_trg * 3 + i] * r_src[i_trg * 3 + i];

        for (int i = 0; i < 3; ++i)
            rmagtrg[i_trg] += r_trg[i_trg * 3 + i] * r_trg[i_trg * 3 + i];
    }

    using namespace nvcuda;
    wmma::fragment<wmma::matrix_a, 16, 16, 8, tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, tf32, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> c_frag;

    // Initialize the output to zero
    wmma::fill_fragment(c_frag, 0.0f);

    // Load the inputs
    wmma::load_matrix_sync(a_frag, A, K);
    wmma::load_matrix_sync(b_frag, B, N);

    for (int i = 0; i < a_frag.num_elements; ++i)
        a_frag.x[i] = wmma::__float_to_tf32(a_frag.x[i]);
    for (int i = 0; i < b_frag.num_elements; ++i)
        b_frag.x[i] = wmma::__float_to_tf32(b_frag.x[i]);
    
    // Perform the matrix multiplication
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    for (int i = 0; i < c_frag.num_elements; ++i)
        c_frag.x[i] *= -2;

    // Store the output
    wmma::store_matrix_sync(dr_mat, c_frag, N, wmma::mem_row_major);

    if (i_trg < 16) {
        for (int i_src = 0; i_src < 16; ++i_src) {
            dr_mat[i_src * 16 + i_trg] += rmagsrc[i_src] + rmagtrg[i_trg];
        }
    }
}

template <typename T>
T rinv_host(const T &r_src, const T &r_trg) {
    int n_trg = r_trg.size() / 3;
    int n_src = r_src.size() / 3;
    std::vector<float> dr2(n_src * n_trg);

    for (int i = 0; i < n_src; ++i) {
        for (int j = 0; j < n_trg; ++j) {
            float x = r_trg[j * 3 + 0] - r_src[i * 3 + 0];
            float y = r_trg[j * 3 + 1] - r_src[i * 3 + 1];
            float z = r_trg[j * 3 + 2] - r_src[i * 3 + 2];
            dr2[i * n_trg + j] = x * x + y * y + z * z;
        }
    }

    return dr2;
}

int main(int argc, char *argv[]) {

    constexpr int n_trg = 16;
    constexpr int n_src = 16;
    std::vector<float> r_src(3 * n_src);
    std::vector<float> r_trg(3 * n_trg);
    std::vector<float> dr2(n_src * n_trg);

    for (int i = 0; i < n_src; ++i)
        r_src[i * 3] = drand48();

    for (int i = 0; i < n_trg; ++i)
        r_trg[i * 3] = drand48();

    float *x_src_d, *x_trg_d, *dx_d;

    checkCudaErrors(cudaMalloc((void **)&x_src_d, r_src.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&x_trg_d, r_trg.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&dx_d, dr2.size() * sizeof(float)));

    checkCudaErrors(cudaMemcpy(x_src_d, r_src.data(), r_src.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(x_trg_d, r_trg.data(), r_trg.size() * sizeof(float), cudaMemcpyHostToDevice));

    dr2_driver<<<1, 32>>>(x_src_d, x_trg_d, dx_d);

    checkCudaErrors(cudaMemcpy(dr2.data(), dx_d, dr2.size() * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(x_src_d);
    cudaFree(x_trg_d);

    auto dr2_h = rinv_host(r_src, r_trg);
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            float err = fabs(1.0 - dr2_h[i * 16 + j] / dr2[i * 16 + j]);
            if (err > 1E-3)
                printf("%d %d %g %g %g\n", i, j, err, dr2[i * 16 + j], dr2_h[i * 16 + j]);
        }
    }

    return 0;
}
