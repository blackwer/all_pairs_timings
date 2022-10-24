#include <Eigen/Core>

#include <chrono>
#include <cstdlib>
#include <iostream>

#include <pvfmm.hpp>

#include "kernels.h"
#include "timer.hpp"

static auto g_memmgr = pvfmm::mem::MemoryManager(0);

template <typename T>
using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template <typename T>
using MatRef = Eigen::Ref<Mat<T>>;

inline std::pair<int, int> get_chunk_start_and_size(int i_thr, int n_thr, int prob_size) {
    const int chunk_small = prob_size / n_thr;
    const int chunk_big = chunk_small + 1;
    const int remainder = prob_size % n_thr;

    if (i_thr < remainder)
        return {chunk_big * i_thr, chunk_big};
    else
        return {remainder * chunk_big + (i_thr - remainder) * chunk_small, chunk_small};
}

template <typename T>
Mat<T> stokeslet_direct_cpu(const MatRef<T> &r_src, const MatRef<T> &r_trg, const MatRef<T> &f_src) {
    Mat<T> u_trg = Mat<T>::Zero(3, r_trg.cols());

#pragma omp parallel for schedule(static)
    for (int i = 0; i < omp_get_num_threads(); ++i) {
        auto [i_trg_0, n_trg] = get_chunk_start_and_size(i, omp_get_num_threads(), r_trg.cols());

        pvfmm::stokes_vel::Eval(const_cast<T *>(r_src.data()), r_src.cols(), const_cast<T *>(f_src.data()), 1,
                                const_cast<T *>(r_trg.data() + i_trg_0 * 3), n_trg, u_trg.data() + i_trg_0 * 3,
                                &g_memmgr);
    }
    return u_trg;
}

int main(int argc, char *argv[]) {
    int thread_level;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &thread_level);

    if (argc != 3)
        return EXIT_FAILURE;

    int n_src = atoi(argv[1]);
    int n_trg = atoi(argv[2]);

    Mat<double> r_src_d = Mat<double>::Random(3, n_src);
    Mat<double> f_src_d = Mat<double>::Random(3, n_src);
    Mat<double> r_trg_d = Mat<double>::Random(3, n_trg);

    Mat<float> r_src_f = r_src_d.cast<float>();
    Mat<float> f_src_f = f_src_d.cast<float>();
    Mat<float> r_trg_f = r_trg_d.cast<float>();

    Timer timer;
    timer.start();
    Mat<double> u_double = stokeslet_direct_cpu<double>(r_src_d, r_trg_d, f_src_d);
    timer.stop();
    std::cout << timer.elapsedMilliseconds() << std::endl;

    timer.start();
    Mat<float> u_float = stokeslet_direct_cpu<float>(r_src_f, r_trg_f, f_src_f);
    timer.stop();
    std::cout << timer.elapsedMilliseconds() << std::endl;

    for (int i = 0; i < 10; ++i) {
        Mat<float> u_float_gpu = u_float;
        kernels::stokeslet_direct_gpu(r_src_f.data(), f_src_f.data(), n_src, r_trg_f.data(), u_float_gpu.data(), n_trg);
    }

    timer.start();
    Mat<float> u_float_gpu = u_float;
    kernels::stokeslet_direct_gpu(r_src_f.data(), f_src_f.data(), n_src, r_trg_f.data(), u_float_gpu.data(), n_trg);
    timer.stop();
    std::cout << timer.elapsedMilliseconds() << std::endl;

    timer.start();
    Mat<double> u_double_gpu = u_double;
    kernels::stokeslet_direct_gpu(r_src_d.data(), f_src_d.data(), n_src, r_trg_d.data(), u_double_gpu.data(), n_trg);
    timer.stop();
    std::cout << timer.elapsedMilliseconds() << std::endl;

    return 0;
}
