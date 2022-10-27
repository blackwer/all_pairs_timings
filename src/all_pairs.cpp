#include <pvfmm.hpp>

#include "all_pairs.hpp"
#include "kernels.hpp"

Mat<double> stokeslet_direct_gpu_tiled(const MatRef<double> &r_src, const MatRef<double> &f_src,
                                       const MatRef<double> &r_trg) {
    Mat<double> u_trg = Mat<double>::Zero(3, r_trg.cols());
    kernels::stokeslet_direct_gpu_tiled(r_src.data(), f_src.data(), r_src.cols(), r_trg.data(), u_trg.data(),
                                        u_trg.cols());
    return u_trg;
}

Mat<double> stokeslet_direct_gpu_untiled(const MatRef<double> &r_src, const MatRef<double> &f_src,
                                         const MatRef<double> &r_trg) {
    Mat<double> u_trg = Mat<double>::Zero(3, r_trg.cols());
    kernels::stokeslet_direct_gpu_untiled(r_src.data(), f_src.data(), r_src.cols(), r_trg.data(), u_trg.data(),
                                          u_trg.cols());
    return u_trg;
}

Mat<float> stokeslet_direct_gpu_tiled(const MatRef<float> &r_src, const MatRef<float> &f_src,
                                      const MatRef<float> &r_trg) {
    Mat<float> u_trg = Mat<float>::Zero(3, r_trg.cols());

    kernels::stokeslet_direct_gpu_tiled(r_src.data(), f_src.data(), r_src.cols(), r_trg.data(), u_trg.data(),
                                        u_trg.cols());
    return u_trg;
}

Mat<float> stokeslet_direct_gpu_untiled(const MatRef<float> &r_src, const MatRef<float> &f_src,
                                        const MatRef<float> &r_trg) {
    Mat<float> u_trg = Mat<float>::Zero(3, r_trg.cols());
    kernels::stokeslet_direct_gpu_untiled(r_src.data(), f_src.data(), r_src.cols(), r_trg.data(), u_trg.data(),
                                          u_trg.cols());
    return u_trg;
}

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
    static auto memmgr = pvfmm::mem::MemoryManager(0);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < omp_get_num_threads(); ++i) {
        auto [i_trg_0, n_trg] = get_chunk_start_and_size(i, omp_get_num_threads(), r_trg.cols());

        pvfmm::stokes_vel::Eval(const_cast<T *>(r_src.data()), r_src.cols(), const_cast<T *>(f_src.data()), 1,
                                const_cast<T *>(r_trg.data() + i_trg_0 * 3), n_trg, u_trg.data() + i_trg_0 * 3,
                                &memmgr);
    }
    return u_trg;
}

template Mat<float> stokeslet_direct_cpu<float>(const MatRef<float> &r_src, const MatRef<float> &r_trg,
                                                const MatRef<float> &f_src);
template Mat<double> stokeslet_direct_cpu<double>(const MatRef<double> &r_src, const MatRef<double> &r_trg,
                                                  const MatRef<double> &f_src);
