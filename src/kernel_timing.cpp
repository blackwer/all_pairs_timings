#include <Eigen/Core>

#include <iostream>
#include <string>
#include <vector>

#include <pvfmm.hpp>

#include "kernels.hpp"
#include "timer.hpp"

template <typename T>
using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template <typename T>
using MatRef = Eigen::Ref<Mat<T>>;

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

template <typename T>
std::vector<T> mat2vec(const MatRef<T> &src) {
    std::vector<T> res(src.size());
    for (int i = 0; i < res.size(); ++i)
        res[i] = src.data()[i];
    return res;
}

void print_meas(const std::string &device, const std::string &algo, const std::string &precision, int n_trg,
                double tree, double eval, double tot) {
    using std::to_string;
    std::cout << device + "," + algo + "," + precision + "," + to_string(n_trg) + "," + to_string(tree) + "," +
                     to_string(eval) + "," + to_string(tot) + "\n";
}

int main(int argc, char *argv[]) {
    int thread_level;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &thread_level);
    auto memmgr = pvfmm::mem::MemoryManager(10000000);

    std::vector<int> n_trgs;
    for (int i = 1; i < argc; ++i)
        n_trgs.push_back(std::stoi(argv[i]));

    {
        // Warmup!
        int n_src = 1000;
        int n_trg = n_src;
        Mat<double> r_src_d = 0.5 * (Mat<double>::Random(3, n_src) + Mat<double>::Ones(3, n_src));
        Mat<double> f_src_d = 0.5 * (Mat<double>::Random(3, n_src) + Mat<double>::Ones(3, n_src));
        Mat<double> r_trg_d = 0.5 * (Mat<double>::Random(3, n_trg) + Mat<double>::Ones(3, n_trg));

        stokeslet_direct_cpu<double>(r_src_d, r_trg_d, f_src_d);
        stokeslet_direct_gpu_tiled(r_src_d, f_src_d, r_trg_d);
    }

    std::cout << "Device,precision,ntrg,tree,eval,tot\n";
    for (const auto &n_trg : n_trgs) {
        int n_src = n_trg;
        Mat<double> r_src_d = 0.5 * (Mat<double>::Random(3, n_src) + Mat<double>::Ones(3, n_src));
        Mat<double> f_src_d = 0.5 * (Mat<double>::Random(3, n_src) + Mat<double>::Ones(3, n_src));
        Mat<double> r_trg_d = 0.5 * (Mat<double>::Random(3, n_trg) + Mat<double>::Ones(3, n_trg));

        Mat<float> r_src_f = r_src_d.cast<float>().eval();
        Mat<float> f_src_f = f_src_d.cast<float>().eval();
        Mat<float> r_trg_f = r_trg_d.cast<float>().eval();

    
        Timer timer;

        timer.start();
        Mat<float> u_float = stokeslet_direct_cpu<float>(r_src_f, r_trg_f, f_src_f);
        timer.stop();
        print_meas("cpu", "blocked", "float", n_trg, 0, timer.mSec(), timer.mSec());

        timer.start();
        Mat<double> u_double = stokeslet_direct_cpu<double>(r_src_d, r_trg_d, f_src_d);
        timer.stop();
        print_meas("cpu", "blocked", "double", n_trg, 0, timer.mSec(), timer.mSec());

        timer.start();
        Mat<float> u_float_gpu_untiled = stokeslet_direct_gpu_untiled(r_src_f, f_src_f, r_trg_f);
        timer.stop();
        print_meas("gpu", "direct", "float", n_trg, 0, timer.mSec(), timer.mSec());

        timer.start();
        Mat<double> u_double_gpu_untiled = stokeslet_direct_gpu_untiled(r_src_d, f_src_d, r_trg_d);
        timer.stop();
        print_meas("gpu", "direct","double", n_trg, 0, timer.mSec(), timer.mSec());

        timer.start();
        Mat<float> u_float_gpu_tiled = stokeslet_direct_gpu_tiled(r_src_f, f_src_f, r_trg_f);
        timer.stop();
        print_meas("gpu", "blocked", "float", n_trg, 0, timer.mSec(), timer.mSec());

        timer.start();
        Mat<double> u_double_gpu_tiled = stokeslet_direct_gpu_tiled(r_src_d, f_src_d, r_trg_d);
        timer.stop();
        print_meas("gpu", "blocked", "double", n_trg, 0, timer.mSec(), timer.mSec());

        // Construct tree.
        size_t max_pts = 2000;
        int mult_order = 8;

        const pvfmm::Kernel<float> &kernel_fn_f = pvfmm::StokesKernel<float>::velocity();
        const pvfmm::Kernel<double> &kernel_fn_d = pvfmm::StokesKernel<double>::velocity();

        // Load matrices.
        pvfmm::PtFMM<float> matrices_f(&memmgr);
        matrices_f.Initialize(mult_order, MPI_COMM_WORLD, &kernel_fn_f);

        pvfmm::PtFMM<double> matrices_d(&memmgr);
        matrices_d.Initialize(mult_order, MPI_COMM_WORLD, &kernel_fn_d);

        auto r_src_f_vec = mat2vec<float>(r_src_f);
        auto f_src_f_vec = mat2vec<float>(f_src_f);
        auto r_trg_f_vec = mat2vec<float>(r_trg_f);
        std::vector<float> dummy_f;

        auto r_src_d_vec = mat2vec<double>(r_src_d);
        auto f_src_d_vec = mat2vec<double>(f_src_d);
        auto r_trg_d_vec = mat2vec<double>(r_trg_d);
        std::vector<double> dummy_d;

        timer.start();
        auto *tree_f = PtFMM_CreateTree(r_src_f_vec, r_src_f_vec, dummy_f, dummy_f, r_src_f_vec, MPI_COMM_WORLD,
                                        max_pts, pvfmm::FreeSpace);

        // FMM Setup
        tree_f->SetupFMM(&matrices_f);
        timer.stop();
        auto tree_time = timer.mSec();

        // Run FMM
        timer.start();
        std::vector<float> u_fmm_f;
        PtFMM_Evaluate(tree_f, u_fmm_f, n_trg);
        timer.stop();
        auto eval_time = timer.mSec();
        print_meas("cpu", "fmm", "float", n_trg, tree_time, eval_time, tree_time + eval_time);

        timer.start();
        auto *tree_d = PtFMM_CreateTree(r_src_d_vec, r_src_d_vec, dummy_d, dummy_d, r_src_d_vec, MPI_COMM_WORLD,
                                        max_pts, pvfmm::FreeSpace);

        // FMM Setup
        tree_d->SetupFMM(&matrices_d);
        timer.stop();
        tree_time = timer.mSec();

        timer.start();
        // Run FMM
        std::vector<double> u_fmm_d;
        PtFMM_Evaluate(tree_d, u_fmm_d, n_trg);
        timer.stop();
        eval_time = timer.mSec();
        print_meas("cpu", "fmm", "double", n_trg, tree_time, eval_time, tree_time + eval_time);
    }
    return 0;
}
