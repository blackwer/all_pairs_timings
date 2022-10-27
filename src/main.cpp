#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "all_pairs.hpp"
#include "timer.hpp"

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

    std::set<std::string> flags;
    std::vector<int> n_trgs;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg.substr(0, 2) == "--")
            flags.insert(arg.substr(2, arg.length()));
        else
            n_trgs.push_back(std::stoi(arg));
    }

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

    std::cout << "device,algo,prec,ntrg,tree,eval,tot\n";
    for (const auto &n_trg : n_trgs) {
        int n_src = n_trg;
        Mat<double> r_src_d = 0.5 * (Mat<double>::Random(3, n_src) + Mat<double>::Ones(3, n_src));
        Mat<double> f_src_d = 0.5 * (Mat<double>::Random(3, n_src) + Mat<double>::Ones(3, n_src));
        Mat<double> r_trg_d = 0.5 * (Mat<double>::Random(3, n_trg) + Mat<double>::Ones(3, n_trg));

        Mat<float> r_src_f = r_src_d.cast<float>().eval();
        Mat<float> f_src_f = f_src_d.cast<float>().eval();
        Mat<float> r_trg_f = r_trg_d.cast<float>().eval();

        Timer timer;

        if (flags.count("gpu")) {
            if (flags.count("float")) {
                timer.start();
                Mat<float> u_float_gpu_untiled = stokeslet_direct_gpu_untiled(r_src_f, f_src_f, r_trg_f);
                timer.stop();
                print_meas("gpu", "direct", "float", n_trg, 0, timer.mSec(), timer.mSec());
            }

            if (flags.count("double")) {
                timer.start();
                Mat<double> u_double_gpu_untiled = stokeslet_direct_gpu_untiled(r_src_d, f_src_d, r_trg_d);
                timer.stop();
                print_meas("gpu", "direct", "double", n_trg, 0, timer.mSec(), timer.mSec());
            }

            if (flags.count("float")) {
                timer.start();
                Mat<float> u_float_gpu_tiled = stokeslet_direct_gpu_tiled(r_src_f, f_src_f, r_trg_f);
                timer.stop();
                print_meas("gpu", "blocked", "float", n_trg, 0, timer.mSec(), timer.mSec());
            }

            if (flags.count("double")) {
                timer.start();
                Mat<double> u_double_gpu_tiled = stokeslet_direct_gpu_tiled(r_src_d, f_src_d, r_trg_d);
                timer.stop();
                print_meas("gpu", "blocked", "double", n_trg, 0, timer.mSec(), timer.mSec());
            }
        }

        if (flags.count("cpu")) {
            if (flags.count("float")) {
                timer.start();
                Mat<float> u_float = stokeslet_direct_cpu<float>(r_src_f, r_trg_f, f_src_f);
                timer.stop();
                print_meas("cpu", "blocked", "float", n_trg, 0, timer.mSec(), timer.mSec());
            }

            if (flags.count("double")) {
                timer.start();
                Mat<double> u_double = stokeslet_direct_cpu<double>(r_src_d, r_trg_d, f_src_d);
                timer.stop();
                print_meas("cpu", "blocked", "double", n_trg, 0, timer.mSec(), timer.mSec());
            }

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

            if (flags.count("float")) {
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
            }

            if (flags.count("double")) {
                timer.start();
                auto *tree_d = PtFMM_CreateTree(r_src_d_vec, r_src_d_vec, dummy_d, dummy_d, r_src_d_vec, MPI_COMM_WORLD,
                                                max_pts, pvfmm::FreeSpace);

                // FMM Setup
                tree_d->SetupFMM(&matrices_d);
                timer.stop();
                auto tree_time = timer.mSec();

                timer.start();
                // Run FMM
                std::vector<double> u_fmm_d;
                PtFMM_Evaluate(tree_d, u_fmm_d, n_trg);
                timer.stop();
                auto eval_time = timer.mSec();
                print_meas("cpu", "fmm", "double", n_trg, tree_time, eval_time, tree_time + eval_time);
            }
        }
    }

    return 0;
}
