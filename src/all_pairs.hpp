#ifndef ALL_PAIRS_HPP
#define ALL_PAIRS_HPP

#include <Eigen/Core>
#include <pvfmm.hpp>

template <typename T>
using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template <typename T>
using MatRef = Eigen::Ref<Mat<T>>;

Mat<double> stokeslet_direct_gpu_tiled(const MatRef<double> &r_src, const MatRef<double> &f_src,
                                       const MatRef<double> &r_trg);
Mat<double> stokeslet_direct_gpu_untiled(const MatRef<double> &r_src, const MatRef<double> &f_src,
                                         const MatRef<double> &r_trg);
Mat<float> stokeslet_direct_gpu_tiled(const MatRef<float> &r_src, const MatRef<float> &f_src,
                                      const MatRef<float> &r_trg);
Mat<float> stokeslet_direct_gpu_untiled(const MatRef<float> &r_src, const MatRef<float> &f_src,
                                        const MatRef<float> &r_trg);
template <typename T>
Mat<T> stokeslet_direct_cpu(const MatRef<T> &r_src, const MatRef<T> &r_trg, const MatRef<T> &f_src);

#endif
