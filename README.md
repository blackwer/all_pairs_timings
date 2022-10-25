I only really designed this for use at FI. It requires pvfmm >= 1.3, cuda, mpi, eigen, and some
kind of blas/fftw implementation. In no way do I claim that any of this is optimal (I actually
know much of it isn't). It's for instructional purposes only.

```bash
source setenv.sh
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native"
OMP_PROC_BIND=spread OMP_PLACES=threads ./all_pairs 1000 3162 10000 31623 100000 316228 1000000 3162278 > meas.dat
```
