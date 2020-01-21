// #include <omp.h>
#include <iostream>

#ifdef __cplusplus
extern "C" {
#endif

void matmul(const float *a, const float *b, float *c, const int n)
{
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            float s = 0;
            for (std::size_t k = 0; k < n; ++k) {
                s += a[i*n + k] * b[k*n + j];
            }
            c[i*n + j] = s;
        }
    }
}

#ifdef __cplusplus
}
#endif