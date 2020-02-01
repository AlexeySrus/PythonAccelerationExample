import numpy as np
from timeit import default_timer as time
import argparse
from tqdm import tqdm
import numba
from numba import cuda, float32
import ctypes
import torch


def matrix_product(a, b):
    res = np.zeros((len(a), len(b)), dtype=np.float32)

    for i in range(len(a)):
        for j in range(len(b)):
            s = 0
            for k in range(len(a[i])):
                s += a[i][k] * b[k][j]

    return res


@numba.jit(nopython=True, parallel=False)
def matrix_product_numba(a, b):
    res = np.zeros((len(a), len(b)), dtype=np.float32)

    for i in range(len(a)):
        for j in range(len(b)):
            s = 0
            for k in range(len(a[i])):
                s += a[i][k] * b[k][j]
            res[i][j] = s

    return res


@numba.jit(nopython=True, parallel=True)
def matrix_product_numba_parallel(a, b):
    res = np.zeros((len(a), len(b)), dtype=np.float32)

    for i in numba.prange(len(a)):
        for j in range(len(b)):
            s = 0
            for k in range(len(a[i])):
                s += a[i][k] * b[k][j]
            res[i][j] = s

    return res


TPB = 16

# Numba example
@cuda.jit
def matrix_product_numba_cuda(A, B, C):
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid

    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.
    for i in range(bpg):
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()

    C[x, y] = tmp

@torch.jit.script
def matrix_product_torch(a, b):
    res = torch.zeros((len(a), len(b)), dtype=torch.float32)

    for i in range(len(a)):
        for j in range(len(b)):
            s = torch.zeros(1)
            for k in range(len(a[i])):
                s += a[i][k] * b[k][j]

    return res


libmatmul = ctypes.CDLL('./libmatmul.so')
libmatmul.matmul.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int
]

libmatmul.parallel_matmul.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int
]


def matrix_product_c(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    res = np.zeros_like(a)

    libmatmul.matmul(
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        res.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        a.shape[0],
    )

    return res


def matrix_product_c_parallel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    res = np.zeros_like(a)

    libmatmul.parallel_matmul(
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        res.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        a.shape[0],
    )

    return res


def parse_args():
    args_parser = argparse.ArgumentParser(
        description='Test frogram for presentation'
    )

    args_parser.add_argument('n', type=int, default=5)

    return args_parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    n = args.n

    with open('time_results.txt', 'w') as f:
        f.write(
            'N, numpy, pytorch_time, numba, parallel numba, cuda_numba_time, C, C with OpenMP\n'
        )

        for n in tqdm(list(range(1, 100, 10)) + list(range(100, 1000, 100)) + list(range(1000, 10001, 1000))):
            # print('\nN = {}'.format(n))

            start_time = time()
            # a = [
            #     [i / (j + 1) for j in range(n)]
            #     for i in range(n)
            # ]
            # a = np.array(a, dtype=np.float32)
            a = np.random.rand(n, n).astype(np.float32)
            b = np.random.rand(n, n).astype(np.float32)
            finish_time = time()
            generation_time = finish_time - start_time
            # print('Matrix generation time: {:.2f}'.format(generation_time))

            # start_time = time()
            # c = matrix_product(a, b)
            # finish_time = time()
            # print('Multiplication 1 time: {:.2f}'.format(finish_time - start_time))

            # start_time = time()
            # c = matrix_product_torch(torch.FloatTensor(a), torch.FloatTensor(b))
            # finish_time = time()
            # print('Multiplication torch time: {:.2f}'.format(finish_time - start_time))

            start_time = time()
            c = a @ b
            finish_time = time()
            numpy_time = finish_time - start_time
            # print('Multiplication numpy time: {:.2f}'.format(numpy_time))

            ta = torch.FloatTensor(a)
            tb = torch.FloatTensor(b)
            start_time = time()
            c = ta @ tb
            finish_time = time()
            pytorch_time = finish_time - start_time
            # print('Multiplication pytorch time: {:.2f}'.format(pytorch_time))

            start_time = time()
            c = matrix_product_numba(a, b)
            finish_time = time()
            numba_time = finish_time - start_time
            # print('Multiplication numba time: {:.2f}'.format(numba_time))

            start_time = time()
            c = matrix_product_numba_parallel(a, b)
            finish_time = time()
            parallel_numba_time = finish_time - start_time
            # print('Multiplication numba parallel time: {:.2f}'.format(parallel_numba_time))

            start_time = time()
            c = matrix_product_numba_parallel(a, b)
            finish_time = time()
            cuda_numba_time = finish_time - start_time
            # print('Multiplication numba cuda time: {:.2f}'.format(cuda_numba_time))

            start_time = time()
            c = matrix_product_c(a, b)
            finish_time = time()
            c_time = finish_time - start_time
            # print('Multiplication C time: {:.2f}'.format(c_time))

            start_time = time()
            c = matrix_product_c_parallel(a, b)
            finish_time = time()
            parallel_c_time = finish_time - start_time
            # print('Multiplication C with OpenMP time: {:.2f}'.format(parallel_c_time))

            f.write(
                '{}, {:.7f}, {:.7f}, {:.7f}, {:.7f}, {:.7f}, {:.7f}, {:.7f}\n'.format(
                    n,
                    numpy_time,
                    pytorch_time,
                    numba_time,
                    parallel_numba_time,
                    cuda_numba_time,
                    c_time,
                    parallel_c_time
                )
            )

    # for line in matrix_product(a, a):
    #     print(line)
