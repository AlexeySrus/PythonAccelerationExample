import numpy as np
from timeit import default_timer as time
import argparse
import numba
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


def matrix_product_c(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    res = np.zeros_like(a)

    libmatmul.matmul(
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

    start_time = time()
    a = [
        [i / (j + 1) for j in range(n)]
        for i in range(n)
    ]
    a = np.array(a, dtype=np.float32)
    finish_time = time()
    print('Matrix generation time: {:.2f}'.format(finish_time - start_time))

    start_time = time()
    c = matrix_product(a, a)
    finish_time = time()
    print('Multiplication 1 time: {:.2f}'.format(finish_time - start_time))

    start_time = time()
    c = matrix_product_torch(torch.FloatTensor(a), torch.FloatTensor(a))
    finish_time = time()
    print('Multiplication torch time: {:.2f}'.format(finish_time - start_time))

    start_time = time()
    c = a @ a
    finish_time = time()
    print('Multiplication numpy time: {:.2f}'.format(finish_time - start_time))

    start_time = time()
    c = matrix_product_numba(a, a)
    finish_time = time()
    print('Multiplication numba time: {:.2f}'.format(finish_time - start_time))

    start_time = time()
    c = matrix_product_numba_parallel(a, a)
    finish_time = time()
    print('Multiplication numba parallel time: {:.2f}'.format(finish_time - start_time))

    start_time = time()
    c = matrix_product_c(a, a)
    finish_time = time()
    print('Multiplication C time: {:.2f}'.format(finish_time - start_time))




    # for line in matrix_product(a, a):
    #     print(line)
