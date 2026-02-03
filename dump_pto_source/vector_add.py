# https://github.com/tile-ai/tilelang-ascend/blob/d2405b5975edaa36166b702d2ef8c1952b7b7337/examples/elementwise/elementwise_add_developer.py

import torch
import tilelang
import tilelang.language as T
from tilelang import jit

from patch_libgen import get_patched_compile_lib
from tilelang.jit.adapter.libgen import LibraryGenerator

patched_compile_lib = get_patched_compile_lib(
    src_dump_path="vector_add.cpp"
)
LibraryGenerator.compile_lib = patched_compile_lib  # monkey-patch

tilelang.disable_cache()  # ensure the (patched) compile pass is always triggered

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}

M = 512
N = 1024
block_M = 128
block_N = 128

VEC_NUM = 2

@jit(out_idx=[-1], pass_configs=pass_configs, target="pto")
def tile_add(M: int, N: int, block_M: int, block_N: int, dtype: str = 'float'):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def add_kernel(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            
            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

            for i, j in T.Parallel(block_M // VEC_NUM, block_N):
                c_ub[i, j] = a_ub[i, j] + b_ub[i, j]

            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return add_kernel

func = tile_add(M, N, block_M, block_N)

torch.manual_seed(0)

a = torch.randn(M, N).npu()
b = torch.randn(M, N).npu()

torch.npu.synchronize()
print("init successful!")

c = func(a, b)

ref_c = a + b

torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
print("Kernel Output Match!")
