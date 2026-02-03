#include "tl_templates/pto/common.h"
#include <pto/pto-inst.hpp>
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace pto;

AICORE void add_kernel_kernel(__gm__ float *A_handle, __gm__ float *B_handle, __gm__ float *C_handle, uint64_t ffts_Addr) {
  auto cid = get_block_idx();
  set_ffts_base_addr(ffts_Addr);

  tl::ascend_pto::TileUbDataND<float, 64, 128, 64, 128> a_ub;
  TASSIGN(a_ub, 0);
  tl::ascend_pto::TileUbDataND<float, 64, 128, 64, 128> b_ub;
  TASSIGN(b_ub, 32768);
  tl::ascend_pto::TileUbDataND<float, 64, 128, 64, 128> c_ub;
  TASSIGN(c_ub, 65536);
  auto vid = get_subblockid();
#if defined(__DAV_C220_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    tl::ascend_pto::copy_gm_to_ub<float, float, 1, 1, 1, 64, 128, 1, 1, 1024 * 512, 1024, 1, 64, 128>(A_handle + ((((cid / 8) * 131072) + (vid * 65536)) + ((cid % 8) * 128)), a_ub);
    tl::ascend_pto::copy_gm_to_ub<float, float, 1, 1, 1, 64, 128, 1, 1, 1024 * 512, 1024, 1, 64, 128>(B_handle + ((((cid / 8) * 131072) + (vid * 65536)) + ((cid % 8) * 128)), b_ub);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    TADD(c_ub, a_ub, b_ub);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    tl::ascend_pto::copy_ub_to_gm<float, float, 1, 1, 1, 64, 128, 1, 1, 1024 * 512, 1024, 1, 64, 128, 64, 128>(C_handle + ((((cid / 8) * 131072) + (vid * 65536)) + ((cid % 8) * 128)), 65536, 0,4);
#endif
}

extern "C" __global__ AICORE void launch_kernel(__gm__ uint8_t *A_handle, __gm__ uint8_t *B_handle, __gm__ uint8_t *C_handle, uint64_t fftsAddr)
{
    add_kernel_kernel(reinterpret_cast<__gm__ float *>(A_handle),
     reinterpret_cast<__gm__ float *>(B_handle),
     reinterpret_cast<__gm__ float *>(C_handle),
     reinterpret_cast<uint64_t>(fftsAddr));
}

extern "C" void call(uint8_t *A_handle, uint8_t *B_handle, uint8_t *C_handle, void *stream)
{
    uint32_t fftsLen{0};
    uint64_t fftsAddr{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    launch_kernel<<<32, nullptr, stream>>>(A_handle, B_handle, C_handle, fftsAddr);
}
