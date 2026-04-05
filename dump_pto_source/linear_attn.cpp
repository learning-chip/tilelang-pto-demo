#include "tl_templates/pto/common.h"
#include <pto/pto-inst.hpp>
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace pto;

AICORE void main_kernel(__gm__ half *Q_handle, __gm__ half *K_handle, __gm__ half *V_handle, __gm__ half *workspace_1_handle, __gm__ half *workspace_2_handle, __gm__ half *O_handle, uint64_t ffts_Addr) {
  auto cid = get_block_idx();
  set_ffts_base_addr(ffts_Addr);

  tl::ascend_pto::TileMatL1<half, 64, 128, 64, 128> q_l1;
  TASSIGN(q_l1, 0);
  tl::ascend_pto::TileMatL1<half, 64, 128, 64, 128> k_l1;
  TASSIGN(k_l1, 16384);
  tl::ascend_pto::TileMatL1<half, 64, 128, 64, 128> v_l1;
  TASSIGN(v_l1, 32768);
  tl::ascend_pto::TileMatL1<half, 128, 128, 128, 128> h_l1;
  TASSIGN(h_l1, 49152);
  TileAcc<float, 64, 64, 64, 64> acc_l0;
  TASSIGN(acc_l0, 0);
  TileAcc<float, 128, 128, 128, 128> h_l0;
  TASSIGN(h_l0, 16384);
  tl::ascend_pto::TileMatL1<half, 64, 64, 64, 64> acc_l1;
  TASSIGN(acc_l1, 81920);
  TileAcc<float, 64, 128, 64, 128> o_l0;
  TASSIGN(o_l0, 81920);
  tl::ascend_pto::TileUbDataND<half, 64, 128, 64, 128> hsum_ub;
  TASSIGN(hsum_ub, 0);
  tl::ascend_pto::TileUbDataND<half, 32, 64, 32, 64> zero_ub;
  TASSIGN(zero_ub, 16384);
  tl::ascend_pto::TileUbDataND<half, 32, 64, 32, 64> acc_ub;
  TASSIGN(acc_ub, 20480);
  tl::ascend_pto::TileUbDataND<half, 64, 128, 64, 128> h_ub;
  TASSIGN(h_ub, 24576);
  auto vid = get_subblockid();
#if defined(__DAV_C220_CUBE__)

  for (int32_t i = 0; i < 8; ++i) {
      set_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID1);
      wait_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID1);
      tl::ascend_pto::copy_gm_to_l1<half, half, 1, 1, 1, 64, 128, 262144, 131072, 65536, 128, 1, 64, 128>(Q_handle + ((cid * 65536) + (i * 8192)), 0, 0, 64, 128);
      tl::ascend_pto::copy_gm_to_l1<half, half, 1, 1, 1, 64, 128, 262144, 131072, 65536, 128, 1, 64, 128>(K_handle + ((cid * 65536) + (i * 8192)), 16384, 0, 64, 128);
      tl::ascend_pto::copy_gm_to_l1<half, half, 1, 1, 1, 64, 128, 262144, 131072, 65536, 128, 1, 64, 128>(V_handle + ((cid * 65536) + (i * 8192)), 32768, 0, 64, 128);
      tl::ascend_pto::copy_gm_to_l1<half, half, 1, 1, 1, 128, 128, 65536, 32768, 16384, 128, 1, 128, 128>(workspace_2_handle + (cid * 16384), 49152, 0, 128, 128);
      set_flag(PIPE_MTE2, PIPE_M, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_M, EVENT_ID1);
      tl::ascend_pto::gemm_v0<half, float, 64, 64, 128, 64, 64, 128, 128, false, true>(q_l1, k_l1, acc_l0, (bool)1);
      set_flag(PIPE_M, PIPE_FIX, EVENT_ID2);
      wait_flag(PIPE_M, PIPE_FIX, EVENT_ID2);
      tl::ascend_pto::copy_l0c_to_gm<half, float, 1, 1, 1, 64, 64, 16384, 8192, 4096, 64, 1, 64, 64>(workspace_1_handle + (cid * 4096), 0, 0, 64, 64);
      set_flag(PIPE_MTE2, PIPE_M, EVENT_ID3);
      wait_flag(PIPE_MTE2, PIPE_M, EVENT_ID3);
      pipe_barrier(PIPE_M);
      tl::ascend_pto::gemm_v0<half, float, 128, 128, 64, 128, 128, 64, 64, true, false>(k_l1, v_l1, h_l0, (bool)1);
      set_flag(PIPE_MTE2, PIPE_FIX, EVENT_ID4);
      wait_flag(PIPE_MTE2, PIPE_FIX, EVENT_ID4);
      set_flag(PIPE_M, PIPE_FIX, EVENT_ID5);
      wait_flag(PIPE_M, PIPE_FIX, EVENT_ID5);
      tl::ascend_pto::copy_l0c_to_gm<half, float, 1, 1, 1, 128, 128, 65536, 32768, 16384, 128, 1, 128, 128>(workspace_2_handle + (cid * 16384), 16384, 0, 128, 128);
      tl::ascend_pto::set_cross_flag<PIPE_FIX>(0, 2);
      tl::ascend_pto::wait_cross_flag(1);
      set_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID6);
      wait_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID6);
      tl::ascend_pto::copy_gm_to_l1<half, half, 1, 1, 1, 64, 64, 16384, 8192, 4096, 64, 1, 64, 64>(workspace_1_handle + (cid * 4096), 81920, 0, 64, 64);
      set_flag(PIPE_MTE2, PIPE_M, EVENT_ID7);
      wait_flag(PIPE_MTE2, PIPE_M, EVENT_ID7);
      tl::ascend_pto::gemm_v0<half, float, 64, 128, 64, 64, 128, 64, 64, false, false>(acc_l1, v_l1, o_l0, (bool)1);
      pipe_barrier(PIPE_M);
      tl::ascend_pto::gemm_v0<half, float, 64, 128, 128, 64, 128, 128, 128, false, false>(q_l1, h_l1, o_l0, (bool)0);
      set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
      pipe_barrier(PIPE_FIX);
      tl::ascend_pto::copy_l0c_to_gm<half, float, 1, 1, 1, 64, 128, 262144, 131072, 65536, 128, 1, 64, 128>(O_handle + ((cid * 65536) + (i * 8192)), 81920, 0, 64, 128);
    }
#endif
#if defined(__DAV_C220_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    TEXPANDS(hsum_ub, 0.000000e+00f);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    TEXPANDS(zero_ub, 0.000000e+00f);

  for (int32_t i_1 = 0; i_1 < 8; ++i_1) {
      tl::ascend_pto::wait_cross_flag(0);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      tl::ascend_pto::copy_gm_to_ub<half, half, 1, 1, 1, 32, 64, 16384, 8192, 4096, 64, 1, 32, 64, pto::PadValue::Zero>(workspace_1_handle + ((cid * 4096) + (vid * 2048)), 20480, 0, 32, 64);
      tl::ascend_pto::copy_gm_to_ub<half, half, 1, 1, 1, 64, 128, 65536, 32768, 16384, 128, 1, 64, 128, pto::PadValue::Zero>(workspace_2_handle + ((cid * 16384) + (vid * 8192)), 24576, 0, 64, 128);

  for (int32_t j = 0; j < 32; ++j) {

  for (int32_t k = 0; k < 64; ++k) {
          pipe_barrier(PIPE_ALL);
          if (((vid * 32) + j) < k) {
            acc_ub.SetValue(((j * 64) + k), zero_ub.GetValue(((j * 64) + k)));
          }
          pipe_barrier(PIPE_ALL);
          pipe_barrier(PIPE_ALL);
        }
      }
      TADD(hsum_ub, hsum_ub, h_ub);
      tl::ascend_pto::copy_ub_to_gm<half, half, 1, 1, 1, 32, 64, 16384, 8192, 4096, 64, 1, 32, 64>(workspace_1_handle + ((cid * 4096) + (vid * 2048)), 20480, 0, 32, 64);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
      tl::ascend_pto::copy_ub_to_gm<half, half, 1, 1, 1, 64, 128, 65536, 32768, 16384, 128, 1, 64, 128>(workspace_2_handle + ((cid * 16384) + (vid * 8192)), 0, 0, 64, 128);
      tl::ascend_pto::set_cross_flag<PIPE_MTE3>(1, 2);
    }
#endif
}

extern "C" __global__ AICORE void launch_kernel(__gm__ uint8_t *Q_handle, __gm__ uint8_t *K_handle, __gm__ uint8_t *V_handle, __gm__ uint8_t *workspace_1_handle, __gm__ uint8_t *workspace_2_handle, __gm__ uint8_t *O_handle, uint64_t fftsAddr)
{
    main_kernel(reinterpret_cast<__gm__ half *>(Q_handle),
     reinterpret_cast<__gm__ half *>(K_handle),
     reinterpret_cast<__gm__ half *>(V_handle),
     reinterpret_cast<__gm__ half *>(workspace_1_handle),
     reinterpret_cast<__gm__ half *>(workspace_2_handle),
     reinterpret_cast<__gm__ half *>(O_handle),
     reinterpret_cast<uint64_t>(fftsAddr));
}

extern "C" void call(uint8_t *Q_handle, uint8_t *K_handle, uint8_t *V_handle, uint8_t *workspace_1_handle, uint8_t *workspace_2_handle, uint8_t *O_handle, void *stream)
{
    uint32_t fftsLen{0};
    uint64_t fftsAddr{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    launch_kernel<<<4, nullptr, stream>>>(Q_handle, K_handle, V_handle, workspace_1_handle, workspace_2_handle, O_handle, fftsAddr);
}
