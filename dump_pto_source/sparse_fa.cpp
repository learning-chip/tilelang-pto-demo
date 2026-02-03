#include "tl_templates/pto/common.h"
#include <pto/pto-inst.hpp>
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace pto;

AICORE void main_kernel(__gm__ half *Q_handle, __gm__ half *KV_handle, __gm__ int *Indices_handle, __gm__ half *Output_handle, __gm__ half *workspace_1_handle, __gm__ float *workspace_2_handle, __gm__ half *workspace_3_handle, __gm__ float *workspace_4_handle, int64_t batch, int64_t seq_len, int64_t seq_len_kv, uint64_t ffts_Addr) {
  auto cid = get_block_idx();
  set_ffts_base_addr(ffts_Addr);

  tl::ascend_pto::TileMatL1<half, 16, 128, 16, 128> q_l1;
  TASSIGN(q_l1, 0);
  tl::ascend_pto::TileMatL1<half, 64, 128, 64, 128> kv_l1;
  TASSIGN(kv_l1, 4096);
  TileAcc<float, 16, 64, 16, 64> acc_s_l0c;
  TASSIGN(acc_s_l0c, 0);
  tl::ascend_pto::TileMatL1<half, 16, 64, 16, 64> acc_s_l1;
  TASSIGN(acc_s_l1, 20480);
  TileAcc<float, 16, 128, 16, 128> acc_o_l0c;
  TASSIGN(acc_o_l0c, 4096);
  tl::ascend_pto::TileUbDataND<float, 8, 128, 8, 128> acc_o;
  TASSIGN(acc_o, 0);
  tl::ascend_pto::TileUbDataND<float, 1, 8, 1, 8> sumexp;
  TASSIGN(sumexp, 4096);
  tl::ascend_pto::TileUbDataND<float, 1, 8, 1, 8> m_i;
  TASSIGN(m_i, 4128);
  tl::ascend_pto::TileUbDataND<int, 1, 64, 1, 64> indices_ub_;
  TASSIGN(indices_ub_, 4160);
  tl::ascend_pto::TileUbDataND<float, 8, 64, 8, 64> acc_s_ub;
  TASSIGN(acc_s_ub, 4672);
  tl::ascend_pto::TileUbDataND<float, 1, 8, 1, 8> m_i_prev;
  TASSIGN(m_i_prev, 6720);
  tl::ascend_pto::TileUbDataND<float, 8, 64, 8, 64> acc_s_ub_;
  TASSIGN(acc_s_ub_, 6752);
  tl::ascend_pto::TileUbDataND<uint8_t, 1, 6144, 1, 6144> tmp_ub;
  TASSIGN(tmp_ub, 8800);
  tl::ascend_pto::TileUbDataND<float, 1, 8, 1, 8> sumexp_i_ub;
  TASSIGN(sumexp_i_ub, 14944);
  tl::ascend_pto::TileUbDataND<half, 8, 64, 8, 64> acc_s_half;
  TASSIGN(acc_s_half, 14976);
  tl::ascend_pto::TileUbDataND<float, 8, 128, 8, 128> acc_o_ub;
  TASSIGN(acc_o_ub, 16000);
  tl::ascend_pto::TileUbDataND<half, 1, 128, 1, 128> kv_ub;
  TASSIGN(kv_ub, 4416);
  tl::ascend_pto::TileUbDataND<half, 8, 128, 8, 128> acc_o_half;
  TASSIGN(acc_o_half, 20096);
  auto vid = get_subblockid();
#if defined(__DAV_C220_CUBE__)
    tl::ascend_pto::copy_gm_to_l1_dynamic<half, half, 1, 1, 1, 16, 128, -1, -1, 128 * 64, 128, 1, 16, 128>(Q_handle + ((((((cid / seq_len) % batch) * seq_len) * 8192) + ((cid % seq_len) * 8192)) + ((((cid / seq_len) / batch) % 4) * 2048)), pto::Shape<1, 1, 1, 16, 128>(), pto::Stride<-1, -1, 128 * 64, 128, 1>(batch, seq_len), q_l1);

  for (int32_t i_i_outer = 0; i_i_outer < 16; ++i_i_outer) {

  for (int32_t i = 0; i < 2; ++i) {
        wait_flag_dev(0);
        set_flag(PIPE_M, PIPE_MTE2, EVENT_ID3);
        wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID3);
        tl::ascend_pto::copy_gm_to_l1<half, half, 1, 1, 1, 64, 128, 128 * 64 * 2184 * 2, 128 * 64 * 2184, 128 * 64, 128, 1, 64, 128>(workspace_1_handle + ((i * 17891328) + (cid * 8192)), kv_l1);
        set_flag(PIPE_MTE2, PIPE_M, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_M, EVENT_ID1);
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID4);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID4);
        pipe_barrier(PIPE_M);
        tl::ascend_pto::gemm_v0<half, float, 16, 64, 128, 16, 64, 128, false, true>(q_l1, kv_l1, acc_s_l0c, (bool)1);
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID2);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID2);
        pipe_barrier(PIPE_FIX);
        tl::ascend_pto::copy_l0c_to_gm<float, float, 1, 1, 1, 16, 64, 64 * 16 * 2184 * 2, 64 * 16 * 2184, 64 * 16, 64, 1, 16, 64>(workspace_2_handle + ((i * 2236416) + (cid * 1024)), acc_s_l0c);
        ffts_cross_core_sync(PIPE_FIX, 289);
      }

  for (int32_t i_1 = 0; i_1 < 2; ++i_1) {
        wait_flag_dev(2);
        set_flag(PIPE_M, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID1);
        tl::ascend_pto::copy_gm_to_l1<half, half, 1, 1, 1, 16, 64, 64 * 16 * 2184 * 2, 64 * 16 * 2184, 64 * 16, 64, 1, 16, 64>(workspace_3_handle + ((i_1 * 2236416) + (cid * 1024)), acc_s_l1);
        set_flag(PIPE_MTE2, PIPE_M, EVENT_ID7);
        wait_flag(PIPE_MTE2, PIPE_M, EVENT_ID7);
        pipe_barrier(PIPE_M);
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID2);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID2);
        tl::ascend_pto::gemm_v0<half, float, 16, 128, 64, 16, 128, 64, false, false>(acc_s_l1, kv_l1, acc_o_l0c, (bool)1);
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        pipe_barrier(PIPE_FIX);
        tl::ascend_pto::copy_l0c_to_gm<float, float, 1, 1, 1, 16, 128, 128 * 16 * 2184 * 2, 128 * 16 * 2184, 128 * 16, 128, 1, 16, 128>(workspace_4_handle + ((i_1 * 4472832) + (cid * 2048)), acc_o_l0c);
        ffts_cross_core_sync(PIPE_FIX, 801);
      }
    }
#endif
#if defined(__DAV_C220_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    TEXPANDS(acc_o, 0.000000e+00f);
    TEXPANDS(sumexp, 0.000000e+00f);
    TEXPANDS(m_i, -1.073742e+09f);

  for (int32_t i_i_outer_1 = 0; i_i_outer_1 < 16; ++i_i_outer_1) {

  for (int32_t i_2 = 0; i_2 < 2; ++i_2) {
        pipe_barrier(PIPE_MTE2);
        tl::ascend_pto::copy_gm_to_ub_dynamic<int, int, 1, 1, 1, 1, 64, -1, -1, 2048 * 4, 2048, 1, 1, 64>(Indices_handle + ((((((((cid / seq_len) % batch) * seq_len) * 8192) + ((cid % seq_len) * 8192)) + ((((cid / seq_len) / batch) % 4) * 2048)) + (i_i_outer_1 * 128)) + (i_2 * 64)), pto::Shape<1, 1, 1, 1, 64>(), pto::Stride<-1, -1, 2048 * 4, 2048, 1>(batch, seq_len), indices_ub_);

  for (int32_t bi_i = 0; bi_i < 32; ++bi_i) {
          pipe_barrier(PIPE_ALL);
          set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
          wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
          tl::ascend_pto::copy_gm_to_ub_dynamic<half, half, 1, 1, 1, 1, 128, -1, -1, 128 * 4, 128, 1, 1, 128>(KV_handle + ((((((cid / seq_len) % batch) * seq_len_kv) * 512) + (indices_ub_.GetValue(((vid * 32) + bi_i)) * 512)) + ((((cid / seq_len) / batch) % 4) * 128)), pto::Shape<1, 1, 1, 1, 128>(), pto::Stride<-1, -1, 128 * 4, 128, 1>(batch, seq_len_kv), kv_ub);
          set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
          wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
          pipe_barrier(PIPE_MTE3);
          tl::ascend_pto::copy_ub_to_gm<half, half, 1, 1, 1, 1, 128, 128 * 64 * 2184 * 2, 128 * 64 * 2184, 128 * 64, 128, 1, 1, 128, 1, 128>(workspace_1_handle + ((((i_2 * 17891328) + (cid * 8192)) + (vid * 4096)) + (bi_i * 128)), 4416, 0,2);
        }
        ffts_cross_core_sync(PIPE_MTE3, 33);
      }

  for (int32_t i_3 = 0; i_3 < 2; ++i_3) {
        pipe_barrier(PIPE_V);
        TEXPANDS(acc_s_ub, 0.000000e+00f);
        pipe_barrier(PIPE_V);
        TMOV(m_i_prev, m_i);
        wait_flag_dev(1);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
        tl::ascend_pto::copy_gm_to_ub<float, float, 1, 1, 1, 8, 64, 64 * 16 * 2184 * 2, 64 * 16 * 2184, 64 * 16, 64, 1, 8, 64>(workspace_2_handle + (((i_3 * 2236416) + (cid * 1024)) + (vid * 512)), acc_s_ub_);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        TADD(acc_s_ub, acc_s_ub, acc_s_ub_);
        pipe_barrier(PIPE_V);
        TMULS(acc_s_ub, acc_s_ub, 8.838835e-02f);
        pipe_barrier(PIPE_V);
        tl::ascend_pto::TileUbDataDN <float, 8, 1, 8, 1> m_i_DN;
        TASSIGN(m_i_DN, 4128);
        TROWMAX(m_i_DN, acc_s_ub, tmp_ub);
        pipe_barrier(PIPE_ALL);
        TRESHAPE(m_i, m_i_DN);
        pipe_barrier(PIPE_V);
        TMAX(m_i, m_i, m_i_prev);
        pipe_barrier(PIPE_V);
        TSUB(m_i_prev, m_i_prev, m_i);
        pipe_barrier(PIPE_V);
        TEXP(m_i_prev, m_i_prev);

  for (int32_t h_i = 0; h_i < 8; ++h_i) {
          pipe_barrier(PIPE_V);
          pipe_barrier(PIPE_ALL);
auto m_i_scalar= m_i.GetValue(h_i);
pipe_barrier(PIPE_ALL);
          tl::ascend_pto::TileUbDataND<float, 1, 64, 1, 64> acc_s_ub_temp;
          TASSIGN(acc_s_ub_temp, 4672 + (h_i * 64) * 4);
          pipe_barrier(PIPE_ALL);
          TADDS(acc_s_ub_temp, acc_s_ub_temp, -m_i_scalar);
        }
        pipe_barrier(PIPE_V);
        TEXP(acc_s_ub, acc_s_ub);
        pipe_barrier(PIPE_V);
        tl::ascend_pto::TileUbDataDN <float, 8, 1, 8, 1> sumexp_i_ub_DN;
        TASSIGN(sumexp_i_ub_DN, 14944);
        TROWSUM(sumexp_i_ub_DN, acc_s_ub, tmp_ub);
        pipe_barrier(PIPE_ALL);
        TRESHAPE(sumexp_i_ub, sumexp_i_ub_DN);
        TMUL(sumexp, sumexp, m_i_prev);
        pipe_barrier(PIPE_V);
        TADD(sumexp, sumexp, sumexp_i_ub);

  for (int32_t h_i_1 = 0; h_i_1 < 8; ++h_i_1) {
          pipe_barrier(PIPE_V);
          pipe_barrier(PIPE_ALL);
auto m_i_prev_scalar= m_i_prev.GetValue(h_i_1);
pipe_barrier(PIPE_ALL);
          tl::ascend_pto::TileUbDataND<float, 1, 128, 1, 128> acc_o_temp;
          TASSIGN(acc_o_temp, 0 + (h_i_1 * 128) * 4);
          pipe_barrier(PIPE_ALL);
          TMULS(acc_o_temp, acc_o_temp, m_i_prev_scalar);
        }
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
        TCVT(acc_s_half, acc_s_ub, pto::RoundMode::CAST_NONE);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        pipe_barrier(PIPE_MTE3);
        tl::ascend_pto::copy_ub_to_gm<half, half, 1, 1, 1, 8, 64, 64 * 16 * 2184 * 2, 64 * 16 * 2184, 64 * 16, 64, 1, 8, 64, 8, 64>(workspace_3_handle + (((i_3 * 2236416) + (cid * 1024)) + (vid * 512)), 14976, 0,2);
        ffts_cross_core_sync(PIPE_MTE3, 545);
      }

  for (int32_t i_4 = 0; i_4 < 2; ++i_4) {
        wait_flag_dev(3);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
        tl::ascend_pto::copy_gm_to_ub<float, float, 1, 1, 1, 8, 128, 128 * 16 * 2184 * 2, 128 * 16 * 2184, 128 * 16, 128, 1, 8, 128>(workspace_4_handle + (((i_4 * 4472832) + (cid * 2048)) + (vid * 1024)), acc_o_ub);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
        pipe_barrier(PIPE_V);
        TADD(acc_o, acc_o, acc_o_ub);
      }
    }

  for (int32_t h_i_2 = 0; h_i_2 < 8; ++h_i_2) {
      pipe_barrier(PIPE_V);
      pipe_barrier(PIPE_ALL);
auto sumexp_scalar= sumexp.GetValue(h_i_2);
pipe_barrier(PIPE_ALL);
      tl::ascend_pto::TileUbDataND<float, 1, 128, 1, 128> acc_o_temp;
      TASSIGN(acc_o_temp, 0 + (h_i_2 * 128) * 4);
      pipe_barrier(PIPE_ALL);
      TDIVS(acc_o_temp, acc_o_temp, sumexp_scalar);
    }
    pipe_barrier(PIPE_V);
    TCVT(acc_o_half, acc_o, pto::RoundMode::CAST_NONE);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    tl::ascend_pto::copy_ub_to_gm_dynamic<half, half, 1, 1, 1, 8, 128, -1, -1, 128 * 64, 128, 1, 8, 128, 8, 128>(Output_handle + (((((((cid / seq_len) % batch) * seq_len) * 8192) + ((cid % seq_len) * 8192)) + ((((cid / seq_len) / batch) % 4) * 2048)) + (vid * 1024)), pto::Shape<1, 1, 1, 8, 128>(), pto::Stride<-1, -1, 128 * 64, 128, 1>(batch, seq_len), 20096, 0,2);
#endif
}

extern "C" __global__ AICORE void launch_kernel(__gm__ uint8_t *Q_handle, __gm__ uint8_t *KV_handle, __gm__ uint8_t *Indices_handle, __gm__ uint8_t *Output_handle, __gm__ uint8_t *workspace_1_handle, __gm__ uint8_t *workspace_2_handle, __gm__ uint8_t *workspace_3_handle, __gm__ uint8_t *workspace_4_handle, int64_t batch, int64_t seq_len, int64_t seq_len_kv, uint64_t fftsAddr)
{
    main_kernel(reinterpret_cast<__gm__ half *>(Q_handle),
     reinterpret_cast<__gm__ half *>(KV_handle),
     reinterpret_cast<__gm__ int *>(Indices_handle),
     reinterpret_cast<__gm__ half *>(Output_handle),
     reinterpret_cast<__gm__ half *>(workspace_1_handle),
     reinterpret_cast<__gm__ float *>(workspace_2_handle),
     reinterpret_cast<__gm__ half *>(workspace_3_handle),
     reinterpret_cast<__gm__ float *>(workspace_4_handle), batch, seq_len, seq_len_kv,
     reinterpret_cast<uint64_t>(fftsAddr));
}

extern "C" void call(uint8_t *Q_handle, uint8_t *KV_handle, uint8_t *Indices_handle, uint8_t *Output_handle, uint8_t *workspace_1_handle, uint8_t *workspace_2_handle, uint8_t *workspace_3_handle, uint8_t *workspace_4_handle, int64_t batch, int64_t seq_len, int64_t seq_len_kv, void *stream)
{
    uint32_t fftsLen{0};
    uint64_t fftsAddr{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    launch_kernel<<<((seq_len * batch) * 4), nullptr, stream>>>(Q_handle, KV_handle, Indices_handle, Output_handle, workspace_1_handle, workspace_2_handle, workspace_3_handle, workspace_4_handle, batch, seq_len, seq_len_kv, fftsAddr);
}
