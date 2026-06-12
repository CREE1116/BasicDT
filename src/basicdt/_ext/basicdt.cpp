// basicdt.cpp — BasicDT: context-cached, subtraction-based axis-aligned decision tree booster
// with native missing-value and categorical handling.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <queue>
#include <random>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "basicdt_core.h"
#include "basicdt_types.h"

struct BasicDTCtx {
  int N = 0, D = 0, D_num = 0, D_cat = 0;
  std::vector<uint8_t> code;            // N·D uint8 bin codes
  std::vector<float> ax_min, ax_range;  // per-feature bin frame
  std::vector<float> Ximp;              // N·D transformed x̃ (numeric static,
                                        // cat columns rewritten per round)
  std::vector<float> col_mean;          // [D_num] numeric impute means μ_f

  // Categorical raw-value dictionaries (static across rounds).
  std::vector<std::unordered_map<int, int>> cat_id;  // raw → dense id
  std::vector<int> cat_card;       // per cat col: n_distinct + 1 (NaN slot)
  std::vector<int32_t> cat_dense;  // N·D_cat dense ids (NaN → card-1)
};

static inline int get_node_depth(int t) {
  int depth = 0;
  while (t > 0) {
    t = (t - 1) / 2;
    depth++;
  }
  return depth;
}

// Route a TRANSFORMED matrix x̃ (no NaN, cats already rank-encoded).
static void _gf_route(const BasicDTTree* tree, const float* X, int N,
                         float* out_pred) {
  int D = tree->D, K = tree->K;
  int T = tree->total_nodes;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < N; i++) {
    const float* GF_RESTRICT xi = X + (size_t)i * D;
    int t = 0;
    for (int dep = 0; dep < tree->max_depth; dep++) {
      if (tree->is_leaf[t]) break;
      int feat = tree->split_feature[t];
      if (feat < 0) break;
      float val = xi[feat];
      int next_t = (val < tree->split_threshold[t]) ? (2 * t + 1) : (2 * t + 2);
      if (next_t >= T) break;
      t = next_t;
    }
    const float* lv = tree->leaf_values.data() + (size_t)t * K;
    float* oi = out_pred + (size_t)i * K;
    for (int k = 0; k < K; k++) oi[k] = lv[k];
  }
}

extern "C" {

// Pre-bin all features once.
GF_API void* basicdt_ctx_create(const float* X, int N, int D, int D_num,
                                 const int* sub, int Ns) {
  auto* ctx = new BasicDTCtx();
  ctx->N = N;
  ctx->D = D;
  ctx->D_num = D_num;
  ctx->D_cat = D - D_num;
  ctx->ax_min.assign(D, 0.0f);
  ctx->ax_range.assign(D, 0.0f);
  ctx->col_mean.assign(D_num, 0.0f);
  ctx->Ximp.assign((size_t)N * D, 0.0f);
  ctx->code.assign((size_t)N * D, 0);

  // ── numeric: μ_f, min/max over the non-missing subsample ─────────────────
  std::vector<float> ax_max(D_num, -1e30f);
  std::vector<float> ax_lo(D_num, 1e30f);
  std::vector<double> sum(D_num, 0.0);
  std::vector<int> cnt(D_num, 0);
  for (int si = 0; si < Ns; si++) {
    const float* GF_RESTRICT xi = X + (size_t)sub[si] * D;
    for (int f = 0; f < D_num; f++) {
      float v = xi[f];
      if (std::isnan(v)) continue;
      if (v < ax_lo[f]) ax_lo[f] = v;
      if (v > ax_max[f]) ax_max[f] = v;
      sum[f] += v;
      cnt[f]++;
    }
  }
  std::vector<float> ax_scale(D_num, 0.0f);
  for (int f = 0; f < D_num; f++) {
    ctx->col_mean[f] = (float)(sum[f] / ((double)cnt[f] + EPS));
    if (cnt[f] == 0) {
      ax_lo[f] = 0.0f;
      continue;
    }
    ctx->ax_min[f] = ax_lo[f];
    float range = ax_max[f] - ax_lo[f];
    if (range > 1e-12f) {
      ctx->ax_range[f] = range;
      ax_scale[f] = (float)AX_BINS / (range + EPS);
    }
  }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < N; i++) {
    const float* GF_RESTRICT xi = X + (size_t)i * D;
    float* GF_RESTRICT ti = ctx->Ximp.data() + (size_t)i * D;
    uint8_t* GF_RESTRICT ci = ctx->code.data() + (size_t)i * D;
    for (int f = 0; f < D_num; f++) {
      float v = xi[f];
      if (std::isnan(v)) v = ctx->col_mean[f];
      ti[f] = v;
      if (ctx->ax_range[f] == 0.0f) continue;
      int b = (int)((v - ctx->ax_min[f]) * ax_scale[f]);
      if (b < 0) b = 0;
      if (b >= AX_BINS) b = AX_BINS - 1;
      ci[f] = (uint8_t)b;
    }
  }

  // ── categorical: value dictionary ────────────────────────────────────────
  if (ctx->D_cat > 0) {
    ctx->cat_id.resize(ctx->D_cat);
    ctx->cat_card.assign(ctx->D_cat, 0);
    ctx->cat_dense.assign((size_t)N * ctx->D_cat, 0);
    for (int fc = 0; fc < ctx->D_cat; fc++) {
      int f = D_num + fc;
      std::vector<int> vals;
      vals.reserve(64);
      for (int i = 0; i < N; i++) {
        float v = X[(size_t)i * D + f];
        if (std::isnan(v)) continue;
        vals.push_back((int)std::lrintf(v));
      }
      std::sort(vals.begin(), vals.end());
      vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
      auto& m = ctx->cat_id[fc];
      m.reserve(vals.size() * 2);
      for (int r = 0; r < (int)vals.size(); r++) m[vals[r]] = r;
      int nan_id = (int)vals.size();  // NaN is its own category
      ctx->cat_card[fc] = nan_id + 1;
      for (int i = 0; i < N; i++) {
        float v = X[(size_t)i * D + f];
        if (std::isnan(v)) {
          ctx->cat_dense[(size_t)i * ctx->D_cat + fc] = nan_id;
        } else {
          auto it = m.find((int)std::lrintf(v));
          ctx->cat_dense[(size_t)i * ctx->D_cat + fc] =
              (it != m.end()) ? it->second : nan_id;
        }
      }
    }
  }
  return static_cast<void*>(ctx);
}

GF_API void basicdt_ctx_free(void* h) { delete static_cast<BasicDTCtx*>(h); }

// ─── basicdt_build ─────────────────────────────────────────────────────────
GF_API void* basicdt_build(void* ctx_handle, const float* G,
                            const float* H, int K, const int* sub, int Ns,
                            int max_depth, float reg_lambda,
                            float* out_pred) {
  auto* ctx = static_cast<BasicDTCtx*>(ctx_handle);
  const int D = ctx->D, D_num = ctx->D_num, D_cat = ctx->D_cat, N = ctx->N;
  const float* GF_RESTRICT Xt = ctx->Ximp.data();
  const int STRIDE = 2 * K + 1;
  const size_t HSZ = (size_t)D * AX_BINS * STRIDE;

  const int internal_depth = std::min(max_depth, 22);
  const int max_leaves = 1 << max_depth;
  int max_nodes = (1 << (internal_depth + 1)) - 1;

  auto* tree = new BasicDTTree();
  tree->K = K;
  tree->D = D;
  tree->D_num = D_num;
  tree->max_depth = internal_depth;
  tree->total_nodes = max_nodes;
  tree->is_leaf.assign(max_nodes, 1);
  tree->split_feature.assign(max_nodes, -1);
  tree->split_threshold.assign(max_nodes, 0.0f);
  tree->leaf_values.assign((size_t)max_nodes * K, 0.0f);
  tree->split_gain.assign(max_nodes, 0.0f);
  tree->na_means.assign(D, 0.0f);
  std::copy(ctx->col_mean.begin(), ctx->col_mean.end(), tree->na_means.begin());

  // ── per-round categorical re-encoding ────────────────────────────────────
  if (D_cat > 0) {
    tree->cat_ranks.assign(D_cat, {});
    int kdom = 0;
    {
      float best_mass = -1.0f;
      for (int c = 0; c < K; c++) {
        float mcl = 0.0f;
        for (int si = 0; si < Ns; si++)
          mcl += std::abs(G[(size_t)sub[si] * K + c]);
        if (mcl > best_mass) {
          best_mass = mcl;
          kdom = c;
        }
      }
    }
    for (int fc = 0; fc < D_cat; fc++) {
      int f = D_num + fc;
      int card = ctx->cat_card[fc];
      if (card <= 1) {
        ctx->ax_range[f] = 0.0f;
        continue;
      }
      std::vector<double> Gs(card, 0.0), Hs(card, 0.0);
      for (int si = 0; si < Ns; si++) {
        int i = sub[si];
        int id = ctx->cat_dense[(size_t)i * D_cat + fc];
        Gs[id] += G[(size_t)i * K + kdom];
        Hs[id] += H[(size_t)i * K + kdom];
      }
      std::vector<float> score(card);
      for (int id = 0; id < card; id++)
        score[id] = (float)(Gs[id] / (Hs[id] + reg_lambda + EPS));
      std::vector<int> ord(card);
      std::iota(ord.begin(), ord.end(), 0);
      std::sort(ord.begin(), ord.end(), [&](int a, int b) {
        if (score[a] != score[b]) return score[a] < score[b];
        return a < b;
      });
      std::vector<float> rank_of(card);
      for (int r = 0; r < card; r++) rank_of[ord[r]] = (float)r;

      ctx->ax_min[f] = 0.0f;
      ctx->ax_range[f] = (float)(card - 1);
      float scale = (float)AX_BINS / ((float)(card - 1) + EPS);
      float* GF_RESTRICT Xw = ctx->Ximp.data();
      uint8_t* GF_RESTRICT cw = ctx->code.data();
      const int32_t* GF_RESTRICT cd = ctx->cat_dense.data();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (int i = 0; i < N; i++) {
        float r = rank_of[cd[(size_t)i * D_cat + fc]];
        Xw[(size_t)i * D + f] = r;
        int b = (int)(r * scale);
        if (b >= AX_BINS) b = AX_BINS - 1;
        cw[(size_t)i * D + f] = (uint8_t)b;
      }
      auto& rk = tree->cat_ranks[fc];
      rk.reserve(ctx->cat_id[fc].size() * 2);
      for (const auto& kv : ctx->cat_id[fc]) rk[kv.first] = rank_of[kv.second];
      tree->na_means[f] = rank_of[card - 1];
    }
  }
  const uint8_t* GF_RESTRICT code = ctx->code.data();

  // Optimized Histogram accumulation lambda (dynamic threads + cache blocking)
  auto accumulate_hist = [&](const int* rows, int nr, float* GF_RESTRICT hb,
                             float* node_P_out) {
    double P_acc = 0.0;
    int nthreads = 1;
#ifdef _OPENMP
    if (nr >= 2048) {
      nthreads = omp_get_max_threads();
    } else if (nr >= 512) {
      nthreads = std::min(4, omp_get_max_threads());
    } else if (nr >= 128) {
      nthreads = std::min(2, omp_get_max_threads());
    }
#endif

    if (nthreads > 1) {
      if (D >= nthreads && D >= 2) {
        // Block-wise Feature-parallelism (highly efficient for larger D, zero merge overhead)
        int block_size = std::max(1, D / nthreads);
#pragma omp parallel for schedule(static) num_threads(nthreads)
        for (int fg = 0; fg < D; fg += block_size) {
          int f_end = std::min(fg + block_size, D);
          for (int f = fg; f < f_end; f++) {
            float* GF_RESTRICT slot_f = hb + (size_t)f * AX_BINS * STRIDE;
            std::memset(slot_f, 0, AX_BINS * STRIDE * sizeof(float));
          }
          for (int si = 0; si < nr; si++) {
            int j = rows[si];
            const float* GF_RESTRICT gj = G + (size_t)j * K;
            const float* GF_RESTRICT hj = H + (size_t)j * K;
            for (int f = fg; f < f_end; f++) {
              int b = code[(size_t)j * D + f];
              float* GF_RESTRICT slot = hb + ((size_t)f * AX_BINS + b) * STRIDE;
              for (int c = 0; c < K; c++) {
                slot[c] += gj[c];
                slot[K + c] += hj[c];
              }
              slot[2 * K] += 1.0f;
            }
          }
        }
      } else {
        // Blocked Sample-parallelism (efficient for smaller D, single parallel region)
        std::memset(hb, 0, HSZ * sizeof(float));

        int B = 16;
        if (K > 5) B = 8;
        if (K > 16) B = 4;
        if (K > 32) B = 2;
        if (K > 64) B = 1;

        int B_max = B;
        int HSZ_block_max = B_max * AX_BINS * STRIDE;
        std::vector<float> local_hists_workspace((size_t)nthreads * HSZ_block_max);

#pragma omp parallel num_threads(nthreads)
        {
          int tid = omp_get_thread_num();

          for (int fb = 0; fb < D; fb += B) {
            int f_end = std::min(fb + B, D);
            int actual_B = f_end - fb;
            int HSZ_block = actual_B * AX_BINS * STRIDE;

            float* GF_RESTRICT lb = local_hists_workspace.data() + (size_t)tid * HSZ_block;
            std::memset(lb, 0, HSZ_block * sizeof(float));

#pragma omp barrier

#pragma omp for schedule(static) nowait
            for (int si = 0; si < nr; si++) {
              int j = rows[si];
              const uint8_t* GF_RESTRICT cj = code + (size_t)j * D;
              const float* GF_RESTRICT gj = G + (size_t)j * K;
              const float* GF_RESTRICT hj = H + (size_t)j * K;
              for (int f = fb; f < f_end; f++) {
                int b = cj[f];
                float* GF_RESTRICT slot = lb + ((size_t)(f - fb) * AX_BINS + b) * STRIDE;
                for (int c = 0; c < K; c++) {
                  slot[c] += gj[c];
                  slot[K + c] += hj[c];
                }
                slot[2 * K] += 1.0f;
              }
            }

#pragma omp barrier

#pragma omp for schedule(static)
            for (int i = 0; i < HSZ_block; i++) {
              float s = 0.0f;
              for (int t = 0; t < nthreads; t++) {
                s += local_hists_workspace[(size_t)t * HSZ_block + i];
              }
              hb[(size_t)fb * AX_BINS * STRIDE + i] = s;
            }

#pragma omp barrier
          }
        }
      }

      if (node_P_out) {
        double P_sum = 0.0;
#pragma omp parallel for reduction(+:P_sum) schedule(static) num_threads(nthreads)
        for (int si = 0; si < nr; si++) {
          int j = rows[si];
          const float* GF_RESTRICT gj = G + (size_t)j * K;
          const float* GF_RESTRICT hj = H + (size_t)j * K;
          for (int c = 0; c < K; c++) {
            P_sum += 0.5 * (double)gj[c] * gj[c] / ((double)hj[c] + reg_lambda + EPS);
          }
        }
        *node_P_out = (float)P_sum;
      }
    } else {
      // Fallback to single-threaded accumulation
      std::memset(hb, 0, HSZ * sizeof(float));
      for (int si = 0; si < nr; si++) {
        int j = rows[si];
        const uint8_t* GF_RESTRICT cj = code + (size_t)j * D;
        const float* GF_RESTRICT gj = G + (size_t)j * K;
        const float* GF_RESTRICT hj = H + (size_t)j * K;
        for (int f = 0; f < D; f++) {
          float* GF_RESTRICT slot = hb + ((size_t)f * AX_BINS + cj[f]) * STRIDE;
          for (int c = 0; c < K; c++) {
            slot[c] += gj[c];
            slot[K + c] += hj[c];
          }
          slot[2 * K] += 1.0f;
        }
        if (node_P_out) {
          for (int c = 0; c < K; c++) {
            P_acc += 0.5 * (double)gj[c] * gj[c] / ((double)hj[c] + reg_lambda + EPS);
          }
        }
      }
      if (node_P_out) *node_P_out = (float)P_acc;
    }
  };

  std::vector<std::vector<int>> node_samp(max_nodes);
  std::vector<std::vector<float>> node_hist(max_nodes);
  std::vector<std::vector<float>> hist_pool;

  auto get_hist = [&]() -> std::vector<float> {
    if (!hist_pool.empty()) {
      auto h = std::move(hist_pool.back());
      hist_pool.pop_back();
      return h;
    }
    return std::vector<float>(HSZ);
  };

  auto recycle_hist = [&](std::vector<float>& h) {
    if (h.size() == HSZ) {
      hist_pool.push_back(std::move(h));
    }
    h.clear();
  };

  std::vector<float> node_G((size_t)max_nodes * K, 0.0f);
  std::vector<float> node_H((size_t)max_nodes * K, 0.0f);
  std::vector<float> node_P(max_nodes, 0.0f);
  std::vector<char> node_has_tot(max_nodes, 0);

  std::vector<float> cand_gain(max_nodes, 0.0f);
  std::vector<float> cand_thr(max_nodes, 0.0f);
  std::vector<int> cand_axis(max_nodes, -1);
  std::vector<int> cand_bcode(max_nodes, 0);

  // ── Axis scan ────────────────────────────────────────────────────────────
  auto eval_axis = [&](int t) -> float {
    int ns = (int)node_samp[t].size();
    const float* GF_RESTRICT hb = node_hist[t].data();

    std::vector<float> Gt(K, 0.0f), Ht(K, 0.0f);
    for (int b = 0; b < AX_BINS; b++) {
      const float* GF_RESTRICT slot = hb + (size_t)b * STRIDE;
      for (int c = 0; c < K; c++) {
        Gt[c] += slot[c];
        Ht[c] += slot[K + c];
      }
    }
    for (int c = 0; c < K; c++) {
      node_G[(size_t)t * K + c] = Gt[c];
      node_H[(size_t)t * K + c] = Ht[c];
    }
    node_has_tot[t] = 1;
    float total_base = 0.0f;
    for (int c = 0; c < K; c++)
      total_base -= 0.5f * Gt[c] * Gt[c] / (Ht[c] + reg_lambda + EPS);

    float Ht_sum = 0.0f;
    for (int c = 0; c < K; c++) {
      Ht_sum += Ht[c];
    }

    float best_gain = 0.0f, best_thr = 0.0f;
    int best_axis = -1, best_axis_b = 0;

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      float l_gain = 0.0f, l_thr = 0.0f;
      int l_axis = -1, l_b = 0;
      std::vector<float> Gc(K), Hc(K);
#ifdef _OPENMP
#pragma omp for schedule(static) nowait
#endif
      for (int f = 0; f < D; f++) {
        if (ctx->ax_range[f] == 0.0f) continue;
        const float* GF_RESTRICT fbuf = hb + (size_t)f * AX_BINS * STRIDE;
        std::fill(Gc.begin(), Gc.end(), 0.0f);
        std::fill(Hc.begin(), Hc.end(), 0.0f);
        int n_left = 0;
        float Hcs = 0.0f;
        for (int b = 0; b < AX_BINS - 1; b++) {
          const float* GF_RESTRICT slot = fbuf + (size_t)b * STRIDE;
          n_left += (int)slot[2 * K];
          for (int c = 0; c < K; c++) {
            Gc[c] += slot[c];
            float hc = slot[K + c];
            Hc[c] += hc;
            Hcs += hc;
          }
          int n_right = ns - n_left;
          if (n_left < 10 || n_right < 10) continue;
          float Hrs = Ht_sum - Hcs;
          if (Hcs < MIN_CHILD_W || Hrs < MIN_CHILD_W) continue;
          float gain = total_base;
          for (int c = 0; c < K; c++) {
            float Gr = Gt[c] - Gc[c], Hr = Ht[c] - Hc[c];
            gain += 0.5f * (Gc[c] * Gc[c] / (Hc[c] + reg_lambda + EPS) +
                            Gr * Gr / (Hr + reg_lambda + EPS));
          }
          if (gain > l_gain || (gain == l_gain && l_axis >= 0 && f < l_axis)) {
            l_gain = gain;
            l_axis = f;
            l_b = b;
            l_thr =
                ctx->ax_min[f] + ((float)(b + 1) / AX_BINS) * ctx->ax_range[f];
          }
        }
      }
#ifdef _OPENMP
#pragma omp critical
#endif
      {
        if (l_axis >= 0 &&
            (l_gain > best_gain ||
             (l_gain == best_gain && (best_axis < 0 || l_axis < best_axis)))) {
          best_gain = l_gain;
          best_axis = l_axis;
          best_axis_b = l_b;
          best_thr = l_thr;
        }
      }
    }

    cand_gain[t] = best_gain;
    cand_thr[t] = best_thr;
    cand_axis[t] = best_axis;
    cand_bcode[t] = best_axis_b;
    return best_gain;
  };

  // ── Best-first growth loop ───────────────────────────────────────────────
  node_samp[0].assign(sub, sub + Ns);
  {
    node_hist[0] = get_hist();
    accumulate_hist(sub, Ns, node_hist[0].data(), &node_P[0]);
  }

  std::priority_queue<std::pair<float, int>> frontier;
  if (Ns >= 20) frontier.push({node_P[0], 0});

  int splits_left = max_leaves - 1;
  while (splits_left > 0 && !frontier.empty()) {
    int t_node = frontier.top().second;
    frontier.pop();

    if (node_hist[t_node].empty()) {
      int par_idx = (t_node - 1) / 2;
      int sib = (t_node % 2 == 1) ? t_node + 1 : t_node - 1;
      bool self_small = node_samp[t_node].size() <= node_samp[sib].size();
      int t_small = self_small ? t_node : sib;
      int t_large = self_small ? sib : t_node;

      node_hist[t_small] = get_hist();
      float* GF_RESTRICT hs = node_hist[t_small].data();
      accumulate_hist(node_samp[t_small].data(), (int)node_samp[t_small].size(),
                      hs, nullptr);
      float* GF_RESTRICT hp = node_hist[par_idx].data();
      for (size_t i = 0; i < HSZ; i++) hp[i] -= hs[i];
      node_hist[t_large] = std::move(node_hist[par_idx]);
    }

    float ag = eval_axis(t_node);
    if (ag <= 0.0f || cand_axis[t_node] < 0) {
      recycle_hist(node_hist[t_node]);
      continue;
    }

    const auto& samp = node_samp[t_node];
    int tl = 2 * t_node + 1, tr_node = 2 * t_node + 2;
    int depth_t = get_node_depth(t_node);

    tree->is_leaf[t_node] = 0;
    tree->split_feature[t_node] = cand_axis[t_node];
    tree->split_threshold[t_node] = cand_thr[t_node];
    tree->split_gain[t_node] = cand_gain[t_node];
    splits_left--;

    int ax = cand_axis[t_node];
    int bcode = cand_bcode[t_node];
    std::vector<int> left_sub, right_sub;
    std::vector<float> GL(K, 0.0f), HL(K, 0.0f);
    float PL = 0.0f;
    int ns = (int)samp.size();
#ifdef _OPENMP
    if (ns >= 8192) {
      int nthreads = omp_get_max_threads();
      std::vector<std::vector<int>> tL(nthreads);
      std::vector<std::vector<int>> tR(nthreads);
      std::vector<std::vector<float>> tGL(nthreads, std::vector<float>(K, 0.0f));
      std::vector<std::vector<float>> tHL(nthreads, std::vector<float>(K, 0.0f));
      std::vector<double> tPL(nthreads, 0.0);
      int actual_threads = nthreads;
#pragma omp parallel num_threads(nthreads)
      {
        int tid = omp_get_thread_num();
        if (tid == 0) actual_threads = omp_get_num_threads();
        auto& Lv = tL[tid];
        auto& Rv = tR[tid];
        int chunk_size = (ns + nthreads - 1) / nthreads;
        Lv.reserve(chunk_size);
        Rv.reserve(chunk_size);
        float* GF_RESTRICT gl = tGL[tid].data();
        float* GF_RESTRICT hl = tHL[tid].data();
        double pl = 0.0;
#pragma omp for schedule(static) nowait
        for (int si = 0; si < ns; si++) {
          int j = samp[si];
          bool go_left = (code[(size_t)j * D + ax] <= (uint8_t)bcode);
          if (go_left) {
            Lv.push_back(j);
            const float* GF_RESTRICT gj = G + (size_t)j * K;
            const float* GF_RESTRICT hj = H + (size_t)j * K;
            for (int c = 0; c < K; c++) {
              gl[c] += gj[c];
              hl[c] += hj[c];
              pl += 0.5 * (double)gj[c] * gj[c] /
                    ((double)hj[c] + reg_lambda + EPS);
            }
          } else {
            Rv.push_back(j);
          }
        }
        tPL[tid] = pl;
      }
      size_t nl = 0, nr = 0;
      for (int t = 0; t < actual_threads; t++) {
        nl += tL[t].size();
        nr += tR[t].size();
      }
      left_sub.reserve(nl);
      right_sub.reserve(nr);
      double PLd = 0.0;
      for (int t = 0; t < actual_threads; t++) {
        left_sub.insert(left_sub.end(), tL[t].begin(), tL[t].end());
        right_sub.insert(right_sub.end(), tR[t].begin(), tR[t].end());
        for (int c = 0; c < K; c++) {
          GL[c] += tGL[t][c];
          HL[c] += tHL[t][c];
        }
        PLd += tPL[t];
      }
      PL = (float)PLd;
    } else
#endif
    {
      left_sub.reserve(ns / 2 + 64);
      right_sub.reserve(ns / 2 + 64);
      for (int si = 0; si < ns; si++) {
        int j = samp[si];
        bool go_left = (code[(size_t)j * D + ax] <= (uint8_t)bcode);
        if (go_left) {
          left_sub.push_back(j);
          const float* GF_RESTRICT gj = G + (size_t)j * K;
          const float* GF_RESTRICT hj = H + (size_t)j * K;
          for (int c = 0; c < K; c++) {
            GL[c] += gj[c];
            HL[c] += hj[c];
            PL += 0.5f * gj[c] * gj[c] / (hj[c] + reg_lambda + EPS);
          }
        } else {
          right_sub.push_back(j);
        }
      }
    }
    node_P[tl] = PL;
    node_P[tr_node] = node_P[t_node] - PL;
    for (int c = 0; c < K; c++) {
      node_G[(size_t)tl * K + c] = GL[c];
      node_H[(size_t)tl * K + c] = HL[c];
      node_G[(size_t)tr_node * K + c] = node_G[(size_t)t_node * K + c] - GL[c];
      node_H[(size_t)tr_node * K + c] = node_H[(size_t)t_node * K + c] - HL[c];
    }
    node_has_tot[tl] = node_has_tot[tr_node] = 1;

    node_samp[tl] = std::move(left_sub);
    node_samp[tr_node] = std::move(right_sub);
    bool can_deepen = (depth_t + 1 < internal_depth) && (splits_left > 0);
    if (can_deepen) {
      for (int child : {tl, tr_node}) {
        int cns = (int)node_samp[child].size();
        if (cns < 20) continue;
        if (node_P[child] > 0.0f) frontier.push({node_P[child], child});
      }
    } else {
      recycle_hist(node_hist[t_node]);
    }
  }
  for (int t = 0; t < max_nodes; t++) {
    recycle_hist(node_hist[t]);
  }

  // ── Leaves smoothing ─────────────────────────────────────────────────────
  {
    std::vector<float> sm((size_t)max_nodes * K, 0.0f);
    std::vector<char> hasv(max_nodes, 0);
    for (int t = 0; t < max_nodes; t++) {
      if (!node_has_tot[t] && node_samp[t].empty()) continue;
      if (!node_has_tot[t]) {
        for (int j : node_samp[t]) {
          for (int c = 0; c < K; c++) {
            node_G[(size_t)t * K + c] += G[(size_t)j * K + c];
            node_H[(size_t)t * K + c] += H[(size_t)j * K + c];
          }
        }
      }
      int par_idx = (t - 1) / 2;
      bool use_parent = (t > 0) && hasv[par_idx];
      for (int c = 0; c < K; c++) {
        float Gs = node_G[(size_t)t * K + c], Hs = node_H[(size_t)t * K + c];
        float raw = -Gs / (Hs + reg_lambda + EPS);
        float v = use_parent
                      ? (Hs * raw + reg_lambda * sm[(size_t)par_idx * K + c]) /
                            (Hs + reg_lambda + EPS)
                      : raw;
        sm[(size_t)t * K + c] = v;
        if (tree->is_leaf[t]) tree->leaf_values[(size_t)t * K + c] = v;
      }
      hasv[t] = 1;
    }
  }

  if (out_pred) {
    std::memset(out_pred, 0, (size_t)N * K * sizeof(float));
    std::vector<uint8_t> in_sub(N, 0);
    for (int si = 0; si < Ns; si++) in_sub[sub[si]] = 1;
    for (int t = 0; t < max_nodes; t++) {
      if (!tree->is_leaf[t] || node_samp[t].empty()) continue;
      const float* lv = tree->leaf_values.data() + (size_t)t * K;
      for (int j : node_samp[t]) {
        float* oi = out_pred + (size_t)j * K;
        for (int k = 0; k < K; k++) oi[k] = lv[k];
      }
    }
    if (Ns < N) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (int i = 0; i < N; i++) {
        if (in_sub[i]) continue;
        const float* GF_RESTRICT xi = Xt + (size_t)i * D;
        int t = 0;
        for (int dep = 0; dep < tree->max_depth; dep++) {
          if (tree->is_leaf[t]) break;
          int feat = tree->split_feature[t];
          t = (xi[feat] < tree->split_threshold[t]) ? (2 * t + 1) : (2 * t + 2);
        }
        const float* lv = tree->leaf_values.data() + (size_t)t * K;
        float* oi = out_pred + (size_t)i * K;
        for (int k = 0; k < K; k++) oi[k] = lv[k];
      }
    }
  }
  return static_cast<void*>(tree);
}

// Predict on RAW X
GF_API void basicdt_predict(void* tree_handle, const float* X, int N, int K,
                             float* out_pred) {
  if (!tree_handle || !out_pred || N <= 0 || !X) return;
  const BasicDTTree* tree = static_cast<const BasicDTTree*>(tree_handle);
  std::memset(out_pred, 0, (size_t)N * K * sizeof(float));
  const int D = tree->D;
  if ((int)tree->na_means.size() != D) {
    _gf_route(tree, X, N, out_pred);
    return;
  }
  const int D_num = tree->D_num;
  std::vector<float> Xt((size_t)N * D);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < N; i++) {
    const float* GF_RESTRICT xi = X + (size_t)i * D;
    float* GF_RESTRICT ti = Xt.data() + (size_t)i * D;
    for (int f = 0; f < D; f++) {
      float v = xi[f];
      if (f < D_num) {
        ti[f] = std::isnan(v) ? tree->na_means[f] : v;
      } else {
        int fc = f - D_num;
        if (std::isnan(v) || fc >= (int)tree->cat_ranks.size()) {
          ti[f] = tree->na_means[f];
        } else {
          const auto& m = tree->cat_ranks[fc];
          auto it = m.find((int)std::lrintf(v));
          ti[f] = (it != m.end()) ? it->second : tree->na_means[f];
        }
      }
    }
  }
  _gf_route(tree, Xt.data(), N, out_pred);
}

// Compact routing node: stores the split feature directly
struct BasicDTCompactNode {
  int32_t feat = -1;
  float thr = 0.0f;
  int32_t left = -1, right = -1;  // compact ids; -1 → leaf
};

// Ensemble predict on RAW X
GF_API void basicdt_predict_ensemble(void* const* handles, int n_trees,
                                      const float* X, int N, int K, float lr,
                                      float* out_pred) {
  if (!handles || n_trees <= 0 || !out_pred) return;
  std::vector<const BasicDTTree*> trees;
  trees.reserve(n_trees > 0 ? n_trees : 0);
  for (int t = 0; t < n_trees; t++) {
    const BasicDTTree* tr = static_cast<const BasicDTTree*>(handles[t]);
    if (tr) trees.push_back(tr);
  }
  if (trees.empty()) return;
  const int n_live = (int)trees.size();
  const int D = trees[0]->D;
  const int D_num = trees[0]->D_num;
  const bool has_meta = (int)trees[0]->na_means.size() == D;
  const int D_cat = has_meta ? (D - D_num) : 0;

  // Shared numeric transform (na_means identical across one run's trees).
  std::vector<float> Xt((size_t)N * D);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < N; i++) {
    const float* GF_RESTRICT xi = X + (size_t)i * D;
    float* GF_RESTRICT ti = Xt.data() + (size_t)i * D;
    for (int f = 0; f < D; f++) {
      float v = xi[f];
      if (has_meta && f < D_num && std::isnan(v)) v = trees[0]->na_means[f];
      ti[f] = v;
    }
  }

  // categorical mapping
  std::vector<int32_t> cat_card(D_cat, 0);
  std::vector<int32_t> codes;
  std::vector<std::unordered_map<int, int32_t>> raw2code(D_cat);
  if (D_cat > 0) {
    for (int fc = 0; fc < D_cat; fc++) {
      if (fc >= (int)trees[0]->cat_ranks.size()) continue;
      const auto& m = trees[0]->cat_ranks[fc];
      raw2code[fc].reserve(m.size() * 2);
      int32_t next = 0;
      for (const auto& kv : m) raw2code[fc][kv.first] = next++;
      cat_card[fc] = next;
    }
    codes.assign((size_t)N * D_cat, 0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < N; i++) {
      const float* GF_RESTRICT xi = X + (size_t)i * D;
      int32_t* GF_RESTRICT ci = codes.data() + (size_t)i * D_cat;
      for (int fc = 0; fc < D_cat; fc++) {
        float v = xi[D_num + fc];
        int32_t c = cat_card[fc];
        if (!std::isnan(v)) {
          auto it = raw2code[fc].find((int)std::lrintf(v));
          if (it != raw2code[fc].end()) c = it->second;
        }
        ci[fc] = c;
      }
    }
  }

  static constexpr int TILE = 16;
  std::vector<std::vector<BasicDTCompactNode>> tile_nodes(TILE);
  std::vector<std::vector<float>> tile_leaves(TILE);
  std::vector<std::vector<float>> tile_rank(D_cat > 0 ? (size_t)TILE * D_cat : 0);
  std::vector<int> heap_of;
  heap_of.reserve(256);

  for (int tile_lo = 0; tile_lo < n_live; tile_lo += TILE) {
    const int tn = std::min(TILE, n_live - tile_lo);

    for (int tt = 0; tt < tn; tt++) {
      const BasicDTTree* tr = trees[tile_lo + tt];

      heap_of.clear();
      heap_of.push_back(0);
      for (size_t q = 0; q < heap_of.size(); q++) {
        int h = heap_of[q];
        if (!tr->is_leaf[h]) {
          heap_of.push_back(2 * h + 1);
          heap_of.push_back(2 * h + 2);
        }
      }
      const int M = (int)heap_of.size();
      auto& nodes = tile_nodes[tt];
      auto& leaf_vals = tile_leaves[tt];
      nodes.assign(M, BasicDTCompactNode{});
      leaf_vals.assign((size_t)M * K, 0.0f);
      int next = 1;
      for (int c = 0; c < M; c++) {
        int h = heap_of[c];
        if (tr->is_leaf[h]) {
          const float* lv = tr->leaf_values.data() + (size_t)h * K;
          std::copy(lv, lv + K, leaf_vals.begin() + (size_t)c * K);
        } else {
          nodes[c].feat = tr->split_feature[h];
          nodes[c].thr = tr->split_threshold[h];
          nodes[c].left = next;
          nodes[c].right = next + 1;
          next += 2;
        }
      }

      for (int fc = 0; fc < D_cat; fc++) {
        auto& tbl = tile_rank[(size_t)tt * D_cat + fc];
        tbl.assign((size_t)cat_card[fc] + 1, tr->na_means[D_num + fc]);
        if (fc < (int)tr->cat_ranks.size()) {
          for (const auto& kv : tr->cat_ranks[fc]) {
            auto it = raw2code[fc].find(kv.first);
            if (it != raw2code[fc].end()) tbl[it->second] = kv.second;
          }
        }
      }
    }

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      std::vector<float> row(D_cat > 0 ? D : 0);
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
      for (int i = 0; i < N; i++) {
        const float* GF_RESTRICT ti = Xt.data() + (size_t)i * D;
        float* GF_RESTRICT oi = out_pred + (size_t)i * K;
        const float* rp = ti;
        if (D_cat > 0) {
          std::memcpy(row.data(), ti, (size_t)D * sizeof(float));
          rp = row.data();
        }
        const int32_t* GF_RESTRICT ci =
            D_cat > 0 ? codes.data() + (size_t)i * D_cat : nullptr;
        for (int tt = 0; tt < tn; tt++) {
          if (D_cat > 0) {
            const auto* tbl0 = tile_rank.data() + (size_t)tt * D_cat;
            for (int fc = 0; fc < D_cat; fc++)
              row[D_num + fc] = tbl0[fc][ci[fc]];
          }
          const BasicDTCompactNode* GF_RESTRICT nd = tile_nodes[tt].data();
          int n = 0;
          while (nd[n].left >= 0) {
            float val = rp[nd[n].feat];
            n = (val < nd[n].thr) ? nd[n].left : nd[n].right;
          }
          const float* lv = tile_leaves[tt].data() + (size_t)n * K;
          for (int k = 0; k < K; k++) oi[k] += lr * lv[k];
        }
      }
    }
  }
}

GF_API void basicdt_tree_free(void* tree_handle) {
  delete static_cast<BasicDTTree*>(tree_handle);
}

// ─── tree meta (de)serialization ───────────────────────────────────────────
GF_API void basicdt_tree_meta_sizes(void* tree_handle, int* sizes) {
  const BasicDTTree* tree = static_cast<const BasicDTTree*>(tree_handle);
  sizes[0] = tree->D_num;
  sizes[1] = (int)tree->cat_ranks.size();
  int total = 0;
  for (const auto& m : tree->cat_ranks) total += (int)m.size();
  sizes[2] = total;
  sizes[3] = (int)tree->na_means.size();
}

GF_API void basicdt_tree_export_meta(void* tree_handle, float* na_means,
                                      int* cat_sizes, int* cat_keys,
                                      float* cat_vals) {
  const BasicDTTree* tree = static_cast<const BasicDTTree*>(tree_handle);
  for (size_t i = 0; i < tree->na_means.size(); i++)
    na_means[i] = tree->na_means[i];
  int off = 0;
  for (size_t fc = 0; fc < tree->cat_ranks.size(); fc++) {
    const auto& m = tree->cat_ranks[fc];
    cat_sizes[fc] = (int)m.size();
    for (const auto& kv : m) {
      cat_keys[off] = kv.first;
      cat_vals[off] = kv.second;
      off++;
    }
  }
}

GF_API void basicdt_tree_import_meta(void* tree_handle, int D_num,
                                      const float* na_means, int na_len,
                                      const int* cat_sizes, int D_cat,
                                      const int* cat_keys,
                                      const float* cat_vals) {
  BasicDTTree* tree = static_cast<BasicDTTree*>(tree_handle);
  tree->D_num = D_num;
  tree->na_means.assign(na_means, na_means + na_len);
  tree->cat_ranks.assign(D_cat, {});
  int off = 0;
  for (int fc = 0; fc < D_cat; fc++) {
    auto& m = tree->cat_ranks[fc];
    m.reserve((size_t)cat_sizes[fc] * 2);
    for (int e = 0; e < cat_sizes[fc]; e++) {
      m[cat_keys[off]] = cat_vals[off];
      off++;
    }
  }
}

// ─── tree structure (de)serialization ──────────────────────────────────────
GF_API int basicdt_get_K(void* handle) { return static_cast<BasicDTTree*>(handle)->K; }
GF_API int basicdt_get_max_depth(void* handle) {
  return static_cast<BasicDTTree*>(handle)->max_depth;
}
GF_API int basicdt_get_total_nodes(void* handle) {
  return static_cast<BasicDTTree*>(handle)->total_nodes;
}
GF_API int basicdt_get_D(void* handle) {
  return static_cast<BasicDTTree*>(handle)->D;
}

GF_API void basicdt_export(void* handle, int* split_feature,
                          float* split_threshold, float* leaf_values,
                          uint8_t* is_leaf) {
  const BasicDTTree* tree = static_cast<const BasicDTTree*>(handle);
  int n = tree->total_nodes, K = tree->K;
  for (int i = 0; i < n; ++i) split_feature[i] = tree->split_feature[i];
  for (int i = 0; i < n; ++i) split_threshold[i] = tree->split_threshold[i];
  for (size_t i = 0; i < (size_t)n * K; ++i) leaf_values[i] = tree->leaf_values[i];
  for (int i = 0; i < n; ++i) is_leaf[i] = tree->is_leaf[i];
}

GF_API void* basicdt_from_arrays(const int* split_feature,
                                const float* split_threshold,
                                const float* leaf_values,
                                const uint8_t* is_leaf, int total_nodes, int K,
                                int max_depth, int D) {
  BasicDTTree* tree = new BasicDTTree();
  tree->total_nodes = total_nodes;
  tree->K = K;
  tree->max_depth = max_depth;
  tree->D = D;
  tree->split_feature.assign(split_feature, split_feature + total_nodes);
  tree->split_threshold.assign(split_threshold, split_threshold + total_nodes);
  tree->leaf_values.assign(leaf_values, leaf_values + (size_t)total_nodes * K);
  tree->is_leaf.assign(is_leaf, is_leaf + total_nodes);
  return static_cast<void*>(tree);
}

GF_API void basicdt_update_gradients(const float* F, const float* oh, int N, int K, float* G, float* H) {
#ifdef _OPENMP
#pragma omp parallel num_threads(omp_get_max_threads())
#endif
  {
    std::vector<float> exp_buf_heap;
    float exp_buf_stack[128];
    float* exp_buf = exp_buf_stack;
    if (K > 128) {
      exp_buf_heap.resize(K);
      exp_buf = exp_buf_heap.data();
    }

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for (int i = 0; i < N; i++) {
      size_t offset = (size_t)i * K;

      // Find max F for numerical stability
      float fmax = F[offset];
      for (int c = 1; c < K; c++) {
        if (F[offset + c] > fmax) {
          fmax = F[offset + c];
        }
      }

      // Sum of exponentials (caching std::exp)
      double sum_exp = 0.0;
      for (int c = 0; c < K; c++) {
        float val = std::exp(F[offset + c] - fmax);
        exp_buf[c] = val;
        sum_exp += val;
      }

      double inv_sum = 1.0 / (sum_exp + 1e-20);

      // Compute P, G, H
      for (int c = 0; c < K; c++) {
        float p = (float)(exp_buf[c] * inv_sum);
        G[offset + c] = p - oh[offset + c];
        H[offset + c] = p * (1.0f - p);
      }
    }
  }
}

} // extern "C"
