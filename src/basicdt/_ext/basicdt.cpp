// basicdt.cpp — BasicDT: context-cached, subtraction-based axis-aligned
// decision tree booster with native missing-value and categorical handling.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <queue>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "basicdt_core.h"
#include "basicdt_types.h"

struct BasicDTCtx {
  const float* X = nullptr;
  int N = 0, D = 0, D_num = 0, D_cat = 0;
  std::vector<uint8_t> code;            // N·D uint8 bin codes
  std::vector<float> ax_min, ax_range;  // per-feature bin frame
  std::vector<float> col_mean;          // [D_num] numeric impute means μ_f

  // Categorical raw-value dictionaries (static across rounds).
  std::vector<std::vector<std::pair<int, int>>> cat_id;  // raw → dense id, sorted
  std::vector<int> cat_card;       // per cat col: n_distinct + 1 (NaN slot)
  std::vector<int32_t> cat_dense;  // N·D_cat dense ids (NaN → card-1)

  // Pre-allocated workspace for histogram accumulation
  std::vector<float> hist_workspace;

  // Cache categorical ranks from the previous round
  std::vector<std::vector<float>> prev_cat_ranks;
};

static void basicdt_tree_build_flat_cat_ranks(BasicDTTree* tree) {
  int D_cat = (int)tree->cat_ranks.size();
  tree->flat_cat_ranks.assign(D_cat, {});
  tree->cat_min_val.assign(D_cat, 0);
  tree->use_flat_lookup.assign(D_cat, 0);
  for (int fc = 0; fc < D_cat; fc++) {
    if (tree->cat_ranks[fc].empty()) continue;
    const auto& rk = tree->cat_ranks[fc];
    int min_k = rk.front().first;
    int max_k = rk.back().first;
    long long range = (long long)max_k - min_k;
    if (range >= 0 && range < 1048576) {
      tree->use_flat_lookup[fc] = 1;
      tree->cat_min_val[fc] = min_k;
      tree->flat_cat_ranks[fc].assign(range + 1,
                                      tree->na_means[tree->D_num + fc]);
      for (const auto& kv : rk) {
        tree->flat_cat_ranks[fc][kv.first - min_k] = kv.second;
      }
    }
  }
}

extern "C" {

// Pre-bin all features once.
GF_API void* basicdt_ctx_create(const float* X, int N, int D, int D_num,
                                const int* sub, int Ns) {
  auto* ctx = new BasicDTCtx();
  ctx->X = X;
  ctx->N = N;
  ctx->D = D;
  ctx->D_num = D_num;
  ctx->D_cat = D - D_num;
  ctx->ax_min.assign(D, 0.0f);
  ctx->ax_range.assign(D, 0.0f);
  ctx->col_mean.assign(D_num, 0.0f);
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
    uint8_t* GF_RESTRICT ci = ctx->code.data() + (size_t)i * D;
    for (int f = 0; f < D_num; f++) {
      float v = xi[f];
      if (std::isnan(v)) v = ctx->col_mean[f];
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
      vals.reserve(N);
      for (int i = 0; i < N; i++) {
        float v = X[(size_t)i * D + f];
        if (std::isnan(v)) continue;
        vals.push_back((int)std::lrintf(v));
      }
      std::sort(vals.begin(), vals.end());
      vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
      auto& m = ctx->cat_id[fc];
      m.reserve(vals.size());
      for (int r = 0; r < (int)vals.size(); r++) {
        m.push_back({vals[r], r});
      }
      int nan_id = (int)vals.size();  // NaN is its own category
      ctx->cat_card[fc] = nan_id + 1;
      for (int i = 0; i < N; i++) {
        float v = X[(size_t)i * D + f];
        if (std::isnan(v)) {
          ctx->cat_dense[(size_t)i * ctx->D_cat + fc] = nan_id;
        } else {
          int target = (int)std::lrintf(v);
          auto it = std::lower_bound(m.begin(), m.end(), std::make_pair(target, 0),
                                     [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                                       return a.first < b.first;
                                     });
          ctx->cat_dense[(size_t)i * ctx->D_cat + fc] =
              (it != m.end() && it->first == target) ? it->second : nan_id;
        }
      }
    }
  }
  return static_cast<void*>(ctx);
}

GF_API void basicdt_ctx_free(void* h) { delete static_cast<BasicDTCtx*>(h); }

// Set the OpenMP team size for subsequent builds/predicts. Histogram building
// is memory-bandwidth bound, so the throughput-optimal thread count is often
// well below the core count and is machine-specific (e.g. it saturates around
// 4 threads on Apple-silicon's shared memory bus, but a many-core server with
// more bandwidth wants more). Exposed as the classifier's n_jobs so callers
// tune per machine instead of the library hard-coding one platform's sweet
// spot. n <= 0 restores the default (all cores). No-op without OpenMP.
GF_API void basicdt_set_num_threads(int n) {
#ifdef _OPENMP
  if (n > 0) omp_set_num_threads(n);
  else omp_set_num_threads(omp_get_num_procs());
#else
  (void)n;
#endif
}

// ─── basicdt_build ─────────────────────────────────────────────────────────
GF_API void* basicdt_build(void* ctx_handle, const float* G, const float* H,
                           int K, const int* sub, int Ns, int max_depth,
                           float reg_lambda, float* out_pred) {
  auto* ctx = static_cast<BasicDTCtx*>(ctx_handle);
  const int D = ctx->D, D_num = ctx->D_num, D_cat = ctx->D_cat, N = ctx->N;
  // Multi-class histogram: each bin holds per-class G[KH], H[KH] and a count,
  // so split gain is scored across ALL classes (XGBoost-grade), not just the
  // dominant one. Layout per bin: [G_0..G_{KH-1}, H_0..H_{KH-1}, count].
  //
  // Binary fast path: for K==2 the softmax identities g_1 = -g_0 and
  // h_1 = p_1(1-p_1) = p_0 p_1 = h_0 are EXACT, so class 1's histogram is a
  // mirror of class 0's and need not be stored or accumulated. KH = 1 here
  // halves the histogram channels, the accumulate inner loop, and the merge —
  // recovering the per-class work XGBoost's hand-tuned binary path does. The
  // gain reconstruction (class 1 contributes G_0²/(H_0+λ), same as class 0)
  // lives in scan_feature; node_G/node_H/leaf_values stay full-K (they are
  // computed from the partition, not the histogram), so results are identical.
  const int KH = (K == 2) ? 1 : K;
  const int STRIDE = 2 * KH + 1;
  const int GOFF = 0, HOFF = KH, COFF = 2 * KH;  // channel offsets within a bin
  const size_t HSZ = (size_t)D * AX_BINS * STRIDE;

  int max_threads = 1;
#ifdef _OPENMP
  max_threads = omp_get_max_threads();
#endif
  size_t required_workspace_size = (size_t)max_threads * HSZ;
  if (ctx->hist_workspace.size() < required_workspace_size) {
    ctx->hist_workspace.resize(required_workspace_size);
  }

  const int internal_depth = std::min(max_depth, 22);
  const int max_leaves = 1 << max_depth;
  const int max_nodes = 2 * max_leaves - 1;

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

  auto* tree = new BasicDTTree();
  tree->K = K;
  tree->D = D;
  tree->D_num = D_num;
  tree->max_depth = internal_depth;
  tree->total_nodes = 1;

  tree->is_leaf.assign(max_nodes, 1);
  tree->split_feature.assign(max_nodes, -1);
  tree->split_threshold.assign(max_nodes, 0.0f);
  tree->leaf_values.assign((size_t)max_nodes * K, 0.0f);
  tree->split_gain.assign(max_nodes, 0.0f);
  tree->left_child.assign(max_nodes, -1);
  tree->right_child.assign(max_nodes, -1);
  tree->na_means.assign(D, 0.0f);
  std::copy(ctx->col_mean.begin(), ctx->col_mean.end(), tree->na_means.begin());

  // ── per-round categorical re-encoding ────────────────────────────────────
  if (D_cat > 0) {
    if (ctx->prev_cat_ranks.empty()) {
      ctx->prev_cat_ranks.assign(D_cat, {});
    }
    tree->cat_ranks.assign(D_cat, {});
    for (int fc = 0; fc < D_cat; fc++) {
      int f = D_num + fc;
      int card = ctx->cat_card[fc];
      if (card <= 1) {
        ctx->ax_range[f] = 0.0f;
        continue;
      }
      std::vector<float> Gs(card, 0.0f), Hs(card, 0.0f);
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

      bool ranks_changed = true;
      if (fc < (int)ctx->prev_cat_ranks.size()) {
        const auto& prev = ctx->prev_cat_ranks[fc];
        if (prev.size() == rank_of.size() &&
            std::equal(prev.begin(), prev.end(), rank_of.begin())) {
          ranks_changed = false;
        }
      }

      if (ranks_changed) {
        ctx->prev_cat_ranks[fc] = rank_of;
        uint8_t* GF_RESTRICT cw = ctx->code.data();
        const int32_t* GF_RESTRICT cd = ctx->cat_dense.data();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < N; i++) {
          float r = rank_of[cd[(size_t)i * D_cat + fc]];
          int b = (int)(r * scale);
          if (b >= AX_BINS) b = AX_BINS - 1;
          cw[(size_t)i * D + f] = (uint8_t)b;
        }
      }
      auto& rk = tree->cat_ranks[fc];
      rk.reserve(ctx->cat_id[fc].size());
      for (const auto& kv : ctx->cat_id[fc]) {
        rk.push_back({kv.first, rank_of[kv.second]});
      }
      std::sort(rk.begin(), rk.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });
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
    if (nr >= 16384) {
      nthreads = omp_get_max_threads();
    } else if (nr >= 4096) {
      nthreads = std::min(4, omp_get_max_threads());
    } else if (nr >= 1024) {
      nthreads = std::min(2, omp_get_max_threads());
    }
#endif

    if (nthreads > 1) {
      // Two parallel strategies trade different memory traffic:
      //   feature-parallel: each thread re-reads every row's G/H for its
      //     feature block → ~nr·KH·2·nthreads gradient reads, no merge.
      //   sample-parallel: one G/H pass + a thread-local histogram merge of
      //     HSZ·nthreads floats.
      // Feature-parallel only wins when the merge it avoids costs more than
      // the extra gradient re-reads, i.e. HSZ ≳ nr·KH·2. The old gate keyed
      // on D alone and mis-picked feature-parallel for low-K / large-nr nodes
      // (e.g. K=2, D=50, nr=60k re-read G/H 12× for no reason). Decide on the
      // actual traffic instead.
      const size_t merge_traffic = (size_t)D * AX_BINS * STRIDE;
      const size_t reread_traffic = (size_t)nr * KH * 2;
      if (D >= 8 && D >= nthreads && merge_traffic > reread_traffic) {
        // Block-wise Feature-parallelism (large D, zero merge overhead)
        int block_size = std::max(1, D / nthreads);
#pragma omp parallel for schedule(static) num_threads(nthreads)
        for (int fg = 0; fg < D; fg += block_size) {
          int f_end = std::min(fg + block_size, D);
          for (int f = fg; f < f_end; f++) {
            float* GF_RESTRICT slot_f = hb + (size_t)f * AX_BINS * STRIDE;
            std::memset(slot_f, 0, AX_BINS * STRIDE * sizeof(float));
          }
          for (int si = 0; si < nr; si++) {
#if defined(__GNUC__) || defined(__clang__)
            if (si + 1 < nr) {
              int next_j = rows[si + 1];
              __builtin_prefetch(code + (size_t)next_j * D, 0, 3);
              __builtin_prefetch(G + (size_t)next_j * K, 0, 3);
              __builtin_prefetch(H + (size_t)next_j * K, 0, 3);
            }
#endif
            int j = rows[si];
            const float* GF_RESTRICT gj = G + (size_t)j * K;
            const float* GF_RESTRICT hj = H + (size_t)j * K;
            for (int f = fg; f < f_end; f++) {
              int b = code[(size_t)j * D + f];
              float* GF_RESTRICT slot = hb + ((size_t)f * AX_BINS + b) * STRIDE;
              for (int c = 0; c < KH; c++) {
                slot[GOFF + c] += gj[c];
                slot[HOFF + c] += hj[c];
              }
              slot[COFF] += 1.0f;
            }
          }
        }
      } else {
        // Sample-parallelism using thread-local workspace
        size_t required_sz = (size_t)nthreads * HSZ;
        if (ctx->hist_workspace.size() < required_sz) {
          ctx->hist_workspace.resize(required_sz);
        }
        float* GF_RESTRICT workspace = ctx->hist_workspace.data();

#pragma omp parallel num_threads(nthreads)
        {
          int tid = omp_get_thread_num();
          float* GF_RESTRICT local_hb = workspace + (size_t)tid * HSZ;
          std::memset(local_hb, 0, HSZ * sizeof(float));

#pragma omp for schedule(static)
          for (int si = 0; si < nr; si++) {
#if defined(__GNUC__) || defined(__clang__)
            if (si + 1 < nr) {
              int next_j = rows[si + 1];
              __builtin_prefetch(code + (size_t)next_j * D, 0, 3);
              __builtin_prefetch(G + (size_t)next_j * K, 0, 3);
              __builtin_prefetch(H + (size_t)next_j * K, 0, 3);
            }
#endif
            int j = rows[si];
            const uint8_t* GF_RESTRICT cj = code + (size_t)j * D;
            const float* GF_RESTRICT gj = G + (size_t)j * K;
            const float* GF_RESTRICT hj = H + (size_t)j * K;
            for (int f = 0; f < D; f++) {
              int b = cj[f];
              float* GF_RESTRICT slot =
                  local_hb + ((size_t)f * AX_BINS + b) * STRIDE;
              for (int c = 0; c < KH; c++) {
                slot[GOFF + c] += gj[c];
                slot[HOFF + c] += hj[c];
              }
              slot[COFF] += 1.0f;
            }
          }

#pragma omp for schedule(static)
          for (size_t i = 0; i < HSZ; i++) {
            float s = 0.0f;
            for (int t = 0; t < nthreads; t++) {
              s += workspace[(size_t)t * HSZ + i];
            }
            hb[i] = s;
          }
        }
      }

      if (node_P_out) {
        double P_sum = 0.0;
#pragma omp parallel for reduction(+ : P_sum) schedule(static) num_threads(nthreads)
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
          for (int c = 0; c < KH; c++) {
            slot[GOFF + c] += gj[c];
            slot[HOFF + c] += hj[c];
          }
          slot[COFF] += 1.0f;
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

  // Pre-allocated workspace and structure for nodes
  std::vector<int> parent(max_nodes, -1);
  std::vector<int> sibling(max_nodes, -1);
  std::vector<int> node_depth(max_nodes, 0);

  // Partition-in-place sample tracking
  std::vector<int> sample_indices(sub, sub + Ns);
  std::vector<int> partition_scratch(Ns);
  std::vector<int> node_start(max_nodes, 0);
  std::vector<int> node_ns(max_nodes, 0);

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

  struct BestSplit {
    float gain = 0.0f;
    float thr = 0.0f;
    int axis = -1;
    int bcode = 0;
  };

  // ── Axis scan ────────────────────────────────────────────────────────────
  auto eval_axis = [&](int t) -> float {
    int ns = node_ns[t];
    const float* GF_RESTRICT hb = node_hist[t].data();

    // Per-class node totals and the no-split objective base, summed over classes.
    const float* GF_RESTRICT GtK = node_G.data() + (size_t)t * K;
    const float* GF_RESTRICT HtK = node_H.data() + (size_t)t * K;
    double total_base_d = 0.0, Ht_sum_d = 0.0;
    for (int c = 0; c < K; c++) {
      total_base_d += -0.5 * (double)GtK[c] * GtK[c] / ((double)HtK[c] + reg_lambda + EPS);
      Ht_sum_d += HtK[c];
    }
    const float total_base = (float)total_base_d;
    const float Ht_sum = (float)Ht_sum_d;
    BestSplit best{0.0f, 0.0f, -1, 0};

    // Scan one feature's histogram for its best split, scoring gain across all
    // K classes. Gc/Hc are caller-owned scratch of length K (per-thread). For
    // K==2 only KH=1 channel is stored; class 1 is its exact mirror, so each
    // accumulated class is weighted by `mirror` (2 = the class plus its twin):
    // hessian sums and the gain term both double, reproducing the full-K math.
    const float mirror = (K == 2) ? 2.0f : 1.0f;
    auto scan_feature = [&](int f, float* GF_RESTRICT Gc, float* GF_RESTRICT Hc,
                            BestSplit& lb) {
      if (ctx->ax_range[f] == 0.0f) return;
      const float* GF_RESTRICT fbuf = hb + (size_t)f * AX_BINS * STRIDE;
      int min_b = 0;
      while (min_b < AX_BINS - 1 && fbuf[(size_t)min_b * STRIDE + COFF] == 0.0f) {
        min_b++;
      }
      int max_b = AX_BINS - 1;
      while (max_b > min_b && fbuf[(size_t)max_b * STRIDE + COFF] == 0.0f) {
        max_b--;
      }
      std::memset(Gc, 0, KH * sizeof(float));
      std::memset(Hc, 0, KH * sizeof(float));
      int n_left = 0;
      for (int b = min_b; b < max_b; b++) {
        const float* GF_RESTRICT slot = fbuf + (size_t)b * STRIDE;
        n_left += (int)slot[COFF];
        float Hl_sum = 0.0f;
        for (int c = 0; c < KH; c++) {
          Gc[c] += slot[GOFF + c];
          Hc[c] += slot[HOFF + c];
          Hl_sum += Hc[c];
        }
        Hl_sum *= mirror;
        int n_right = ns - n_left;
        if (n_left < 10 || n_right < 10) continue;
        if (Hl_sum < MIN_CHILD_W || (Ht_sum - Hl_sum) < MIN_CHILD_W) continue;
        float gain = total_base;
        for (int c = 0; c < KH; c++) {
          float Gr = GtK[c] - Gc[c], Hr = HtK[c] - Hc[c];
          gain += 0.5f * mirror *
                  (Gc[c] * Gc[c] / (Hc[c] + reg_lambda + EPS) +
                   Gr * Gr / (Hr + reg_lambda + EPS));
        }
        if (gain > lb.gain ||
            (gain == lb.gain && lb.axis >= 0 && f < lb.axis)) {
          lb.gain = gain;
          lb.axis = f;
          lb.bcode = b;
          lb.thr = ctx->ax_min[f] + ((float)(b + 1) / AX_BINS) * ctx->ax_range[f];
        }
      }
    };

#ifdef _OPENMP
    // Only pay the parallel-region spawn/barrier cost when the scan is large
    // enough to amortize it. Best-first growth creates many tiny nodes whose
    // D×AX_BINS scan is dwarfed by thread setup; run those serially.
    int max_t = (ns >= 4096 && (size_t)D * AX_BINS >= 4096)
                    ? omp_get_max_threads()
                    : 1;
    if (max_t > 1) {
      std::vector<BestSplit> local_bests(max_t);
#pragma omp parallel num_threads(max_t)
      {
        int tid = omp_get_thread_num();
        BestSplit& l_best = local_bests[tid];
        std::vector<float> Gc(K), Hc(K);
#pragma omp for schedule(static)
        for (int f = 0; f < D; f++) {
          scan_feature(f, Gc.data(), Hc.data(), l_best);
        }
      }
      for (int tid = 0; tid < max_t; tid++) {
        const auto& l_best = local_bests[tid];
        if (l_best.axis >= 0 &&
            (l_best.gain > best.gain ||
             (l_best.gain == best.gain &&
              (best.axis < 0 || l_best.axis < best.axis)))) {
          best = l_best;
        }
      }
    } else {
      std::vector<float> Gc(K), Hc(K);
      for (int f = 0; f < D; f++) scan_feature(f, Gc.data(), Hc.data(), best);
    }
#else
    {
      std::vector<float> Gc(K), Hc(K);
      for (int f = 0; f < D; f++) scan_feature(f, Gc.data(), Hc.data(), best);
    }
#endif

    cand_gain[t] = best.gain;
    cand_thr[t] = best.thr;
    cand_axis[t] = best.axis;
    cand_bcode[t] = best.bcode;
    return best.gain;
  };

  // ── Best-first growth loop ───────────────────────────────────────────────
  node_start[0] = 0;
  node_ns[0] = Ns;
  {
#ifdef _OPENMP
    int max_t = omp_get_max_threads();
    std::vector<float> thread_G((size_t)max_t * K, 0.0f);
    std::vector<float> thread_H((size_t)max_t * K, 0.0f);
#pragma omp parallel num_threads(max_t)
    {
      int tid = omp_get_thread_num();
      float* local_G = thread_G.data() + (size_t)tid * K;
      float* local_H = thread_H.data() + (size_t)tid * K;
#pragma omp for schedule(static)
      for (int si = 0; si < Ns; si++) {
        int j = sub[si];
        const float* GF_RESTRICT gj = G + (size_t)j * K;
        const float* GF_RESTRICT hj = H + (size_t)j * K;
        for (int c = 0; c < K; c++) {
          local_G[c] += gj[c];
          local_H[c] += hj[c];
        }
      }
    }
    for (int t = 0; t < max_t; t++) {
      for (int c = 0; c < K; c++) {
        node_G[c] += thread_G[(size_t)t * K + c];
        node_H[c] += thread_H[(size_t)t * K + c];
      }
    }
#else
    for (int si = 0; si < Ns; si++) {
      int j = sub[si];
      const float* GF_RESTRICT gj = G + (size_t)j * K;
      const float* GF_RESTRICT hj = H + (size_t)j * K;
      for (int c = 0; c < K; c++) {
        node_G[c] += gj[c];
        node_H[c] += hj[c];
      }
    }
#endif
    node_has_tot[0] = 1;

    node_hist[0] = get_hist();
    accumulate_hist(sample_indices.data() + node_start[0], Ns, node_hist[0].data(), &node_P[0]);
  }

  std::priority_queue<std::pair<float, int>> frontier;
  if (Ns >= 20) frontier.push({node_P[0], 0});

  int splits_left = max_leaves - 1;
  while (splits_left > 0 && !frontier.empty()) {
    int t_node = frontier.top().second;
    frontier.pop();

    if (node_hist[t_node].empty()) {
      int par_idx = parent[t_node];
      int sib = sibling[t_node];
      bool self_small = node_ns[t_node] <= node_ns[sib];
      int t_small = self_small ? t_node : sib;
      int t_large = self_small ? sib : t_node;

      node_hist[t_small] = get_hist();
      float* GF_RESTRICT hs = node_hist[t_small].data();
      accumulate_hist(sample_indices.data() + node_start[t_small], node_ns[t_small], hs, nullptr);
      float* GF_RESTRICT hp = node_hist[par_idx].data();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (size_t i = 0; i < HSZ; i++) hp[i] -= hs[i];
      node_hist[t_large] = std::move(node_hist[par_idx]);
    }

    float ag = eval_axis(t_node);
    if (ag <= 0.0f || cand_axis[t_node] < 0) {
      recycle_hist(node_hist[t_node]);
      continue;
    }

    int depth_t = node_depth[t_node];
    int tl = tree->total_nodes;
    int tr_node = tree->total_nodes + 1;
    tree->total_nodes += 2;

    parent[tl] = t_node;
    parent[tr_node] = t_node;
    sibling[tl] = tr_node;
    sibling[tr_node] = tl;
    node_depth[tl] = depth_t + 1;
    node_depth[tr_node] = depth_t + 1;

    tree->left_child[t_node] = tl;
    tree->right_child[t_node] = tr_node;

    tree->is_leaf[t_node] = 0;
    tree->split_feature[t_node] = cand_axis[t_node];
    tree->split_threshold[t_node] = cand_thr[t_node];
    tree->split_gain[t_node] = cand_gain[t_node];
    splits_left--;

    int start = node_start[t_node];
    int ns = node_ns[t_node];
    int ax = cand_axis[t_node];
    int bcode = cand_bcode[t_node];

    std::vector<float> GL(K, 0.0f), HL(K, 0.0f);
    float PL = 0.0f;
    int total_left_count = 0;

#ifdef _OPENMP
    if (ns >= 8192) {
      int nthreads = omp_get_max_threads();
      std::vector<int> count_left(nthreads, 0);
      std::vector<int> count_right(nthreads, 0);
      std::vector<int> start_left(nthreads, 0);
      std::vector<int> start_right(nthreads, 0);

      float tGL_stack[16 * 128] = {0.0f};
      float tHL_stack[16 * 128] = {0.0f};
      std::vector<float> tGL_heap, tHL_heap;
      float* tGL = tGL_stack;
      float* tHL = tHL_stack;
      if (K > 128 || nthreads > 16) {
        tGL_heap.assign((size_t)nthreads * K, 0.0f);
        tHL_heap.assign((size_t)nthreads * K, 0.0f);
        tGL = tGL_heap.data();
        tHL = tHL_heap.data();
      }

      std::vector<double> tPL(nthreads, 0.0);
      int actual_threads = nthreads;

#pragma omp parallel num_threads(nthreads)
      {
        int tid = omp_get_thread_num();
        int num_t = omp_get_num_threads();

        int chunk_size = (ns + num_t - 1) / num_t;
        int chunk_start = tid * chunk_size;
        int chunk_end = std::min(ns, chunk_start + chunk_size);

        int l_count = 0;
        int r_count = 0;
        if (chunk_start < chunk_end) {
          for (int si = chunk_start; si < chunk_end; si++) {
#if defined(__GNUC__) || defined(__clang__)
            if (si + 1 < chunk_end) {
              int next_j = sample_indices[start + si + 1];
              __builtin_prefetch(code + (size_t)next_j * D, 0, 3);
            }
#endif
            int j = sample_indices[start + si];
            if (code[(size_t)j * D + ax] <= (uint8_t)bcode) {
              l_count++;
            } else {
              r_count++;
            }
          }
        }
        count_left[tid] = l_count;
        count_right[tid] = r_count;

#pragma omp barrier

#pragma omp single
        {
          int total_left = 0;
          int total_right = 0;
          for (int t = 0; t < num_t; t++) {
            start_left[t] = total_left;
            start_right[t] = total_right;
            total_left += count_left[t];
            total_right += count_right[t];
          }
          actual_threads = num_t;
        }

        int write_l = start_left[tid];
        int write_r = start_left[actual_threads - 1] + count_left[actual_threads - 1] + start_right[tid];

        float* GF_RESTRICT gl = tGL + (size_t)tid * K;
        float* GF_RESTRICT hl = tHL + (size_t)tid * K;
        double pl = 0.0;

        if (chunk_start < chunk_end) {
          for (int si = chunk_start; si < chunk_end; si++) {
#if defined(__GNUC__) || defined(__clang__)
            if (si + 1 < chunk_end) {
              int next_j = sample_indices[start + si + 1];
              __builtin_prefetch(code + (size_t)next_j * D, 0, 3);
              __builtin_prefetch(G + (size_t)next_j * K, 0, 3);
              __builtin_prefetch(H + (size_t)next_j * K, 0, 3);
            }
#endif
            int j = sample_indices[start + si];
            if (code[(size_t)j * D + ax] <= (uint8_t)bcode) {
              partition_scratch[write_l++] = j;
              const float* GF_RESTRICT gj = G + (size_t)j * K;
              const float* GF_RESTRICT hj = H + (size_t)j * K;
              for (int c = 0; c < K; c++) {
                gl[c] += gj[c];
                hl[c] += hj[c];
              }
              for (int c = 0; c < K; c++) {
                pl += 0.5 * (double)gj[c] * gj[c] / ((double)hj[c] + reg_lambda + EPS);
              }
            } else {
              partition_scratch[write_r++] = j;
            }
          }
        }
        tPL[tid] = pl;
      }

#pragma omp parallel for schedule(static) num_threads(actual_threads)
      for (int si = 0; si < ns; si++) {
        sample_indices[start + si] = partition_scratch[si];
      }

      double PLd = 0.0;
      for (int t = 0; t < actual_threads; t++) {
        const float* GF_RESTRICT gl = tGL + (size_t)t * K;
        const float* GF_RESTRICT hl = tHL + (size_t)t * K;
        for (int c = 0; c < K; c++) {
          GL[c] += gl[c];
          HL[c] += hl[c];
        }
        PLd += tPL[t];
      }
      PL = (float)PLd;
      total_left_count = start_left[actual_threads - 1] + count_left[actual_threads - 1];
    } else
#endif
    {
      int total_left = 0;
      for (int si = 0; si < ns; si++) {
        int j = sample_indices[start + si];
        if (code[(size_t)j * D + ax] <= (uint8_t)bcode) {
          total_left++;
        }
      }
      total_left_count = total_left;
      int write_l = 0;
      int write_r = total_left;
      for (int si = 0; si < ns; si++) {
#if defined(__GNUC__) || defined(__clang__)
        if (si + 1 < ns) {
          int next_j = sample_indices[start + si + 1];
          __builtin_prefetch(code + (size_t)next_j * D, 0, 3);
          __builtin_prefetch(G + (size_t)next_j * K, 0, 3);
          __builtin_prefetch(H + (size_t)next_j * K, 0, 3);
        }
#endif
        int j = sample_indices[start + si];
        if (code[(size_t)j * D + ax] <= (uint8_t)bcode) {
          partition_scratch[write_l++] = j;
          const float* GF_RESTRICT gj = G + (size_t)j * K;
          const float* GF_RESTRICT hj = H + (size_t)j * K;
          for (int c = 0; c < K; c++) {
            GL[c] += gj[c];
            HL[c] += hj[c];
          }
          for (int c = 0; c < K; c++) {
            PL += 0.5f * gj[c] * gj[c] / (hj[c] + reg_lambda + EPS);
          }
        } else {
          partition_scratch[write_r++] = j;
        }
      }
      std::memcpy(sample_indices.data() + start, partition_scratch.data(), ns * sizeof(int));
    }

    int total_left = total_left_count;
    int total_right = ns - total_left;

    node_start[tl] = start;
    node_ns[tl] = total_left;
    node_start[tr_node] = start + total_left;
    node_ns[tr_node] = total_right;

    node_P[tl] = PL;
    node_P[tr_node] = node_P[t_node] - PL;
    for (int c = 0; c < K; c++) {
      node_G[(size_t)tl * K + c] = GL[c];
      node_H[(size_t)tl * K + c] = HL[c];
      node_G[(size_t)tr_node * K + c] = node_G[(size_t)t_node * K + c] - GL[c];
      node_H[(size_t)tr_node * K + c] = node_H[(size_t)t_node * K + c] - HL[c];
    }
    node_has_tot[tl] = node_has_tot[tr_node] = 1;

    bool can_deepen = (depth_t + 1 < internal_depth) && (splits_left > 0);
    if (can_deepen) {
      for (int child : {tl, tr_node}) {
        int cns = node_ns[child];
        if (cns < 20) continue;
        if (node_P[child] > 0.0f) frontier.push({node_P[child], child});
      }
    } else {
      recycle_hist(node_hist[t_node]);
    }
  }

  for (int t = 0; t < tree->total_nodes; t++) {
    recycle_hist(node_hist[t]);
  }

  // ── Leaves smoothing ─────────────────────────────────────────────────────
  {
    int total_n = tree->total_nodes;
    std::vector<float> sm((size_t)total_n * K, 0.0f);
    std::vector<char> hasv(total_n, 0);
    basicdt_tree_build_flat_cat_ranks(tree);
    for (int t = 0; t < total_n; t++) {
      int par_idx = parent[t];
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
    for (int t = 0; t < tree->total_nodes; t++) {
      if (!tree->is_leaf[t] || node_ns[t] == 0) continue;
      const float* lv = tree->leaf_values.data() + (size_t)t * K;
      int t_start = node_start[t];
      int t_ns = node_ns[t];
      for (int si = 0; si < t_ns; si++) {
        int j = sample_indices[t_start + si];
        float* oi = out_pred + (size_t)j * K;
        for (int k = 0; k < K; k++) oi[k] = lv[k];
      }
    }
    if (Ns < N) {
      const bool has_meta = (int)tree->na_means.size() == D;
      const int D_cat = has_meta ? (D - D_num) : 0;
      const bool has_cached_flat = (int)tree->use_flat_lookup.size() == D_cat;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (int i = 0; i < N; i++) {
        if (in_sub[i]) continue;
        const float* GF_RESTRICT xi = ctx->X + (size_t)i * D;
        int t = 0;
        while (!tree->is_leaf[t]) {
          int feat = tree->split_feature[t];
          float val = xi[feat];

          if (has_meta) {
            if (feat < D_num) {
              if (std::isnan(val)) val = tree->na_means[feat];
            } else {
              int fc = feat - D_num;
              if (std::isnan(val) || fc >= (int)tree->cat_ranks.size()) {
                val = tree->na_means[feat];
              } else {
                if (has_cached_flat && tree->use_flat_lookup[fc]) {
                  int ival = (int)std::lrintf(val);
                  int offset_val = ival - tree->cat_min_val[fc];
                  if (offset_val >= 0 &&
                      offset_val < (int)tree->flat_cat_ranks[fc].size()) {
                    val = tree->flat_cat_ranks[fc][offset_val];
                  } else {
                    val = tree->na_means[feat];
                  }
                } else {
                  const auto& m = tree->cat_ranks[fc];
                  int target_key = (int)std::lrintf(val);
                  auto it = std::lower_bound(m.begin(), m.end(), std::make_pair(target_key, 0.0f),
                                             [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
                                               return a.first < b.first;
                                             });
                  val = (it != m.end() && it->first == target_key) ? it->second : tree->na_means[feat];
                }
              }
            }
          }
          t = (val < tree->split_threshold[t]) ? tree->left_child[t] : tree->right_child[t];
        }
        const float* lv = tree->leaf_values.data() + (size_t)t * K;
        float* oi = out_pred + (size_t)i * K;
        for (int k = 0; k < K; k++) oi[k] = lv[k];
      }
    }
  }

  // Shrink tree vectors to actual built size
  int total_n = tree->total_nodes;
  tree->is_leaf.resize(total_n);
  tree->split_feature.resize(total_n);
  tree->split_threshold.resize(total_n);
  tree->leaf_values.resize((size_t)total_n * K);
  tree->split_gain.resize(total_n);
  tree->left_child.resize(total_n);
  tree->right_child.resize(total_n);

  return static_cast<void*>(tree);
}

GF_API void basicdt_predict(void* tree_handle, const float* X, int N, int K,
                            float* out_pred) {
  if (!tree_handle || !out_pred || N <= 0 || !X) return;
  const BasicDTTree* tree = static_cast<const BasicDTTree*>(tree_handle);
  std::memset(out_pred, 0, (size_t)N * K * sizeof(float));
  const int D = tree->D;
  const int D_num = tree->D_num;
  const int T = tree->total_nodes;
  const bool has_meta = (int)tree->na_means.size() == D;
  const int D_cat = has_meta ? (D - D_num) : 0;
  const bool has_cached_flat = (int)tree->use_flat_lookup.size() == D_cat;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < N; i++) {
#if defined(__GNUC__) || defined(__clang__)
    if (i + 1 < N) {
      __builtin_prefetch(X + (size_t)(i + 1) * D, 0, 3);
    }
#endif
    const float* GF_RESTRICT xi = X + (size_t)i * D;
    int t = 0;
    while (!tree->is_leaf[t]) {
      int feat = tree->split_feature[t];
      float val = xi[feat];

      if (has_meta) {
        if (feat < D_num) {
          if (std::isnan(val)) val = tree->na_means[feat];
        } else {
          int fc = feat - D_num;
          if (std::isnan(val) || fc >= (int)tree->cat_ranks.size()) {
            val = tree->na_means[feat];
          } else {
            if (has_cached_flat && tree->use_flat_lookup[fc]) {
              int ival = (int)std::lrintf(val);
              int offset_val = ival - tree->cat_min_val[fc];
              if (offset_val >= 0 &&
                  offset_val < (int)tree->flat_cat_ranks[fc].size()) {
                val = tree->flat_cat_ranks[fc][offset_val];
              } else {
                val = tree->na_means[feat];
              }
            } else {
              const auto& m = tree->cat_ranks[fc];
              int target_key = (int)std::lrintf(val);
              auto it = std::lower_bound(m.begin(), m.end(),
                                         std::make_pair(target_key, 0.0f),
                                         [](const std::pair<int, float>& a,
                                            const std::pair<int, float>& b) {
                                           return a.first < b.first;
                                         });
              val = (it != m.end() && it->first == target_key)
                        ? it->second
                        : tree->na_means[feat];
            }
          }
        }
      }

      t = (val < tree->split_threshold[t]) ? tree->left_child[t]
                                           : tree->right_child[t];
    }
    const float* lv = tree->leaf_values.data() + (size_t)t * K;
    float* oi = out_pred + (size_t)i * K;
    for (int k = 0; k < K; k++) oi[k] = lv[k];
  }
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

  // categorical mapping
  std::vector<int32_t> cat_card(D_cat, 0);
  std::vector<int32_t> codes;
  std::vector<std::vector<std::pair<int, int32_t>>> raw2code(D_cat);
  if (D_cat > 0) {
    for (int fc = 0; fc < D_cat; fc++) {
      if (fc >= (int)trees[0]->cat_ranks.size()) continue;
      const auto& m = trees[0]->cat_ranks[fc];
      raw2code[fc].reserve(m.size());
      int32_t next = 0;
      for (const auto& kv : m) {
        raw2code[fc].push_back({kv.first, next++});
      }
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
          int target = (int)std::lrintf(v);
          const auto& m = raw2code[fc];
          auto it = std::lower_bound(m.begin(), m.end(), std::make_pair(target, 0),
                                     [](const std::pair<int, int32_t>& a, const std::pair<int, int32_t>& b) {
                                       return a.first < b.first;
                                     });
          if (it != m.end() && it->first == target) c = it->second;
        }
        ci[fc] = c;
      }
    }
  }

  // Pre-build compact representations of ALL trees (flat layout)
  std::vector<BasicDTCompactNode> flat_nodes;
  std::vector<float> flat_leaves;
  std::vector<int> node_offset(n_live, 0);
  std::vector<int> leaf_offset(n_live, 0);

  // contiguous flat layout for all ranks
  std::vector<float> flat_ranks;
  std::vector<size_t> rank_offsets(D_cat > 0 ? (size_t)n_live * D_cat : 0, 0);

  // estimate size to reserve
  flat_nodes.reserve(n_live * (2 * (1 << trees[0]->max_depth) - 1));
  flat_leaves.reserve(n_live * (2 * (1 << trees[0]->max_depth) - 1) * K);

  if (D_cat > 0) {
    size_t total_ranks = 0;
    for (int t = 0; t < n_live; t++) {
      for (int fc = 0; fc < D_cat; fc++) {
        total_ranks += (size_t)cat_card[fc] + 1;
      }
    }
    flat_ranks.resize(total_ranks);
  }

  std::vector<int> heap_of;
  heap_of.reserve(256);

  size_t current_rank_offset = 0;

  for (int t = 0; t < n_live; t++) {
    const BasicDTTree* tr = trees[t];

    heap_of.clear();
    heap_of.push_back(0);
    for (size_t q = 0; q < heap_of.size(); q++) {
      int h = heap_of[q];
      if (!tr->is_leaf[h]) {
        heap_of.push_back(tr->left_child[h]);
        heap_of.push_back(tr->right_child[h]);
      }
    }
    const int M = (int)heap_of.size();

    node_offset[t] = (int)flat_nodes.size();
    leaf_offset[t] = (int)flat_leaves.size();

    flat_nodes.resize(flat_nodes.size() + M);
    flat_leaves.resize(flat_leaves.size() + (size_t)M * K);

    BasicDTCompactNode* nodes = flat_nodes.data() + node_offset[t];
    float* leaf_vals = flat_leaves.data() + leaf_offset[t];

    int next = 1;
    for (int c = 0; c < M; c++) {
      int h = heap_of[c];
      if (tr->is_leaf[h]) {
        const float* lv = tr->leaf_values.data() + (size_t)h * K;
        std::copy(lv, lv + K, leaf_vals + (size_t)c * K);
      } else {
        nodes[c].feat = tr->split_feature[h];
        nodes[c].thr = tr->split_threshold[h];
        nodes[c].left = next;
        nodes[c].right = next + 1;
        next += 2;
      }
    }

    for (int fc = 0; fc < D_cat; fc++) {
      size_t card_sz = (size_t)cat_card[fc] + 1;
      rank_offsets[(size_t)t * D_cat + fc] = current_rank_offset;
      float* tbl = flat_ranks.data() + current_rank_offset;
      std::fill(tbl, tbl + card_sz, tr->na_means[D_num + fc]);

      if (fc < (int)tr->cat_ranks.size()) {
        for (const auto& kv : tr->cat_ranks[fc]) {
          int target = kv.first;
          const auto& m = raw2code[fc];
          auto it = std::lower_bound(m.begin(), m.end(), std::make_pair(target, 0),
                                     [](const std::pair<int, int32_t>& a, const std::pair<int, int32_t>& b) {
                                       return a.first < b.first;
                                     });
          if (it != m.end() && it->first == target) {
            tbl[it->second] = kv.second;
          }
        }
      }
      current_rank_offset += card_sz;
    }
  }

  // NOTE: tree-tiling the route loop (16-tree tiles cache-resident across
  // all rows, as in the sibling oblique engine) was tried here and measured
  // NEUTRAL for K=2 and ~12% SLOWER for K=5. BasicDT's axis nodes are tiny
  // (~16 B vs the oblique engine's ~128 B SparseVec nodes), so the whole
  // ensemble streams cheaply and the prefetcher hides it; tiling only adds
  // K-proportional out_pred re-streaming (one accumulate pass per tile).
  // The single all-trees-per-row sweep below is the right layout here.
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    std::vector<float> row(D_cat > 0 ? D : 0);
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for (int i = 0; i < N; i++) {
#if defined(__GNUC__) || defined(__clang__)
      if (i + 1 < N) {
        __builtin_prefetch(X + (size_t)(i + 1) * D, 0, 3);
        if (D_cat > 0) {
          __builtin_prefetch(codes.data() + (size_t)(i + 1) * D_cat, 0, 3);
        }
      }
#endif
      const float* GF_RESTRICT xi = X + (size_t)i * D;
      float* GF_RESTRICT oi = out_pred + (size_t)i * K;
      const float* rp = xi;
      if (D_cat > 0) {
        std::memcpy(row.data(), xi, (size_t)D_num * sizeof(float));
        rp = row.data();
      }
      const int32_t* GF_RESTRICT ci =
          D_cat > 0 ? codes.data() + (size_t)i * D_cat : nullptr;
      for (int t = 0; t < n_live; t++) {
#if defined(__GNUC__) || defined(__clang__)
        if (t + 1 < n_live) {
          __builtin_prefetch(flat_nodes.data() + node_offset[t + 1], 0, 3);
          __builtin_prefetch(flat_leaves.data() + leaf_offset[t + 1], 0, 3);
        }
#endif
        if (D_cat > 0) {
          const size_t base_offset = (size_t)t * D_cat;
          for (int fc = 0; fc < D_cat; fc++) {
            row[D_num + fc] = flat_ranks[rank_offsets[base_offset + fc] + ci[fc]];
          }
        }
        const BasicDTCompactNode* GF_RESTRICT nd =
            flat_nodes.data() + node_offset[t];
        int n = 0;
        while (nd[n].left >= 0) {
          float val = rp[nd[n].feat];
          if (nd[n].feat < D_num && std::isnan(val)) {
            val = trees[0]->na_means[nd[n].feat];
          }
          n = (val < nd[n].thr) ? nd[n].left : nd[n].right;
        }
        const float* lv = flat_leaves.data() + leaf_offset[t] + (size_t)n * K;
        for (int k = 0; k < K; k++) oi[k] += lr * lv[k];
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
    m.reserve(cat_sizes[fc]);
    for (int e = 0; e < cat_sizes[fc]; e++) {
      m.push_back({cat_keys[off], cat_vals[off]});
      off++;
    }
    std::sort(m.begin(), m.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
  }
  basicdt_tree_build_flat_cat_ranks(tree);
}

// ─── tree structure (de)serialization ──────────────────────────────────────
GF_API int basicdt_get_K(void* handle) {
  return static_cast<BasicDTTree*>(handle)->K;
}
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
                           uint8_t* is_leaf, int* left_child,
                           int* right_child) {
  const BasicDTTree* tree = static_cast<const BasicDTTree*>(handle);
  int n = tree->total_nodes, K = tree->K;
  for (int i = 0; i < n; ++i) split_feature[i] = tree->split_feature[i];
  for (int i = 0; i < n; ++i) split_threshold[i] = tree->split_threshold[i];
  for (size_t i = 0; i < (size_t)n * K; ++i)
    leaf_values[i] = tree->leaf_values[i];
  for (int i = 0; i < n; ++i) is_leaf[i] = tree->is_leaf[i];
  for (int i = 0; i < n; ++i) left_child[i] = tree->left_child[i];
  for (int i = 0; i < n; ++i) right_child[i] = tree->right_child[i];
}

GF_API void* basicdt_from_arrays(const int* split_feature,
                                 const float* split_threshold,
                                 const float* leaf_values,
                                 const uint8_t* is_leaf, const int* left_child,
                                 const int* right_child, int total_nodes, int K,
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
  tree->left_child.assign(left_child, left_child + total_nodes);
  tree->right_child.assign(right_child, right_child + total_nodes);
  return static_cast<void*>(tree);
}

GF_API void basicdt_update_gradients(const float* F, const float* oh, int N,
                                     int K, float* G, float* H) {
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    float exp_buf_stack[128];
    std::vector<float> exp_buf_tl;
    float* exp_buf = exp_buf_stack;
    if (K > 128) {
      exp_buf_tl.resize(K);
      exp_buf = exp_buf_tl.data();
    }

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for (int i = 0; i < N; i++) {
#if defined(__GNUC__) || defined(__clang__)
      if (i + 1 < N) {
        __builtin_prefetch(F + (size_t)(i + 1) * K, 0, 3);
        __builtin_prefetch(oh + (size_t)(i + 1) * K, 0, 3);
      }
#endif
      size_t offset = (size_t)i * K;

      // Find max F for numerical stability
      float fmax = F[offset];
      for (int c = 1; c < K; c++) {
        if (F[offset + c] > fmax) {
          fmax = F[offset + c];
        }
      }

      // Sum of exponentials (caching std::expf)
      float sum_exp = 0.0f;
      for (int c = 0; c < K; c++) {
        float val = std::expf(F[offset + c] - fmax);
        exp_buf[c] = val;
        sum_exp += val;
      }

      float inv_sum = 1.0f / (sum_exp + 1e-20f);

      // Compute P, G, H
      for (int c = 0; c < K; c++) {
        float p = (float)(exp_buf[c] * inv_sum);
        G[offset + c] = p - oh[offset + c];
        H[offset + c] = p * (1.0f - p);
      }
    }
  }
}

}  // extern "C"
