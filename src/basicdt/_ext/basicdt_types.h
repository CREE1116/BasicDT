#pragma once

#include <cstdint>
#include <vector>
#include <unordered_map>

// BasicDTTree — fitted axis-aligned tree (sparse heap layout, serializable).

struct BasicDTTree {
  int max_depth   = 0;
  int K           = 0;
  int D           = 0;
  int total_nodes = 0;
  std::vector<int>     split_feature;   // [total_nodes]: split feature index (-1 if leaf)
  std::vector<float>   split_threshold; // [total_nodes]: threshold value
  std::vector<float>   leaf_values;     // [total_nodes × K]
  std::vector<uint8_t> is_leaf;         // [total_nodes]
  std::vector<float>   split_gain;      // [total_nodes]

  int D_num = 0;
  std::vector<float> na_means;  // [D]: numeric μ_f impute; cat cols: NaN-category rank
  std::vector<std::unordered_map<int, float>> cat_ranks;  // [D_cat]: raw value → rank
};
