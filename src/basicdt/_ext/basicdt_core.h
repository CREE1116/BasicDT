#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <utility>
#include <vector>

#if defined(_WIN32) || defined(_WIN64)
#define GF_API __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#define GF_API __attribute__((visibility("default")))
#else
#define GF_API
#endif

#if defined(_MSC_VER)
#define GF_RESTRICT __restrict
#else
#define GF_RESTRICT __restrict__
#endif

static constexpr float EPS = 1e-8f;
static constexpr float MIN_CHILD_W = 0.1f;
static constexpr int AX_BINS = 256;      // global pre-binned feature codes
