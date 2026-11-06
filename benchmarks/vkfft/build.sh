#!/bin/bash
# Build the VkFFT-Metal feasibility benchmark (macOS / Apple silicon only).
# VkFFT and metal-cpp are header-only and come from the third_party/VkFFT
# submodule (pinned to v1.3.4): `git submodule update --init third_party/VkFFT`.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
vk="$here/../../third_party/VkFFT"
out="${1:-$here/vkfft_bench}"
clang++ -std=c++17 -O3 -DVKFFT_BACKEND=5 \
    -I "$vk/vkFFT" -I "$vk/metal-cpp" \
    -framework Metal -framework Foundation -framework QuartzCore \
    "$here/vkfft_bench.cpp" -o "$out"
echo "built $out"
