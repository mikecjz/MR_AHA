/**
 * @file mr_metal_cpp_impl.cpp
 * @brief The one translation unit that instantiates metal-cpp.
 *
 * metal-cpp is header-only, but its Objective-C selector and class lookups
 * must be emitted exactly once per binary, in a translation unit that defines
 * the `*_PRIVATE_IMPLEMENTATION` macros before including the headers. Every
 * other Metal-backend source includes the headers without these macros.
 *
 * metal-cpp comes from the `third_party/VkFFT` submodule, so that VkFFT and
 * the library share one copy.
 *
 * @author Junzhou Chen (junzhou@chen.engineer)
 */

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"
#include "QuartzCore/QuartzCore.hpp"
