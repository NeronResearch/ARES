#include "PixelMotion.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <cstdint>
#include <omp.h>  // Enable OpenMP parallelism

// Cross-platform branch prediction hint
#ifdef __GNUC__
    #define LIKELY(x) __builtin_expect(!!(x), 1)
    #define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
    #define LIKELY(x) (x)
    #define UNLIKELY(x) (x)
#endif

PixelMotion::PixelMotion(int width, int height, const Settings& settings)
    : width(width), height(height), settings(settings) {}

std::vector<uint8_t> PixelMotion::computeMotion(const std::vector<uint8_t>& img1,
                                                const std::vector<uint8_t>& img2) {
    if (img1.size() != img2.size()) {
        throw std::runtime_error("Images must be the same size.");
    }

    const size_t totalPixels = img1.size();
    std::vector<uint8_t> motionMap(totalPixels);

    // Precompute constants outside the loop
    const float threshold = settings.motionThreshold;
    const float invRange = 1.0f / (255.0f - threshold);
    const bool useAbs = settings.useAbsoluteDiff;
    const float contrast = settings.contrastBoost;
    const bool normalize = settings.normalizeOutput;
    const bool applyContrast = (contrast != 1.0f);

    // Main loop: parallel + vectorized
    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < totalPixels; ++i) {
        // Load pixels as float
        float a = static_cast<float>(img1[i]);
        float b = static_cast<float>(img2[i]);

        // Compute difference (absolute or signed)
        float diff = useAbs ? std::abs(b - a) : (b - a);

        // Threshold check (branchless alternative would be slightly faster but less readable)
        if (LIKELY(diff < threshold)) {
            motionMap[i] = 0;
            continue;
        }

        // Scale to 0–255 range based on threshold
        float scaled = (diff - threshold) * invRange;

        // Optional contrast curve
        if (applyContrast) {
            scaled = std::pow(scaled, 1.0f / contrast);
        }

        // Clamp and convert to byte
        float result = std::clamp(scaled * 255.0f, 0.0f, 255.0f);

        // Optional normalization (rarely needed if scaled properly, but preserved for behavior)
        if (normalize) {
            result = std::clamp(result, 0.0f, 255.0f);
        }

        motionMap[i] = static_cast<uint8_t>(result);
    }

    return motionMap;
}
