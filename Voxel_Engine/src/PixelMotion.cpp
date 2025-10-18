#include "PixelMotion.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <cstdint>
#include <omp.h>

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
    if (UNLIKELY(img1.size() != img2.size())) {
        throw std::runtime_error("Images must be the same size.");
    }
    
    const size_t totalPixels = img1.size();
    std::vector<uint8_t> motionMap(totalPixels);
    
    const uint8_t* __restrict__ p1 = img1.data();
    const uint8_t* __restrict__ p2 = img2.data();
    uint8_t* __restrict__ pOut = motionMap.data();
    
    const float threshold = settings.motionThreshold;
    const float invRange = 255.0f / std::max(255.0f - threshold, 1.0f);
    const bool useAbs = settings.useAbsoluteDiff;
    
    // If contrast is 1.0, use ultra-fast integer path
    if (settings.contrastBoost == 1.0f) {
        // Pure integer math - BLAZING fast
        const int thresholdInt = static_cast<int>(threshold);
        const int scale = static_cast<int>(invRange * 256.0f); // Fixed-point scaling
        
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < totalPixels; ++i) {
            int a = p1[i];
            int b = p2[i];
            int diff = useAbs ? std::abs(b - a) : (b - a);
            
            // Branchless: if diff < threshold, result = 0, else scale it
            int aboveThreshold = (diff >= thresholdInt);
            int scaled = ((diff - thresholdInt) * scale) >> 8;
            int result = std::min(scaled * aboveThreshold, 255);
            
            pOut[i] = static_cast<uint8_t>(result);
        }
    } else {
        // Contrast path - precompute LUT
        const float invContrast = 1.0f / settings.contrastBoost;
        alignas(64) uint8_t lut[256];
        
        for (int i = 0; i < 256; ++i) {
            if (i < threshold) {
                lut[i] = 0;
            } else {
                float normalized = (i - threshold) * invRange / 255.0f;
                float contrasted = std::pow(normalized, invContrast);
                lut[i] = static_cast<uint8_t>(std::min(contrasted * 255.0f, 255.0f));
            }
        }
        
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < totalPixels; ++i) {
            int a = p1[i];
            int b = p2[i];
            uint8_t diff = useAbs ? 
                static_cast<uint8_t>(std::abs(b - a)) : 
                static_cast<uint8_t>(b - a);
            pOut[i] = lut[diff];
        }
    }
    
    return motionMap;
}