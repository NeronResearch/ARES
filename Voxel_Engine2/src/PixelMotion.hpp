#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <omp.h>

class PixelMotion {
public:
    struct Settings {
        float motionThreshold = 10.0f;   // minimum diff to consider motion
        bool useAbsoluteDiff = true;     // absolute vs signed diff
    };

    PixelMotion(int width, int height, const Settings& settings)
        : width(width), height(height), settings(settings) {}

    std::vector<uint8_t> computeMotion(const std::vector<uint8_t>& img1,
                                       const std::vector<uint8_t>& img2) {
        if (img1.size() != img2.size()) {
            throw std::runtime_error("Images must be the same size.");
        }

        const size_t totalPixels = img1.size();
        std::vector<uint8_t> motionMap(totalPixels);

        const uint8_t* __restrict__ p1 = img1.data();
        const uint8_t* __restrict__ p2 = img2.data();
        uint8_t* __restrict__ pOut = motionMap.data();

        const float threshold = settings.motionThreshold;
        const bool useAbs = settings.useAbsoluteDiff;

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < totalPixels; ++i) {
            int a = p1[i];
            int b = p2[i];
            int diff = useAbs ? std::abs(b - a) : (b - a);
            pOut[i] = (diff >= threshold) ? 255 : 0;
        }

        return motionMap;
    }

private:
    int width;
    int height;
    Settings settings;
};
