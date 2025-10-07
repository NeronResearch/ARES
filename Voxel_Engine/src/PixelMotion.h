#pragma once
#include <vector>
#include <cstdint>

class PixelMotion {
public:
    struct Settings {
        float motionThreshold = 10.0f;    // minimum diff to consider motion
        bool useAbsoluteDiff = true;      // absolute vs signed diff
        bool normalizeOutput = true;      // scale motion to 0–255
        float contrastBoost = 1.0f;       // >1 increases white intensity
    };

    PixelMotion(int width, int height, const Settings& settings);
    std::vector<uint8_t> computeMotion(const std::vector<uint8_t>& img1,
                                       const std::vector<uint8_t>& img2);

private:
    int width;
    int height;
    Settings settings;
};
