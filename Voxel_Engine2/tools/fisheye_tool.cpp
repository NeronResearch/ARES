#include <iostream>
#include <cmath>
#include <vector>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../third_party/stb_image.h"
#include "../third_party/stb_image_write.h"

int main() {
    const char* inputPath  = "data/1_0200.jpg";
    const char* outputPath = "out/fisheye_tool.png";

    int width, height, channels;
    unsigned char* img = stbi_load(inputPath, &width, &height, &channels, 3);
    if (!img) {
        std::cerr << "Failed to load image: " << inputPath << "\n";
        return 1;
    }

    float cx = width  / 2.0f;
    float cy = height / 2.0f;
    float xOffset = width * 0.02f;
    cx += xOffset;

    float corner_r = std::sqrt(cx * cx + cy * cy);
    float r_draw = corner_r * 0.7f;
    int thickness = 6;

    std::cout << "Image size: " << width << "x" << height
              << "  center=(" << cx << "," << cy << ")"
              << "  offset_x=" << xOffset
              << "  corner_r=" << corner_r
              << "  r_draw=" << r_draw << "\n";

    std::vector<unsigned char> out(width * height * 3);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * 3;
            float dx = x - cx;
            float dy = y - cy;
            float r = std::sqrt(dx * dx + dy * dy);

            // Copy original pixel
            out[idx + 0] = img[idx + 0];
            out[idx + 1] = img[idx + 1];
            out[idx + 2] = img[idx + 2];

            // Draw thin red ring
            if (std::fabs(r - r_draw) < thickness / 2.0f) {
                out[idx + 0] = 255;
                out[idx + 1] = 0;
                out[idx + 2] = 0;
            }
        }
    }

    stbi_write_png(outputPath, width, height, 3, out.data(), width * 3);
    stbi_image_free(img);

    std::cout << "Wrote " << outputPath << "\n";
    return 0;
}
