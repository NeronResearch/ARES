#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <vector>
#include <queue>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

using namespace std;


std::vector<uint8_t> loadImage(const string& filename, int& width, int& height, int& channels) {
    unsigned char* data = stbi_load(filename.c_str(), &width, &height, &channels, 0);
    if (!data) {
        cerr << "Failed to load image: " << filename << endl;
        return {};
    }
    vector<uint8_t> imageData(data, data + (width * height * channels));
    stbi_image_free(data);
    return imageData;
}

std::vector<uint8_t> generateSkyMask(const std::vector<uint8_t>& motionMap, int width, int height) {
    std::vector<uint8_t> groundMask(width * height, 0);
    std::vector<uint8_t> visited(width * height, 0);

    auto idx = [&](int x, int y) { return y * width + x; };
    std::queue<pair<int,int>> q;

    // Flood fill from bottom row
    for (int x = 0; x < width; ++x) {
        int y = height - 1;
        if (motionMap[idx(x, y)] > 0) {
            q.push({x, y});
            visited[idx(x, y)] = 1;
            groundMask[idx(x, y)] = 255;
        }
    }

    const int dx[4] = {1, -1, 0, 0};
    const int dy[4] = {0, 0, 1, -1};

    while (!q.empty()) {
        auto [x, y] = q.front();
        q.pop();
        for (int k = 0; k < 4; ++k) {
            int nx = x + dx[k];
            int ny = y + dy[k];
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int nIdx = idx(nx, ny);
                if (!visited[nIdx] && motionMap[nIdx] > 0) {
                    visited[nIdx] = 1;
                    groundMask[nIdx] = 255;
                    q.push({nx, ny});
                }
            }
        }
    }

    // -----------------------------------------
    // Step 1: Morphological opening (remove small noise)
    // -----------------------------------------
    const int kernelSize = 3;
    std::vector<uint8_t> cleaned(groundMask.size(), 0);
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            int sum = 0;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    sum += groundMask[idx(x + kx, y + ky)] > 0;
                }
            }
            // Keep pixel if majority of neighbors are ground
            if (sum >= 5) cleaned[idx(x, y)] = 255;
        }
    }
    groundMask.swap(cleaned);

    // -----------------------------------------
    // Step 2: Push skyline down ~10 pixels
    // -----------------------------------------
    const int margin = 10;
    for (int x = 0; x < width; ++x) {
        // find topmost ground pixel in this column
        int topGround = -1;
        for (int y = 0; y < height; ++y) {
            if (groundMask[idx(x, y)] == 255) {
                topGround = y;
                break;
            }
        }
        // push skyline margin upward
        if (topGround != -1) {
            for (int m = 1; m <= margin && (topGround - m) >= 0; ++m) {
                groundMask[idx(x, topGround - m)] = 255;
            }
        }
    }

    // -----------------------------------------
    // Final invert to get sky mask
    // -----------------------------------------
    std::vector<uint8_t> skyMask(width * height, 0);
    for (int i = 0; i < width * height; ++i) {
        skyMask[i] = groundMask[i] ? 0 : 255;
    }

    return skyMask;
}

int main(int argc, char** argv) {

    string img1path = "cam1_0000.jpg";
    string img2path = "cam2_0002.jpg";

    int imgW1, imgH1, imgW2, imgH2, channels1, channels2;

    vector<uint8_t> img1raw = loadImage(img1path, imgW1, imgH1, channels1);
    vector<uint8_t> img2raw = loadImage(img2path, imgW2, imgH2, channels2);

    std::vector<uint8_t> skyMask = generateSkyMask(img1raw, imgW1, imgH1);

}