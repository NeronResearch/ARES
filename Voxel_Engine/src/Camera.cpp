#include "Camera.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <limits>
#include "../third_party/stb_image.h"
#include "../third_party/stb_image_write.h"

constexpr float M_PI = 3.14159265358979323846f;

// Camera Implementation
Camera::Camera(XYZ position,
               Matrix3x3 rotation,
               float sensorSize,
               int fov,
               int imageWidth,
               int imageHeight,
               std::vector<uint8_t> imageData)
    : position(position),
      rotation(rotation),
      sensorSize(sensorSize),
      fov(fov),
      imageWidth(imageWidth),
      imageHeight(imageHeight),
      imageData(std::move(imageData)) {}

std::vector<uint8_t> Camera::loadImage(const std::string& filename, int& width, int& height, int& channels) {
    unsigned char* data = stbi_load(filename.c_str(), &width, &height, &channels, 0);
    if (!data) return {};
    
    std::vector<uint8_t> imageData(data, data + (width * height * channels));
    stbi_image_free(data);
    return imageData;
}

std::vector<uint8_t> Camera::convertToGrayscale(const std::vector<uint8_t>& imageData, int width, int height, int channels) {
    if (channels == 1) return imageData;
    
    std::vector<uint8_t> grayscaleData(width * height);
    
    if (channels == 3) {
        #pragma omp parallel for simd
        for (int i = 0; i < width * height; ++i) {
            int pixelIndex = i * 3;
            grayscaleData[i] = (imageData[pixelIndex] + imageData[pixelIndex + 1] + imageData[pixelIndex + 2]) / 3;
        }
    } else if (channels == 4) {
        #pragma omp parallel for simd
        for (int i = 0; i < width * height; ++i) {
            int pixelIndex = i * 4;
            grayscaleData[i] = (imageData[pixelIndex] + imageData[pixelIndex + 1] + imageData[pixelIndex + 2]) / 3;
        }
    }
    
    return grayscaleData;
}

std::vector<uint8_t> Camera::computeMotionMap(const std::vector<uint8_t>& img1, 
                                                       const std::vector<uint8_t>& img2,
                                                       int width, int height) {
    std::vector<uint8_t> motionMap(width * height, 0);
    
    const int step = 8; // SETTING!
    
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y += step) {
        for (int x = 0; x < width; x += step) {
            int idx = y * width + x;
            if (idx < static_cast<int>(img1.size())) {
                int diff = std::abs(static_cast<int>(img1[idx]) - static_cast<int>(img2[idx]));
                uint8_t motion = static_cast<uint8_t>(std::min(255, diff * 4));
                
                for (int dy = 0; dy < step && (y + dy) < height; ++dy) {
                    for (int dx = 0; dx < step && (x + dx) < width; ++dx) {
                        motionMap[(y + dy) * width + (x + dx)] = motion;
                    }
                }
            }
        }
    }
    
    return motionMap;
}

Ray Camera::generateRay(int px, int py) const {
    // Pre-compute constants to avoid repeated calculations
    static thread_local float aspectRatio = static_cast<float>(imageWidth) / imageHeight;
    static thread_local float fovRad = fov * (M_PI / 180.0f);
    static thread_local float tanHalfFovX = std::tan(fovRad / 2.0f);
    static thread_local float tanHalfFovY = tanHalfFovX / aspectRatio;
    static thread_local float invWidth = 2.0f / imageWidth;
    static thread_local float invHeight = 2.0f / imageHeight;
    
    // Optimized NDC calculation
    float ndcX = (px + 0.5f) * invWidth - 1.0f;
    float ndcY = 1.0f - (py + 0.5f) * invHeight;

    float camX = ndcX * tanHalfFovX;
    float camY = ndcY * tanHalfFovY;
    float camZ = -1.0f;

    // Direct matrix transformation without intermediate XYZ object
    float dirX = rotation.m[0][0]*camX + rotation.m[0][1]*camY + rotation.m[0][2]*camZ;
    float dirY = rotation.m[1][0]*camX + rotation.m[1][1]*camY + rotation.m[1][2]*camZ;
    float dirZ = rotation.m[2][0]*camX + rotation.m[2][1]*camY + rotation.m[2][2]*camZ;

    // Fast inverse square root approximation for normalization
    float lenSq = dirX*dirX + dirY*dirY + dirZ*dirZ;
    float invLen = 1.0f / std::sqrt(lenSq);

    XYZ dirWorld(dirX * invLen, dirY * invLen, dirZ * invLen);
    return { position, dirWorld };
}

float Camera::getPixelBrightness(int px, int py) const {
    // Branchless bounds checking for better performance
    bool inBounds = (px >= 0) & (px < imageWidth) & (py >= 0) & (py < imageHeight);
    if (!inBounds) return 0.0f;
    
    // Direct array access with pre-computed index
    const size_t index = static_cast<size_t>(py) * imageWidth + px;
    return imageData[index] * (1.0f / 255.0f);  // Multiply instead of divide
}

int Camera::getImageWidth() const { return imageWidth; }
int Camera::getImageHeight() const { return imageHeight; }
const Matrix3x3& Camera::getRotation() const { return rotation; }
const XYZ& Camera::getPosition() const { return position; }
int Camera::getFOV() const { return fov; }
