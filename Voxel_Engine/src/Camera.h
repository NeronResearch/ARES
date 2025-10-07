#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include "XYZ.h"
#include "Matrix3x3.h"

struct Ray {
    XYZ origin;
    XYZ direction;
    int intersects;
};

class Camera {
public:
    Camera(XYZ position,
           Matrix3x3 rotation,
           float sensorSize,
           int fov,
           int imageWidth,
           int imageHeight,
           std::vector<uint8_t> imageData);

    // Static utility methods for image processing
    static std::vector<uint8_t> loadImage(const std::string& filename, int& width, int& height, int& channels);
    static std::vector<uint8_t> convertToGrayscale(const std::vector<uint8_t>& imageData, int width, int height, int channels);
    static std::vector<uint8_t> computeMotionMap(const std::vector<uint8_t>& img1, 
                                                         const std::vector<uint8_t>& img2,
                                                         int width, int height);

    // Ray generation and pixel operations
    Ray generateRay(int px, int py) const;
    float getPixelBrightness(int px, int py) const;

    // Getters
    int getImageWidth() const;
    int getImageHeight() const;
    const Matrix3x3& getRotation() const;
    const XYZ& getPosition() const;
    int getFOV() const;

private:
    XYZ position;
    Matrix3x3 rotation;
    float sensorSize;
    int fov;
    int imageWidth;
    int imageHeight;
    std::vector<uint8_t> imageData;
};
