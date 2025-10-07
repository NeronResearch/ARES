#include "ImageDenoise.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <string>
#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../third_party/stb_image.h"
#include "../third_party/stb_image_write.h"

ImageDenoise::ImageDenoise(int width, int height, const Settings& settings)
    : width(width), height(height), settings(settings) {}

std::vector<uint8_t> ImageDenoise::denoise(const std::vector<uint8_t>& inputImage) {
    if (inputImage.size() != static_cast<size_t>(width * height)) {
        throw std::runtime_error("Input image size doesn't match expected dimensions");
    }
    
    // Save the input image as raw.jpg
    saveImageAsJPEG(inputImage, "raw.jpg");
    
    std::vector<uint8_t> result;
    switch (settings.method) {
        case Method::MEDIAN:
            result = medianFilter(inputImage, settings.kernelSize);
            break;
        case Method::GAUSSIAN:
            result = gaussianBlur(inputImage, settings.kernelSize, settings.gaussianSigma);
            break;
        case Method::BILATERAL:
            result = bilateralFilter(inputImage, settings.kernelSize, settings.bilateralSigmaColor, settings.bilateralSigmaSpace);
            break;
        case Method::MORPHOLOGICAL:
            result = morphologicalClean(inputImage, settings.kernelSize);
            break;
        case Method::THRESHOLD_CLEAN:
            result = thresholdClean(inputImage, settings.noiseThreshold, settings.minNeighborCount);
            break;
        default:
            result = inputImage; // Return original if unknown method
            break;
    }
    
    // Save the output image as denoised.jpg
    saveImageAsJPEG(result, "denoised.jpg");
    
    return result;
}

std::vector<uint8_t> ImageDenoise::medianFilter(const std::vector<uint8_t>& inputImage, int kernelSize) {
    std::vector<uint8_t> result(inputImage.size());
    const int radius = kernelSize / 2;
    
    #pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            std::vector<uint8_t> neighborhood = getNeighborhood(inputImage, x, y, kernelSize);
            
            // Sort and find median
            std::sort(neighborhood.begin(), neighborhood.end());
            result[getIndex(x, y)] = neighborhood[neighborhood.size() / 2];
        }
    }
    
    return result;
}

std::vector<uint8_t> ImageDenoise::gaussianBlur(const std::vector<uint8_t>& inputImage, int kernelSize, float sigma) {
    std::vector<uint8_t> result(inputImage.size());
    const int radius = kernelSize / 2;
    
    #pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            float weightSum = 0.0f;
            
            for (int ky = -radius; ky <= radius; ++ky) {
                for (int kx = -radius; kx <= radius; ++kx) {
                    int nx = x + kx;
                    int ny = y + ky;
                    
                    if (isValidPixel(nx, ny)) {
                        float weight = gaussianWeight(kx, ky, sigma);
                        sum += inputImage[getIndex(nx, ny)] * weight;
                        weightSum += weight;
                    }
                }
            }
            
            result[getIndex(x, y)] = static_cast<uint8_t>(std::clamp(sum / weightSum, 0.0f, 255.0f));
        }
    }
    
    return result;
}

std::vector<uint8_t> ImageDenoise::bilateralFilter(const std::vector<uint8_t>& inputImage, int kernelSize, float sigmaColor, float sigmaSpace) {
    std::vector<uint8_t> result(inputImage.size());
    const int radius = kernelSize / 2;
    const float colorCoeff = -0.5f / (sigmaColor * sigmaColor);
    const float spaceCoeff = -0.5f / (sigmaSpace * sigmaSpace);
    
    #pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const uint8_t centerValue = inputImage[getIndex(x, y)];
            float sum = 0.0f;
            float weightSum = 0.0f;
            
            for (int ky = -radius; ky <= radius; ++ky) {
                for (int kx = -radius; kx <= radius; ++kx) {
                    int nx = x + kx;
                    int ny = y + ky;
                    
                    if (isValidPixel(nx, ny)) {
                        const uint8_t neighborValue = inputImage[getIndex(nx, ny)];
                        
                        // Spatial weight (distance in image)
                        float spatialDist = kx * kx + ky * ky;
                        float spatialWeight = std::exp(spatialDist * spaceCoeff);
                        
                        // Color weight (intensity difference)
                        float colorDist = static_cast<float>(centerValue - neighborValue);
                        colorDist *= colorDist;
                        float colorWeight = std::exp(colorDist * colorCoeff);
                        
                        float totalWeight = spatialWeight * colorWeight;
                        sum += neighborValue * totalWeight;
                        weightSum += totalWeight;
                    }
                }
            }
            
            result[getIndex(x, y)] = static_cast<uint8_t>(std::clamp(sum / weightSum, 0.0f, 255.0f));
        }
    }
    
    return result;
}

std::vector<uint8_t> ImageDenoise::morphologicalClean(const std::vector<uint8_t>& inputImage, int kernelSize) {
    // Morphological opening (erosion followed by dilation) to remove small noise
    std::vector<uint8_t> eroded(inputImage.size());
    std::vector<uint8_t> result(inputImage.size());
    
    // Erosion pass
    #pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            eroded[getIndex(x, y)] = morphologicalErode(inputImage, x, y, kernelSize);
        }
    }
    
    // Dilation pass
    #pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            result[getIndex(x, y)] = morphologicalDilate(eroded, x, y, kernelSize);
        }
    }
    
    return result;
}

std::vector<uint8_t> ImageDenoise::thresholdClean(const std::vector<uint8_t>& inputImage, uint8_t noiseThreshold, uint8_t minNeighborCount) {
    std::vector<uint8_t> result = inputImage; // Start with copy
    const int radius = 1; // Use 3x3 neighborhood
    
    #pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const uint8_t centerValue = inputImage[getIndex(x, y)];
            
            // Only process pixels that might be noise (below threshold)
            if (centerValue <= noiseThreshold) {
                uint8_t brightNeighborCount = 0;
                
                // Count bright neighbors
                for (int ky = -radius; ky <= radius; ++ky) {
                    for (int kx = -radius; kx <= radius; ++kx) {
                        if (kx == 0 && ky == 0) continue; // Skip center pixel
                        
                        int nx = x + kx;
                        int ny = y + ky;
                        
                        if (isValidPixel(nx, ny) && inputImage[getIndex(nx, ny)] > noiseThreshold) {
                            brightNeighborCount++;
                        }
                    }
                }
                
                // If not enough bright neighbors, consider this isolated noise and remove it
                if (brightNeighborCount < minNeighborCount) {
                    result[getIndex(x, y)] = 0; // Set to black (remove noise)
                }
            }
        }
    }
    
    return result;
}

// Helper function implementations
uint8_t ImageDenoise::getPixelSafe(const std::vector<uint8_t>& image, int x, int y) const {
    if (isValidPixel(x, y)) {
        return image[getIndex(x, y)];
    }
    return 0; // Return black for out-of-bounds
}

std::vector<uint8_t> ImageDenoise::getNeighborhood(const std::vector<uint8_t>& image, int centerX, int centerY, int kernelSize) const {
    std::vector<uint8_t> neighborhood;
    const int radius = kernelSize / 2;
    neighborhood.reserve(kernelSize * kernelSize);
    
    for (int ky = -radius; ky <= radius; ++ky) {
        for (int kx = -radius; kx <= radius; ++kx) {
            int nx = centerX + kx;
            int ny = centerY + ky;
            neighborhood.push_back(getPixelSafe(image, nx, ny));
        }
    }
    
    return neighborhood;
}

float ImageDenoise::gaussianWeight(int dx, int dy, float sigma) const {
    float dist = dx * dx + dy * dy;
    return std::exp(-dist / (2.0f * sigma * sigma));
}

uint8_t ImageDenoise::morphologicalErode(const std::vector<uint8_t>& image, int centerX, int centerY, int kernelSize) const {
    const int radius = kernelSize / 2;
    uint8_t minValue = 255;
    
    for (int ky = -radius; ky <= radius; ++ky) {
        for (int kx = -radius; kx <= radius; ++kx) {
            int nx = centerX + kx;
            int ny = centerY + ky;
            if (isValidPixel(nx, ny)) {
                minValue = std::min(minValue, image[getIndex(nx, ny)]);
            }
        }
    }
    
    return minValue;
}

uint8_t ImageDenoise::morphologicalDilate(const std::vector<uint8_t>& image, int centerX, int centerY, int kernelSize) const {
    const int radius = kernelSize / 2;
    uint8_t maxValue = 0;
    
    for (int ky = -radius; ky <= radius; ++ky) {
        for (int kx = -radius; kx <= radius; ++kx) {
            int nx = centerX + kx;
            int ny = centerY + ky;
            if (isValidPixel(nx, ny)) {
                maxValue = std::max(maxValue, image[getIndex(nx, ny)]);
            }
        }
    }
    
    return maxValue;
}

void ImageDenoise::saveImageAsJPEG(const std::vector<uint8_t>& image, const std::string& filename, int quality) const {
    // STB image write expects data in a specific format
    // For grayscale images, we need to provide 1 component per pixel
    int result = stbi_write_jpg(filename.c_str(), width, height, 1, image.data(), quality);
    
    if (!result) {
        throw std::runtime_error("Failed to save image: " + filename);
    }
}