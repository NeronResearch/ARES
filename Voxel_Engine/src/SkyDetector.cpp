#include "SkyDetector.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <memory>
#include <chrono>

// Enable compiler optimizations
#ifdef _MSC_VER
    #pragma optimize("O2", on)
#endif

// OpenMP for parallel processing if available
#ifdef _OPENMP
    #include <omp.h>
#endif

SkyDetector::SkyDetector() {
    initializeKernels();
}

void SkyDetector::initializeKernels() {
    // Use smaller kernels for faster morphological operations
    rectKernel_ = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    ellipseKernel_ = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    sharpenKernel_ = (cv::Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
}

cv::Mat SkyDetector::findContours(const cv::Mat& thresholdImg, int nrow, int ncol) {
    // Reuse temporary matrix if possible
    if (tempMat1_.rows != nrow || tempMat1_.cols != ncol) {
        tempMat1_ = cv::Mat::zeros(nrow, ncol, CV_8UC1);
    } else {
        tempMat1_.setTo(0);
    }
    cv::Mat& mask = tempMat1_;
    
    const double area = static_cast<double>(nrow) * ncol;
    const double halfA = 0.4 * area;
    const double halfY = ncol * 0.5;
    
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(thresholdImg, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    if (contours.empty()) {
        return mask.clone();
    }
    
    // Pre-calculate contour areas and bounding rects for efficiency
    std::vector<double> areas;
    std::vector<cv::Rect> boundingRects;
    areas.reserve(contours.size());
    boundingRects.reserve(contours.size());
    
    for (const auto& cnt : contours) {
        areas.push_back(cv::contourArea(cnt));
        boundingRects.push_back(cv::boundingRect(cnt));
    }
    
    // Find the largest contour index
    auto maxAreaIt = std::max_element(areas.begin(), areas.end());
    size_t maxContourIdx = std::distance(areas.begin(), maxAreaIt);
    
    // Process contours with optimized logic
    for (size_t i = 0; i < contours.size(); ++i) {
        const cv::Rect& rect = boundingRects[i];
        
        // Early exit conditions
        if (rect.y != 0) continue; // Must start from top
        
        const int bottomVertex = rect.y + rect.height;
        const double contourArea = areas[i];
        
        if (i == maxContourIdx) { // Biggest contour
            if ((contourArea >= halfA && rect.x == 0) || (bottomVertex <= halfY)) {
                cv::drawContours(mask, contours, static_cast<int>(i), cv::Scalar(255), cv::FILLED);
            }
        } else if (contourArea > 1000 && bottomVertex <= halfY) {
            // Calculate shape match only when needed
            double match = cv::matchShapes(contours[i], contours[maxContourIdx], cv::CONTOURS_MATCH_I1, 0.0);
            if (match < 0.1) { // More lenient threshold for efficiency
                cv::drawContours(mask, contours, static_cast<int>(i), cv::Scalar(255), cv::FILLED);
            }
        }
    }
    
    return mask.clone();
}

double SkyDetector::calculateSSIM(const cv::Mat& img1, const cv::Mat& img2) {
    // Ultra-fast SSIM approximation using mean values only
    cv::Scalar mean1 = cv::mean(img1);
    cv::Scalar mean2 = cv::mean(img2);
    cv::Scalar stddev1, stddev2;
    cv::meanStdDev(img1, mean1, stddev1);
    cv::meanStdDev(img2, mean2, stddev2);
    
    // Simplified correlation coefficient as SSIM approximation
    double mu1 = mean1[0];
    double mu2 = mean2[0];
    double sigma1 = stddev1[0];
    double sigma2 = stddev2[0];
    
    // Covariance approximation
    cv::Mat diff1, diff2;
    img1.convertTo(diff1, CV_32F, 1.0, -mu1);
    img2.convertTo(diff2, CV_32F, 1.0, -mu2);
    
    cv::Mat covariance;
    cv::multiply(diff1, diff2, covariance);
    cv::Scalar cov = cv::mean(covariance);
    
    // Fast SSIM approximation
    const double C1 = 6.5025;
    const double C2 = 58.5225;
    
    double numerator = (2.0 * mu1 * mu2 + C1) * (2.0 * cov[0] + C2);
    double denominator = (mu1 * mu1 + mu2 * mu2 + C1) * (sigma1 * sigma1 + sigma2 * sigma2 + C2);
    
    return (denominator > 0) ? numerator / denominator : 0.0;
}

cv::Mat SkyDetector::detectSkyMaskInternal(const cv::Mat& img) {
    // std::cout << "SkyDetector: Input image size: " << img.cols << "x" << img.rows 
              // << ", channels: " << img.channels() << std::endl;
    
    // Resize image to smaller standard size for faster processing (4x less pixels)
    constexpr int rRow = 320, rCol = 240;
    cv::Mat imgResized;
    cv::resize(img, imgResized, cv::Size(rRow, rCol), 0, 0, cv::INTER_LINEAR);
    // std::cout << "SkyDetector: Resized to: " << imgResized.cols << "x" << imgResized.rows << std::endl;
    
    // Extract blue channel directly without split for better performance
    cv::Mat blueChannel(imgResized.rows, imgResized.cols, CV_8UC1);
    cv::extractChannel(imgResized, blueChannel, 0); // Extract blue channel directly (index 0 = blue in BGR)
    
    // Use faster blur with smaller kernel
    cv::Mat blueGaussian;
    cv::blur(blueChannel, blueGaussian, cv::Size(3, 3));
    
    // Apply Otsu's thresholding
    cv::Mat th;
    double val = cv::threshold(blueGaussian, th, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    const int nrow = blueChannel.rows;
    const int ncol = blueChannel.cols;
    
    // Find initial contours
    cv::Mat mask = findContours(th, nrow, ncol);
    
    // Use pre-created kernels
    cv::Scalar total = cv::sum(mask);
    cv::Mat finalMask;
    
    if (total[0] == 0) {
        // Night image or no sky detected - return empty mask
        finalMask = cv::Mat::zeros(nrow, ncol, CV_8UC1);
    } else {
        // Simplified day image processing - skip expensive SSIM calculation
        // Just use basic morphological operations for speed
        cv::morphologyEx(mask, finalMask, cv::MORPH_OPEN, ellipseKernel_, cv::Point(-1, -1), 1);
        
        // Optional: Add a simple edge-based refinement if needed
        cv::Scalar maskMean = cv::mean(mask);
        if (maskMean[0] > 128) {  // Large sky region detected
            cv::Mat edges;
            cv::Canny(blueChannel, edges, val * 0.3, val * 0.7);
            cv::bitwise_and(edges, mask, edges);
            
            // Simple morphological closing on edges
            cv::Mat closing;
            cv::morphologyEx(edges, closing, cv::MORPH_CLOSE, rectKernel_);
            
            // Combine with original mask
            cv::Mat combined;
            cv::bitwise_or(finalMask, closing, combined);
            cv::morphologyEx(combined, finalMask, cv::MORPH_OPEN, ellipseKernel_);
        }
    }
    
    // std::cout << "SkyDetector: Final mask size: " << finalMask.cols << "x" << finalMask.rows << std::endl;
    return finalMask;
}

std::vector<uint8_t> SkyDetector::detectSkyMask(const std::vector<uint8_t>& imageData, 
                                                 int width, int height, int channels) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (imageData.empty() || width <= 0 || height <= 0 || channels < 1) {
        throw std::runtime_error("Invalid image parameters");
    }
    
    if (imageData.size() != static_cast<size_t>(width * height * channels)) {
        throw std::runtime_error("Image data size doesn't match dimensions");
    }
    
    // Store original dimensions
    int originalWidth = width;
    int originalHeight = height;
    
    // Create OpenCV Mat from input data
    cv::Mat img;
    if (channels == 3) {
        // Convert RGB to BGR for OpenCV (fix channel order)
        std::vector<uint8_t> bgrData(imageData.size());
        for (size_t i = 0; i < imageData.size(); i += 3) {
            bgrData[i] = imageData[i + 2];     // B (blue from RGB)
            bgrData[i + 1] = imageData[i + 1]; // G (green)
            bgrData[i + 2] = imageData[i];     // R (red from RGB)
        }
        img = cv::Mat(height, width, CV_8UC3, bgrData.data()).clone();
    } else if (channels == 1) {
        img = cv::Mat(height, width, CV_8UC1, const_cast<uint8_t*>(imageData.data())).clone();
    } else {
        throw std::runtime_error("Unsupported number of channels. Only 1 or 3 channels supported.");
    }
    
    // Process the image (this returns a 640x480 mask)
    cv::Mat mask = detectSkyMaskInternal(img);
    
    // CRITICAL FIX: Resize mask back to original image dimensions
    cv::Mat finalMask;
    if (mask.cols != originalWidth || mask.rows != originalHeight) {
        cv::resize(mask, finalMask, cv::Size(originalWidth, originalHeight), 0, 0, cv::INTER_NEAREST);
        // std::cout << "Resized mask from " << mask.cols << "x" << mask.rows 
                  // << " back to original " << originalWidth << "x" << originalHeight << std::endl;
    } else {
        finalMask = mask;
    }
    
    // Convert result back to vector with correct size
    std::vector<uint8_t> result(finalMask.total());
    std::memcpy(result.data(), finalMask.data, finalMask.total());
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Sky detection completed in " << duration.count() << "ms (output size: " 
              << originalWidth << "x" << originalHeight << ")" << std::endl;
    
    return result;
}

std::vector<uint8_t> SkyDetector::detectSkyMaskFromFile(const std::string& imagePath, 
                                                        int& outWidth, int& outHeight) {
    // Load image using OpenCV for this convenience method
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (img.empty()) {
        throw std::runtime_error("Could not load image from " + imagePath);
    }
    
    outWidth = img.cols;
    outHeight = img.rows;
    
    // Convert to RGB vector
    cv::Mat rgbImg;
    cv::cvtColor(img, rgbImg, cv::COLOR_BGR2RGB);
    
    std::vector<uint8_t> imageData(rgbImg.total() * rgbImg.channels());
    std::memcpy(imageData.data(), rgbImg.data, imageData.size());
    
    return detectSkyMask(imageData, outWidth, outHeight, 3);
}