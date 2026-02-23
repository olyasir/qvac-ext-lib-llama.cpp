#pragma once

// Image loading, resizing, normalization, and cropping utilities for OCR pipeline.
// Uses OpenCV for decoding and resizing. All outputs are CHW float32 (ggml WHC memory layout).

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

namespace ocr {

struct Image {
    std::vector<float> data;  // CHW float32 (which is WHC in ggml memory order)
    int width  = 0;
    int height = 0;
    int channels = 3;

    size_t size_bytes() const { return data.size() * sizeof(float); }
};

struct PreparedImage {
    Image image;
    float scale;           // resize ratio (new / original, along the fitting axis)
    int pad_left, pad_top; // padding offset in pixels (in target space)
};

// Normalization constants — detection (DocTR db_mobilenet_v3_large defaults)
static constexpr float det_mean[] = {0.798f, 0.785f, 0.772f};
static constexpr float det_std[]  = {0.264f, 0.2749f, 0.287f};

// Normalization constants — recognition (DocTR crnn_vgg16_bn defaults)
static constexpr float rec_mean[] = {0.694f, 0.695f, 0.693f};
static constexpr float rec_std[]  = {0.299f, 0.296f, 0.301f};

// Convert cv::Mat (HWC uint8 RGB) to CHW float32 Image, with per-channel normalization
inline Image mat_to_image(const cv::Mat & mat, const float mean[3], const float std_[3]) {
    Image img;
    img.width    = mat.cols;
    img.height   = mat.rows;
    img.channels = 3;
    img.data.resize(mat.cols * mat.rows * 3);

    for (int y = 0; y < mat.rows; y++) {
        const uint8_t * row = mat.ptr<uint8_t>(y);
        for (int x = 0; x < mat.cols; x++) {
            for (int ch = 0; ch < 3; ch++) {
                float val = row[x * 3 + ch] / 255.0f;
                val = (val - mean[ch]) / std_[ch];
                img.data[ch * mat.rows * mat.cols + y * mat.cols + x] = val;
            }
        }
    }

    return img;
}

// Load image from file as cv::Mat in RGB order
inline cv::Mat load_mat(const std::string & path) {
    cv::Mat bgr = cv::imread(path, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        throw std::runtime_error("Failed to load image: " + path);
    }
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    return rgb;
}

// Load image from file, always returns 3-channel RGB CHW float32 (unnormalized [0,1])
inline Image load_image(const std::string & path) {
    cv::Mat rgb = load_mat(path);

    Image img;
    img.width    = rgb.cols;
    img.height   = rgb.rows;
    img.channels = 3;
    img.data.resize(rgb.cols * rgb.rows * 3);

    for (int y = 0; y < rgb.rows; y++) {
        const uint8_t * row = rgb.ptr<uint8_t>(y);
        for (int x = 0; x < rgb.cols; x++) {
            for (int ch = 0; ch < 3; ch++) {
                img.data[ch * rgb.rows * rgb.cols + y * rgb.cols + x] = row[x * 3 + ch] / 255.0f;
            }
        }
    }

    return img;
}

// Aspect-ratio-preserving resize with padding.
//   symmetric=true  -> center the image, pad all sides (detection)
//   symmetric=false -> top-left align, pad bottom-right (recognition)
// Padding is done pre-normalization (pad value = 0 in [0,255] space).
// Returns PreparedImage with padding metadata.
inline PreparedImage resize_preserve_ar(const cv::Mat & src, int target_w, int target_h,
                                         bool symmetric,
                                         const float mean[3], const float std_[3]) {
    float scale_w = (float)target_w / src.cols;
    float scale_h = (float)target_h / src.rows;
    float scale   = std::min(scale_w, scale_h);

    int new_w = (int)std::round(src.cols * scale);
    int new_h = (int)std::round(src.rows * scale);

    // Clamp to target (rounding can overshoot by 1)
    new_w = std::min(new_w, target_w);
    new_h = std::min(new_h, target_h);

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    // Create padded canvas (black = 0)
    cv::Mat padded = cv::Mat::zeros(target_h, target_w, CV_8UC3);

    int pad_left = 0, pad_top = 0;
    if (symmetric) {
        pad_left = (target_w - new_w) / 2;
        pad_top  = (target_h - new_h) / 2;
    }
    // else: top-left aligned, pad_left=0, pad_top=0

    resized.copyTo(padded(cv::Rect(pad_left, pad_top, new_w, new_h)));

    // Convert to CHW float32 with normalization
    Image img = mat_to_image(padded, mean, std_);

    return PreparedImage{std::move(img), scale, pad_left, pad_top};
}

// Prepare image for detection: resize to 1024x1024 with symmetric padding
inline PreparedImage prepare_detection(const std::string & path, int target_w = 1024, int target_h = 1024) {
    cv::Mat rgb = load_mat(path);
    return resize_preserve_ar(rgb, target_w, target_h, /*symmetric=*/true, det_mean, det_std);
}

// Prepare cropped region for recognition: resize to 128x32 with asymmetric padding
inline PreparedImage prepare_recognition(const cv::Mat & rgb, int target_w = 128, int target_h = 32) {
    return resize_preserve_ar(rgb, target_w, target_h, /*symmetric=*/false, rec_mean, rec_std);
}

// Prepare image file for recognition (--recognize-only path)
inline PreparedImage prepare_recognition(const std::string & path, int target_w = 128, int target_h = 32) {
    cv::Mat rgb = load_mat(path);
    return resize_preserve_ar(rgb, target_w, target_h, /*symmetric=*/false, rec_mean, rec_std);
}

// Crop a region from a cv::Mat (pixel coordinates, clamped to bounds)
inline cv::Mat crop_mat(const cv::Mat & src, int x0, int y0, int x1, int y1) {
    x0 = std::max(0, x0);
    y0 = std::max(0, y0);
    x1 = std::min(src.cols, x1);
    y1 = std::min(src.rows, y1);

    int w = x1 - x0;
    int h = y1 - y0;
    if (w <= 0 || h <= 0) {
        throw std::runtime_error("Invalid crop region");
    }

    return src(cv::Rect(x0, y0, w, h)).clone();
}

} // namespace ocr
