#pragma once

// Post-processing for OCR pipeline:
// - Contour-based bounding box extraction for text detection (OpenCV, matches DocTR DBPostProcessor)
// - CTC greedy decoding for text recognition

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace ocr {

// --- Bounding box ---

struct BBox {
    float x0, y0, x1, y1;  // normalized [0,1] coordinates
    float confidence;
};

// Contour-based bounding box extraction from detection probability map.
// Matches DocTR's DBPostProcessor:
//   1. Binarize probability map
//   2. Find contours (external only)
//   3. For each contour: compute mean confidence, expand via "unclip", get bounding rect
// Returns bounding boxes in normalized [0,1] coordinates.
inline std::vector<BBox> extract_boxes(const float * prob_map, int width, int height,
                                        float bin_thresh = 0.3f,
                                        float box_thresh = 0.1f,
                                        float unclip_ratio = 1.5f,
                                        int min_size = 2) {
    // Wrap prob_map as cv::Mat (no copy)
    cv::Mat prob(height, width, CV_32FC1, const_cast<float *>(prob_map));

    // Binarize
    cv::Mat binary;
    cv::threshold(prob, binary, bin_thresh, 255, cv::THRESH_BINARY);
    binary.convertTo(binary, CV_8UC1);

    // Find external contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<BBox> boxes;
    float inv_w = 1.0f / width;
    float inv_h = 1.0f / height;

    for (auto & contour : contours) {
        if (contour.size() < 4) continue;

        cv::Rect brect = cv::boundingRect(contour);
        if (brect.width < min_size || brect.height < min_size) continue;

        // Compute confidence: mean probability within the contour (cropped for efficiency)
        cv::Mat crop_mask = cv::Mat::zeros(brect.height, brect.width, CV_8UC1);
        std::vector<cv::Point> shifted;
        shifted.reserve(contour.size());
        for (auto & pt : contour) {
            shifted.push_back({pt.x - brect.x, pt.y - brect.y});
        }
        cv::fillPoly(crop_mask, std::vector<std::vector<cv::Point>>{shifted}, 255);

        float score = (float)cv::mean(prob(brect), crop_mask)[0];
        if (score < box_thresh) continue;

        // Unclip: expand bounding rect by offset = area * unclip_ratio / perimeter
        // This matches DocTR's Clipper-based polygon expansion for axis-aligned text
        double area = cv::contourArea(contour);
        double perimeter = cv::arcLength(contour, true);
        if (perimeter < 1.0) continue;

        float offset = (float)(area * unclip_ratio / perimeter);

        int x0 = std::max(0, brect.x - (int)std::ceil(offset));
        int y0 = std::max(0, brect.y - (int)std::ceil(offset));
        int x1 = std::min(width,  brect.x + brect.width  + (int)std::ceil(offset));
        int y1 = std::min(height, brect.y + brect.height + (int)std::ceil(offset));

        if (x1 - x0 < min_size || y1 - y0 < min_size) continue;

        boxes.push_back({
            x0 * inv_w,
            y0 * inv_h,
            x1 * inv_w,
            y1 * inv_h,
            score
        });
    }

    // Sort by confidence descending
    std::sort(boxes.begin(), boxes.end(),
              [](const BBox & a, const BBox & b) { return a.confidence > b.confidence; });

    return boxes;
}

// --- CTC Greedy Decode ---

// Decodes logits (vocab_size+1, seq_len) into text string.
// blank_idx is typically the last class (vocab_size).
// vocab maps indices [0..vocab_size-1] to characters.
inline std::string ctc_greedy_decode(const float * logits, int vocab_plus_blank, int seq_len,
                                      const std::string & vocab) {
    int blank_idx = vocab_plus_blank - 1;

    std::string result;
    int prev_idx = -1;

    for (int t = 0; t < seq_len; t++) {
        // Find argmax for this timestep
        const float * col = logits + t * vocab_plus_blank;
        int best_idx = 0;
        float best_val = col[0];
        for (int c = 1; c < vocab_plus_blank; c++) {
            if (col[c] > best_val) {
                best_val = col[c];
                best_idx = c;
            }
        }

        // CTC rules: skip blanks and consecutive duplicates
        if (best_idx != blank_idx && best_idx != prev_idx) {
            if (best_idx < (int)vocab.size()) {
                result += vocab[best_idx];
            }
        }
        prev_idx = best_idx;
    }

    return result;
}

} // namespace ocr
