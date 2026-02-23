// Fabric OCR — DocTR-style text detection + recognition pipeline
//
// Usage:
//   ./fabric-ocr <detection.gguf> <recognition.gguf> <image.jpg>
//   ./fabric-ocr --recognize-only <recognition.gguf> <cropped_word.jpg>
//
// The pipeline:
//   1. Load input image
//   2. Run DBNet detection -> probability map -> bounding boxes
//   3. Remap bboxes from padded detection space to original image coords
//   4. Crop each text region
//   5. Run CRNN recognition -> CTC decode -> text
//   6. Print results

#include "../../src/fabric/runner.h"
#include "dbnet.h"
#include "crnn.h"
#include "image_utils.h"
#include "postprocess.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void print_usage(const char * prog) {
    fprintf(stderr, "Usage:\n");
    fprintf(stderr, "  %s <detection.gguf> <recognition.gguf> <image>\n", prog);
    fprintf(stderr, "  %s --recognize-only <recognition.gguf> <cropped_word_image>\n", prog);
}

// Run recognition on a single prepared image
static std::string recognize(fabric::Runner & rec_runner,
                              const ocr::Image & img,
                              const std::string & vocab,
                              int vocab_size, int seq_len) {
    // Prepare zero initial states for LSTM
    int hidden_size = 128;
    std::vector<float> zeros(hidden_size, 0.0f);

    rec_runner.alloc_graph();
    rec_runner.set_input("image", img.data.data());
    rec_runner.set_input("h0_l0_f", zeros.data());
    rec_runner.set_input("c0_l0_f", zeros.data());
    rec_runner.set_input("h0_l0_b", zeros.data());
    rec_runner.set_input("c0_l0_b", zeros.data());
    rec_runner.set_input("h0_l1_f", zeros.data());
    rec_runner.set_input("c0_l1_f", zeros.data());
    rec_runner.set_input("h0_l1_b", zeros.data());
    rec_runner.set_input("c0_l1_b", zeros.data());
    rec_runner.compute();

    auto logits = rec_runner.get_output();

    return ocr::ctc_greedy_decode(logits.data(), vocab_size + 1, seq_len, vocab);
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    bool recognize_only = false;
    int arg_idx = 1;

    if (strcmp(argv[1], "--recognize-only") == 0) {
        recognize_only = true;
        arg_idx = 2;
    }

    if (recognize_only) {
        if (argc < 4) {
            print_usage(argv[0]);
            return 1;
        }

        const char * rec_path = argv[arg_idx];
        const char * img_path = argv[arg_idx + 1];

        printf("Loading recognition model: %s\n", rec_path);

        // Build recognition runner
        fabric::Runner rec_runner(rec_path, [](fabric::Context & ctx, fabric::Model & m) -> fabric::Tensor {
            auto x = ctx.new_input("image", GGML_TYPE_F32, {128, 32, 3, 1});
            auto model = ocr::CRNN::load(m, "rec");
            return model.forward(ctx, x);
        }, /*graph_size=*/4096);

        auto & m = rec_runner.weights();
        std::string vocab = m.get_str("doctr.vocab");
        int vocab_size = m.get_u32("doctr.vocab_size");
        int seq_len = m.get_u32("doctr.seq_len");

        printf("Vocab: %d chars, seq_len: %d\n", vocab_size, seq_len);

        printf("Loading image: %s\n", img_path);
        auto prepared = ocr::prepare_recognition(std::string(img_path));

        std::string text = recognize(rec_runner, prepared.image, vocab, vocab_size, seq_len);
        printf("Recognized: %s\n", text.c_str());

        return 0;
    }

    // Full pipeline mode
    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }

    const char * det_path = argv[arg_idx];
    const char * rec_path = argv[arg_idx + 1];
    const char * img_path = argv[arg_idx + 2];

    // --- Detection ---
    double t0 = now_ms();
    printf("Loading detection model: %s\n", det_path);

    fabric::Runner det_runner(det_path, [](fabric::Context & ctx, fabric::Model & m) -> fabric::Tensor {
        auto x = ctx.new_input("image", GGML_TYPE_F32, {1024, 1024, 3, 1});
        auto model = ocr::DBNet::load(m, "det");
        return model.forward(ctx, x);
    });

    double t1 = now_ms();
    // --- Recognition ---
    printf("Loading recognition model: %s\n", rec_path);

    fabric::Runner rec_runner(rec_path, [](fabric::Context & ctx, fabric::Model & m) -> fabric::Tensor {
        auto x = ctx.new_input("image", GGML_TYPE_F32, {128, 32, 3, 1});
        auto model = ocr::CRNN::load(m, "rec");
        return model.forward(ctx, x);
    }, /*graph_size=*/4096);

    auto & rec_m = rec_runner.weights();
    std::string vocab = rec_m.get_str("doctr.vocab");
    int vocab_size = rec_m.get_u32("doctr.vocab_size");
    int seq_len = rec_m.get_u32("doctr.seq_len");

    printf("Vocab: %d chars, seq_len: %d\n", vocab_size, seq_len);
    double t2 = now_ms();

    // --- Load image and run detection ---
    printf("Loading image: %s\n", img_path);
    cv::Mat orig_rgb = ocr::load_mat(img_path);
    printf("Image size: %dx%d\n", orig_rgb.cols, orig_rgb.rows);

    printf("Running detection...\n");
    auto det_prepared = ocr::prepare_detection(std::string(img_path));
    double t3 = now_ms();
    auto prob_map = det_runner(det_prepared.image.data.data());
    double t4 = now_ms();

    // Determine probability map dimensions from output size
    int prob_total = (int)prob_map.size();
    int prob_side = (int)std::sqrt(prob_total);

    // Extract bounding boxes (in normalized [0,1] coords of the padded 1024x1024 image)
    auto boxes = ocr::extract_boxes(prob_map.data(), prob_side, prob_side);
    printf("Found %zu text regions\n", boxes.size());

    if (boxes.empty()) {
        printf("No text detected.\n");
        return 0;
    }

    // --- Remap bboxes from padded detection space to original image coords ---
    // Detection used symmetric padding: the actual image content is centered.
    int det_w = det_prepared.image.width;   // 1024
    int det_h = det_prepared.image.height;  // 1024
    float pad_x_norm = (float)det_prepared.pad_left / det_w;
    float pad_y_norm = (float)det_prepared.pad_top  / det_h;
    float content_w_px = det_w - 2 * det_prepared.pad_left;  // symmetric
    float content_h_px = det_h - 2 * det_prepared.pad_top;
    float content_w_norm = content_w_px / det_w;
    float content_h_norm = content_h_px / det_h;

    // --- Run recognition on each region ---
    printf("\n--- Recognized text ---\n");
    double t5 = now_ms();
    double rec_compute_ms = 0;

    for (size_t i = 0; i < boxes.size(); i++) {
        auto & box = boxes[i];

        // Remap from padded-image normalized coords to original-image normalized coords
        float orig_x0 = (box.x0 - pad_x_norm) / content_w_norm;
        float orig_y0 = (box.y0 - pad_y_norm) / content_h_norm;
        float orig_x1 = (box.x1 - pad_x_norm) / content_w_norm;
        float orig_y1 = (box.y1 - pad_y_norm) / content_h_norm;

        // Clamp to [0,1]
        orig_x0 = std::clamp(orig_x0, 0.0f, 1.0f);
        orig_y0 = std::clamp(orig_y0, 0.0f, 1.0f);
        orig_x1 = std::clamp(orig_x1, 0.0f, 1.0f);
        orig_y1 = std::clamp(orig_y1, 0.0f, 1.0f);

        // Convert to pixel coords in original image (unclip already expanded the boxes)
        int x0 = (int)(orig_x0 * orig_rgb.cols);
        int y0 = (int)(orig_y0 * orig_rgb.rows);
        int x1 = (int)(orig_x1 * orig_rgb.cols);
        int y1 = (int)(orig_y1 * orig_rgb.rows);

        // Crop from original image and prepare for recognition
        cv::Mat cropped = ocr::crop_mat(orig_rgb, x0, y0, x1, y1);
        auto prepared = ocr::prepare_recognition(cropped);

        double tr0 = now_ms();
        std::string text = recognize(rec_runner, prepared.image, vocab, vocab_size, seq_len);
        double tr1 = now_ms();
        rec_compute_ms += (tr1 - tr0);

        printf("[%zu] (%.1f%%) [%d,%d,%d,%d]: %s\n",
               i, box.confidence * 100.0f,
               x0, y0, x1, y1,
               text.c_str());
    }

    double t6 = now_ms();
    printf("\n--- Timing ---\n");
    printf("Load det model:    %7.0f ms\n", t1 - t0);
    printf("Load rec model:    %7.0f ms\n", t2 - t1);
    printf("Preprocess:        %7.0f ms\n", t3 - t2);
    printf("Detection infer:   %7.0f ms\n", t4 - t3);
    printf("Post-process:      %7.0f ms\n", t5 - t4);
    printf("Recognition total: %7.0f ms  (%zu words, %.0f ms/word)\n",
           t6 - t5, boxes.size(), (t6 - t5) / boxes.size());
    printf("  rec compute:     %7.0f ms  (%.0f ms/word)\n",
           rec_compute_ms, rec_compute_ms / boxes.size());
    printf("  rec overhead:    %7.0f ms  (crop+prep+decode)\n", (t6 - t5) - rec_compute_ms);
    printf("Total:             %7.0f ms\n", t6 - t0);

    return 0;
}
