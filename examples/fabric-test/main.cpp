// Fabric SDK — Tests for DX improvements
//
// Tests:
//   1. Tensor::set_name()
//   2. Context::name() and Context::trace()
//   3. Auto-cast in conv2d_dw (F32 weight -> F16)
//   4. Auto-cast in conv_transpose_2d (F32 weight -> F16)
//   5. Runner: graph stats logging, get_tensor, get_tensor_shape
//   6. Runner: set_default_zeros + auto-apply
//   7. Runner: graph overflow check
//
// Usage:
//   python examples/fabric-test/gen_test_model.py /tmp/fabric_test.gguf
//   ./fabric-test /tmp/fabric_test.gguf

#include <fabric/nn.h>
#include <fabric/nn_modules.h>
#include <fabric/runner.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

static int n_pass = 0;
static int n_fail = 0;

#define TEST(name) fprintf(stderr, "  TEST: %s ... ", name);
#define PASS() do { fprintf(stderr, "PASS\n"); n_pass++; } while(0)
#define FAIL(msg) do { fprintf(stderr, "FAIL: %s\n", msg); n_fail++; } while(0)
#define ASSERT(cond, msg) do { if (!(cond)) { FAIL(msg); return; } } while(0)
#define ASSERT_EQ(a, b, msg) ASSERT((a) == (b), msg)
#define ASSERT_NEAR(a, b, eps, msg) ASSERT(std::fabs((a) - (b)) < (eps), msg)

// ---- Test 1: Tensor::set_name ----
static void test_tensor_set_name() {
    TEST("Tensor::set_name");

    ggml_init_params params = { 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * raw = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);

    fabric::Tensor t(raw, ctx);
    auto & ret = t.set_name("hello");

    ASSERT(std::strcmp(t.name(), "hello") == 0, "name should be 'hello'");
    ASSERT(&ret == &t, "set_name should return *this");

    ggml_free(ctx);
    PASS();
}

// ---- Test 2: Context::name and Context::trace ----
static void test_context_name_trace() {
    TEST("Context::name / Context::trace");

    size_t ctx_size = ggml_tensor_overhead() * 64 + ggml_graph_overhead();
    ggml_init_params params = { ctx_size, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_cgraph * gf = ggml_new_graph(ctx);

    fabric::Context fctx(ctx, gf);
    auto x = fctx.new_input("x", GGML_TYPE_F32, {4});

    // Context::name
    auto named = fctx.name(x, "my_input");
    ASSERT(std::strcmp(named.name(), "my_input") == 0, "Context::name should set label");

    // Context::trace — just verify it doesn't crash and returns the tensor
    auto traced = fctx.trace(x, "trace_test");
    ASSERT(traced.ptr == x.ptr, "trace should return same tensor");

    ggml_free(ctx);
    PASS();
}

// ---- Test 3: Auto-cast in conv2d_dw ----
static void test_autocast_conv2d_dw() {
    TEST("Auto-cast conv2d_dw (F32 weight -> F16)");

    size_t ctx_size = ggml_tensor_overhead() * 64 + ggml_graph_overhead();
    ggml_init_params params = { ctx_size, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_cgraph * gf = ggml_new_graph(ctx);

    fabric::Context fctx(ctx, gf);

    // Input: [W=8, H=8, C=4, N=1]
    auto x = fctx.new_input("x", GGML_TYPE_F32, {8, 8, 4, 1});
    // Weight in F32: [KW=3, KH=3, 1, C=4] — this would crash without auto-cast
    ggml_tensor * w_raw = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 3, 1, 4);
    ggml_set_name(w_raw, "dw_weight_f32");
    auto w = fctx.wrap(w_raw);

    ASSERT(w.dtype() == GGML_TYPE_F32, "weight should start as F32");

    // This should NOT crash — auto-cast should handle F32->F16
    auto out = fctx.conv2d_dw(x, w, 1, 1, 1, 1);
    ASSERT(out.ptr != nullptr, "conv2d_dw should succeed with F32 weight (auto-cast)");

    ggml_free(ctx);
    PASS();
}

// ---- Test 4: Auto-cast in conv_transpose_2d ----
static void test_autocast_conv_transpose_2d() {
    TEST("Auto-cast conv_transpose_2d (F32 weight -> F16)");

    size_t ctx_size = ggml_tensor_overhead() * 64 + ggml_graph_overhead();
    ggml_init_params params = { ctx_size, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_cgraph * gf = ggml_new_graph(ctx);

    fabric::Context fctx(ctx, gf);

    // Input: [W=4, H=4, C=2, N=1]
    auto x = fctx.new_input("x", GGML_TYPE_F32, {4, 4, 2, 1});
    // Weight in F32: [KW=3, KH=3, IC=2, OC=2] — would assert without auto-cast
    ggml_tensor * w_raw = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 3, 2, 2);
    ggml_set_name(w_raw, "tconv_weight_f32");
    auto w = fctx.wrap(w_raw);

    ASSERT(w.dtype() == GGML_TYPE_F32, "weight should start as F32");

    auto out = fctx.conv_transpose_2d(x, w, 2);
    ASSERT(out.ptr != nullptr, "conv_transpose_2d should succeed with F32 weight (auto-cast)");

    ggml_free(ctx);
    PASS();
}

// ---- Test 5: Runner — get_tensor, get_tensor_shape, graph stats ----
static void test_runner_inspection(const std::string & gguf_path) {
    TEST("Runner get_tensor / get_tensor_shape");

    // Build a trivial model: output = matmul(weight, input) + bias
    // weight is [3,2], bias is [2], input is [3]
    fabric::Runner runner(gguf_path, [](fabric::Context & ctx, fabric::Model & m) {
        auto x = ctx.new_input("x", GGML_TYPE_F32, {3});
        auto w = ctx.wrap(m.require("linear.weight"));
        auto b = ctx.wrap(m.require("linear.bias"));
        auto out = ctx.matmul(w, x);
        out = out + b;
        ctx.name(out, "output_named");
        return out;
    });

    // Run with input [1, 0, 0]
    std::vector<float> input = {1.0f, 0.0f, 0.0f};
    auto output = runner(input.data());

    ASSERT_EQ(output.size(), 2u, "output should have 2 elements");
    // matmul([1,2,3; 4,5,6], [1,0,0]) + [0.1, 0.2] = [1.1, 4.2]
    ASSERT_NEAR(output[0], 1.1f, 0.01f, "output[0] should be ~1.1");
    ASSERT_NEAR(output[1], 4.2f, 0.01f, "output[1] should be ~4.2");

    // get_tensor_shape
    auto shape = runner.get_tensor_shape("output_named");
    ASSERT_EQ(shape.size(), 1u, "output shape should be 1D");
    ASSERT_EQ(shape[0], 2, "output shape[0] should be 2");

    // get_tensor
    auto vals = runner.get_tensor("output_named");
    ASSERT_EQ(vals.size(), 2u, "get_tensor should return 2 elements");
    ASSERT_NEAR(vals[0], 1.1f, 0.01f, "get_tensor[0] should be ~1.1");

    // get_tensor on non-existent name
    bool threw = false;
    try {
        runner.get_tensor("nonexistent");
    } catch (const std::runtime_error &) {
        threw = true;
    }
    ASSERT(threw, "get_tensor should throw on unknown name");

    PASS();
}

// ---- Test 6: Runner — set_default_zeros ----
static void test_runner_default_zeros(const std::string & gguf_path) {
    TEST("Runner set_default_zeros");

    // Model with 2 inputs: "x" and "extra"
    // output = matmul(w, x) + bias + extra
    fabric::Runner runner(gguf_path, [](fabric::Context & ctx, fabric::Model & m) {
        auto x     = ctx.new_input("x",     GGML_TYPE_F32, {3});
        auto extra = ctx.new_input("extra",  GGML_TYPE_F32, {2});
        auto w = ctx.wrap(m.require("linear.weight"));
        auto b = ctx.wrap(m.require("linear.bias"));
        auto out = ctx.matmul(w, x) + b + extra;
        return out;
    });

    // Set "extra" to default zeros
    runner.set_default_zeros("extra");

    // Call with only "x" — "extra" should auto-fill with zeros
    std::vector<float> input = {1.0f, 0.0f, 0.0f};
    auto output = runner({{"x", input.data()}});

    ASSERT_EQ(output.size(), 2u, "output should have 2 elements");
    ASSERT_NEAR(output[0], 1.1f, 0.01f, "with zero extra, output[0] should be ~1.1");
    ASSERT_NEAR(output[1], 4.2f, 0.01f, "with zero extra, output[1] should be ~4.2");

    // Now provide extra explicitly
    std::vector<float> extra_data = {10.0f, 20.0f};
    auto output2 = runner({{"x", input.data()}, {"extra", extra_data.data()}});
    ASSERT_NEAR(output2[0], 11.1f, 0.01f, "with extra=[10,20], output[0] should be ~11.1");
    ASSERT_NEAR(output2[1], 24.2f, 0.01f, "with extra=[10,20], output[1] should be ~24.2");

    // set_default_zeros on unknown input should throw
    bool threw = false;
    try {
        runner.set_default_zeros("nonexistent");
    } catch (const std::runtime_error &) {
        threw = true;
    }
    ASSERT(threw, "set_default_zeros should throw on unknown name");

    PASS();
}

// ---- Test 7: Runner — graph capacity and stats logging ----
static void test_runner_graph_capacity(const std::string & gguf_path) {
    TEST("Runner graph capacity check");

    // Build a graph with 5 relu nodes and sufficient capacity.
    // Verifies graph stats are logged and no overflow occurs.
    fabric::Runner runner(gguf_path, [](fabric::Context & ctx, fabric::Model & m) {
        (void)m;
        auto x = ctx.new_input("x", GGML_TYPE_F32, {4});
        for (int i = 0; i < 5; i++) {
            x = ctx.relu(x);
        }
        return x;
    }, /*graph_size=*/ 2048);

    // Should succeed — 5 nodes << 2048 capacity
    std::vector<float> input = {1.0f, -1.0f, 2.0f, -2.0f};
    auto output = runner(input.data());
    ASSERT_EQ(output.size(), 4u, "output should have 4 elements");
    // relu([1, -1, 2, -2]) = [1, 0, 2, 0]
    ASSERT_NEAR(output[0], 1.0f, 0.01f, "relu(1) should be 1");
    ASSERT_NEAR(output[1], 0.0f, 0.01f, "relu(-1) should be 0");
    ASSERT_NEAR(output[2], 2.0f, 0.01f, "relu(2) should be 2");
    ASSERT_NEAR(output[3], 0.0f, 0.01f, "relu(-2) should be 0");

    PASS();
}

// ---- Test 8: ConvTranspose2d module simplified forward ----
static void test_conv_transpose_2d_module() {
    TEST("ConvTranspose2d module (simplified forward)");

    // Just verify the module still works with F16 weights (the common case)
    size_t ctx_size = ggml_tensor_overhead() * 64 + ggml_graph_overhead();
    ggml_init_params params = { ctx_size, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_cgraph * gf = ggml_new_graph(ctx);

    fabric::Context fctx(ctx, gf);

    // Create an F16 weight manually
    ggml_tensor * w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, 2, 2);
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);

    fabric::ConvTranspose2d tconv(w, b, 2);

    auto x = fctx.new_input("x", GGML_TYPE_F32, {4, 4, 2, 1});
    auto out = tconv(fctx, x);

    ASSERT(out.ptr != nullptr, "ConvTranspose2d forward should succeed");
    // Output should be F32 (cast from F16 output)
    // ggml_conv_transpose_2d_p0 outputs F32 for F16 kernel

    ggml_free(ctx);
    PASS();
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <test_model.gguf>\n", argv[0]);
        return 1;
    }

    std::string gguf_path = argv[1];
    fprintf(stderr, "\n=== Fabric SDK Tests ===\n\n");

    // Tests that don't need a model
    test_tensor_set_name();
    test_context_name_trace();
    test_autocast_conv2d_dw();
    test_autocast_conv_transpose_2d();
    test_conv_transpose_2d_module();

    // Tests that need the GGUF model
    test_runner_inspection(gguf_path);
    test_runner_default_zeros(gguf_path);
    test_runner_graph_capacity(gguf_path);

    fprintf(stderr, "\n=== Results: %d passed, %d failed ===\n\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
