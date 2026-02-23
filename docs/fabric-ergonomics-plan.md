# Fabric SDK — PyTorch-Level Ergonomics for C++ Inference

## Context: Why PyTorch Won (and Others Didn't)

Before PyTorch, ML frameworks failed for the same reason: **they made simple things hard.**

| Framework | Why it lost | Fabric parallel |
|-----------|-------------|-----------------|
| **Caffe** | Config-file driven, hard to extend | — |
| **Theano** | Symbolic graphs, slow compilation, unreadable errors | — |
| **TensorFlow 1.x** | Sessions, placeholders, explicit graph, verbose | `ctx` threading everywhere, 8-arg functions |
| **Chainer** | Had the right ideas but lost momentum | — |

**PyTorch's 5 success factors:**

1. **Tensors feel like numbers** — `x + 1.0`, `x * y`, `x.reshape(-1)`. No ceremony.
2. **Modules are callable** — `y = model(x)`, not `y = model.forward(session, x)`
3. **Progressive disclosure** — beginners use `nn.Linear`, experts use `F.linear`
4. **One way to do things** — one `Linear`, one `Conv2d`, consistent `.forward()` everywhere
5. **Framework disappears** — you think about your model, not the framework

**Fabric's current state:** The high-level Pipeline is clean, but the model-definition layer has accumulated TF1-style friction: explicit context passing, 8+ positional args, duplicate types (`Linear` vs `GenericLinear`), manual bias broadcasting, and no scalar arithmetic on tensors.

**Goal:** Apply PyTorch's principles to make Fabric's C++ inference DSL feel natural. A user should think about their model architecture, not about ggml internals.

---

## The Plan: 3 Tiers of Improvement

### Tier 1: "Tensors Feel Like Numbers" (Trivial, Ship Together)

These are all additive one-liners. Zero regression risk. Massive ergonomic payoff.

#### 1a. Scalar arithmetic on Tensor

**Before:** `Tensor y = ctx.scale(x, 0.5f);` — can't do `x * 0.5f`

**After:**
```cpp
Tensor y = x * 0.5f;    // ggml_scale
Tensor z = x + 1.0f;    // ggml_add1 or scale_bias
Tensor w = x - 2.0f;    // ggml_add1 with -2.0f
Tensor q = x / 3.0f;    // ggml_scale(x, 1/3)
Tensor r = 2.0f * x;    // commutative
```

**Files:** `tensor.h` (declarations), `tensor.cpp` (implementations using `ggml_scale`, `ggml_add1`)

#### 1b. Tensor introspection properties

**Before:** `x.ptr->type` / `ggml_nelements(x.ptr)` — leaks ggml

**After:**
```cpp
x.dtype()       // ggml_type
x.numel()       // total elements
x.nbytes()      // total bytes
x.ndim()        // meaningful dimensions
x.stride(i)     // byte stride
x.name()        // tensor name (from ggml)
```

**Files:** `tensor.h` (inline one-liners)

#### 1c. Tensor member methods (reshape, flatten, activations)

**Before:** `ctx.reshape_3d(x, a, b, c)` / `ctx.relu(x)` / `ctx.flatten(x)`

**After:**
```cpp
x.reshape(a, b, c)      // overloaded 1-4 args
x.flatten()
x.contiguous()
x.permute(0, 2, 1, 3)
x.transpose()
x.relu()                 // activations as member methods
x.gelu()
x.sigmoid()
x.silu()
x.scale(0.5f)
x.clamp(0.0f, 1.0f)
```

Enables chaining: `x = conv1(ctx, x).relu().flatten();`

**Files:** `tensor.h` (declarations), `tensor.cpp` (implementations, all use stored `ctx_`)

#### 1d. Free-function activations (PyTorch F.relu style)

```cpp
x = fabric::relu(x);
x = fabric::gelu(x);
x = fabric::sigmoid(x);
```

**Files:** `tensor.h` or new `functional.h` — inline one-liners using `x.ctx_`

#### 1e. Debug printing

```cpp
std::cerr << x;   // Tensor[768, 32] f32 "Qcur"
x.print();         // same, to stderr
```

**Files:** `tensor.h` (`operator<<` overload)

---

### Tier 2: "Modules Are Callable" (Easy, High Impact)

#### 2a. operator() on all modules

**Before:** `x = conv1.forward(ctx, x);`
**After:** `x = conv1(ctx, x);`

One-line addition per module — `operator()` delegates to `forward()`.

**Files:** `nn_modules.h` (Conv2d, Conv1d, GenericLinear, GenericLayerNorm, GenericRMSNorm, Dropout), `modules.h` (Linear, RMSNorm, LayerNorm, FusedQKVLinear)

#### 2b. Unify Linear / GenericLinear

**Problem:** Two types doing the same thing. `Linear` uses LoRA+GraphContext. `GenericLinear` uses matmul+Context.

**Solution:** Single `Linear` with overloaded `forward`:
```cpp
struct Linear {
    Tensor forward(Context & ctx, Tensor x) const;       // generic: matmul
    Tensor forward(GraphContext & ctx, Tensor x) const;   // LLM: lora_mm
    // operator() for both
};
```

`GenericLinear` becomes `using GenericLinear = Linear;` for backward compat.

Same pattern for `LayerNorm`/`GenericLayerNorm` and `RMSNorm`/`GenericRMSNorm`.

**Files:** `modules.h`, `nn_modules.h`

#### 2c. Module::load() factory (prefix-based weight loading)

**Before:**
```cpp
Conv2d conv1(m.require("conv1.weight"), m.tensor("conv1.bias"), /*s=*/1, /*p=*/1);
```

**After:**
```cpp
auto conv1 = Conv2d::load(m, "conv1", {.padding=1});
auto fc1 = Linear::load(m, "fc1");
```

Convention: `prefix.weight` required, `prefix.bias` optional (nullptr if missing).

**Files:** `nn_modules.h` (static factory methods)

---

### Tier 3: "Framework Disappears" (Moderate, Cleanup)

#### 3a. ConvParams struct (kill positional arg explosion)

**Before:** `Conv2d(w, b, 1, 1, 1)` — what is what?

**After:**
```cpp
Conv2d(w, b, {.stride=1, .padding=1})               // C++20 designated init
Conv2d::load(m, "conv1", {.stride=2, .padding=3})   // with load()
```

```cpp
struct ConvParams {
    int stride = 1, padding = 0, dilation = 1;
};
```

**Files:** `nn_modules.h`

#### 3b. Runner reuse (pre-allocate, run many times)

**Before:** `runner.compute(inputs)` re-allocates every call

**After:**
```cpp
runner.prepare();                              // allocate once
runner.set_input("image", img1.data());
runner.run();
auto scores1 = runner.get_output();

runner.set_input("image", img2.data());        // reuse allocation
runner.run();
auto scores2 = runner.get_output();
```

**Files:** `runner.h`

#### 3c. Update CNN example as showcase

Rewrite `examples/fabric-cnn/main.cpp` to use all the new ergonomics:

```cpp
struct MnistCNN {
    fabric::Conv2d conv1, conv2;
    fabric::Linear fc1, fc2;

    MnistCNN(fabric::Model & m)
        : conv1(Conv2d::load(m, "conv1", {.padding=1}))
        , conv2(Conv2d::load(m, "conv2", {.padding=1}))
        , fc1(Linear::load(m, "fc1"))
        , fc2(Linear::load(m, "fc2")) {}

    fabric::Tensor forward(fabric::Context & ctx, fabric::Tensor x) {
        x = conv1(ctx, x).relu();
        x = ctx.max_pool2d(x, 2, 2);
        x = conv2(ctx, x).relu();
        x = ctx.max_pool2d(x, 2, 2);
        x = fc1(ctx, x.flatten()).relu();
        return fc2(ctx, x);
    }
};
```

Compare: `conv1(ctx, x).relu()` vs old `ctx.relu(conv1.forward(ctx, x))`.

---

## Files Summary

| File | Changes |
|------|---------|
| `src/fabric/tensor.h` | Scalar ops, member methods (reshape/flatten/relu/...), introspection, debug print, free-function activations |
| `src/fabric/tensor.cpp` | Implementations for all new Tensor operators and methods |
| `src/fabric/nn_modules.h` | operator(), ConvParams, Linear unification, load() factories |
| `src/fabric/modules.h` | operator() on LLM modules, Linear unification |
| `src/fabric/runner.h` | prepare()/set_input()/run()/get_output() for reuse |
| `examples/fabric-cnn/main.cpp` | Rewrite to showcase new ergonomics |

**Untouched:** All 16 LLM model files, context.h, nn.h, model.h, pipeline.h, fabric.h, all llama.cpp source.

## What C++ Cannot Match (Inherent Limitations)

1. **Explicit ctx passing** — ggml uses static graphs, so a context must exist for graph building. Tensor stores `ctx_` which helps, but module forward() still needs it. This is the one TF1-ism we can't fully eliminate.
2. **No auto parameter collection** — no C++ reflection, so no `model.parameters()`. Users list modules manually.
3. **No dynamic shapes** — ggml graphs are concrete. Can't build once and run with different sizes.
4. **No kwargs** — C++20 designated initializers (`{.stride=2}`) are the closest approximation.

## Verification

1. **Build:** `cmake --build build -j$(nproc)` — all existing targets compile
2. **LLM regression:** Build and run `fabric-external-example` (XGLM) — identical output
3. **CNN showcase:** Updated `fabric-cnn` example compiles, runs, produces correct MNIST predictions
4. **New features:** Verify scalar ops (`x * 0.5f`), method chaining (`x.relu().flatten()`), operator() (`conv1(ctx, x)`), load() factories, runner reuse all work in the CNN example
