# ViT-Engine-CUDA

A hardware-accelerated, custom CUDA backend for the Vision Transformer (ViT-Base-16). This engine achieves strict mathematical parity with the PyTorch `timm` baseline, bypassing the ATen dispatcher to execute highly tuned mathematical kernels directly on NVIDIA Ampere architecture. Final evaluation score: 199/200 on the Imagenette validation subset.

## Progress Documentation

The development of this engine transitioned from individual mathematical operations to a fully integrated PyTorch extension. The progression and resolution of critical roadblocks are documented below.

### Phase 1: Foundational Kernels
The initial stage mapped standard tensor operations to CUDA grids.
* **Patch Embedding:** Implemented a custom 2D convolution kernel mapping 16x16 pixel patches to 768-dimensional vectors. 
* **Positional Encoding:** Implemented element-wise addition for pre-trained spatial encodings and handled the memory concatenation of the global classification (CLS) token. Fixed an out-of-bounds memory read that initially caused accuracy drops by ensuring the single CLS vector was correctly mapped across all batch dimensions.
* **Classification Head:** Built a vector-matrix multiplication kernel to isolate the 0th sequence index (CLS token) and project it to target class probabilities.

### Phase 2: Transformer Block Components
* **Layer Normalization:** Developed a block-reduction kernel computing mean and variance across the 768-embedding dimension using hardware-level warp shuffle operations (`__shfl_down_sync`).
* **Multi-Layer Perceptron (MLP):** Developed the feed-forward network expanding to 3072 dimensions. Fixed initial missing bias additions.

### Phase 3: Flash Attention 2 for Ampere
The self-attention mechanism was explicitly tuned for RTX 30-series (Ampere) hardware.
* Setup asynchronous memory pipelines using `__pipeline_memcpy_async` to bypass the L1 cache.
* Integrated double buffering with shared memory tiles to overlap math execution with data fetching.
* Applied Online Softmax to fuse the calculation inside SRAM.
* **Bug Fix:** Corrected an intra-warp reduction issue where `__shfl_down_sync` defaulted to 32 threads. Bounding it strictly to 16 threads prevented mathematical cross-contamination between adjacent tokens processed in the same warp.

### Phase 4: Precision and Memory Alignment
Reaching mathematical parity with PyTorch required resolving hardware and memory discrepancies.
* **Contiguous Memory:** Standard PyTorch slices utilize strided memory. The engine was updated to explicitly enforce contiguous memory layouts before extracting raw pointers in the C++ wrappers, resolving massive L2 divergence cascades.
* **TF32 vs FP32:** PyTorch defaults to TensorFloat-32 on Ampere. The test suite was explicitly configured to disable TF32, ensuring PyTorch matched the strict 23-bit FP32 precision (`__fmaf_rn`) executed by the custom kernels.
* **GELU Precision:** The hardware-accelerated `tanh` approximation initially caused a 91.22 L2 divergence due to a missing scaling constant. This was replaced with the standard mathematical error function (`erff`) scaled by `0.70710678f`, achieving near 0.000000 L2 divergence against PyTorch's native GELU.

### Phase 5: Final Validation
The final hurdle was a batch dimension bug in the grid launch configurations. Hardcoded batch sizes of `1` in the Pybind11 wrappers caused the engine to correctly process only the first image of any batch, returning uninitialized memory for the rest. Scaling the `dim3 grid` configurations dynamically with the batch dimension brought the validation score to 199/200.

## Repository Structure

### Core Extension (`ext/`)
Contains all C++ and CUDA source code.
* `setup.py`: Compilation script invoking NVCC and building the Pybind11 shared library.
* `binding.cpp`: The Pybind11 registry exposing C++ functions to Python.
* `*_wrapper.cpp`: Thin C++ bridges extracting raw float pointers from `at::Tensor` objects, ensuring memory contiguity, and launching CUDA grids.
* `*.cu`: The raw CUDA kernels (`attention.cu`, `layernorm.cu`, `mlp.cu`, `patch_embed.cu`, `pos_encoding.cu`, `classifier.cu`).

### Testing and Validation (`tests/` & `scripts/`)
* `tests/test_*.py`: Isolated unit tests comparing every individual CUDA kernel against its equivalent PyTorch `torch.nn` layer.
* `scripts/layerwise_compare.py`: Executes a full forward pass, extracting intermediate activations from both `timm` and `vit_cuda`, calculating L2 norm and max absolute differences for every block.
* `scripts/eval_imagenette.py`: The main validation script. Loads the Imagenette dataset via `DataLoader`, runs batched inference, calculates Top-1 and Top-5 accuracy, and outputs a PCA projection of the embeddings.
* `scripts/debug_*.py`: Isolated scratchpad scripts used to trace specific mathematical divergences.

### High-Level Application
* `inference.py`: Standard evaluation pipeline. Instantiates the custom `ViTCUDA` python class, downloads `timm` pre-trained weights, assigns them as registered buffers, and executes image classification.
* `benchmark.py`: Latency measurement tool using `torch.cuda.Event` to measure execution time differences between the custom backend and native PyTorch.
* `app.py`: Gradio web application for interactive inference using webcam feeds or uploaded images.

## Technical Architecture Details

1. **Memory:** The engine requires strictly contiguous memory blocks. Strided memory is not supported. All PyTorch `[out_features, in_features]` weight matrices are handled natively within the kernels without requiring pre-transposition.
2. **Grid Parameters:** 768 embedding dimensions and 12 attention heads are hardcoded directly into the C++ configuration to maximize register allocation limits and SM occupancy.
3. **Read-Only Caching:** Input pointers are decorated with `const float* __restrict__` to force routing through the GPU's dedicated read-only data cache via `__ldg()` intrinsics.

## Usage

**Compilation:**
```bash
cd ext
python setup.py install
```

### Architecture Image
![vit](docs/architecture.png)

Git hooks
---------

To run tests automatically before pushing, install the repo hooks:

Unix / Git Bash:

```bash
./scripts/install_hooks.sh
```

Windows PowerShell:

```powershell
./scripts/install_hooks.ps1
```

The pre-push hook will build the extension (editable install) and run `pytest -q`. If tests fail, the push is aborted.
