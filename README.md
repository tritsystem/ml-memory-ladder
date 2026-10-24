# ML Memory & Cache Systems Ladder

Interactive 8-level study guide, spaced-repetition quiz (64 Leitner-box questions), and 16 dual-language (**C++ & Python**) executable code labs for low-level ML memory allocation fixes, GPU `CUDACachingAllocator` / stream lifetime bugs, and LLM KV-cache (`DynamicCache`, `StaticCache`, circular ring-buffer) state management.

Same shape as [ML Ladder](https://github.com/tritsystem/ml-ladder) and [Godot Ladder](https://github.com/tritsystem/godot-ladder): every exercise includes a full **Variable & Parameter Reference Table**, a **100% Line-by-Line Code Walkthrough** for both C++ and Python, and an in-page C++ & Python test runner (with visible + hidden edge-case tests).

Live: https://tritsystem.github.io/ml-memory-ladder/

Full textbook study guide: [`STUDY_GUIDE.md`](./STUDY_GUIDE.md)

## 8 Levels & 16 Dual-Language (C++ & Python) Labs

| Level | Topic | Exercises (`C++` & `Python`) | Real OSS Bug / Finding |
|---|---|---|---|
| **1** | Raw Pointers, Stack vs. Heap & `NULL` Guards | `1.1` Unchecked `mmap`/`malloc` `NULL` Guard (`init_shm_workspace`)<br>`1.2` Compound Dereference Guard (`parse_packet_header`) | `microsoft/DeepSpeed` `#8565` (`csrc/aio/py_test/test_ds_aio.cpp`) |
| **2** | C/C++ Heap Lifetimes, Reassignment Leaks & Early-Exit Cleanup | `2.1` Pointer Reassignment & Early-Exit Leak (`verify_identity_packet`)<br>`2.2` Use-After-Free vs. Leak on Early Return (`remove_filament`) | `cfengine/core` `#6358`, `eclipse-mosquitto/mosquitto` `#3762`, `OpenSC` `#3762` |
| **3** | Tensor Views, Strides, Aliasing & Storage Lifetimes | `3.1` Per-Iteration Scratch Leak (`run_inference_batches`)<br>`3.2` Dangling Tensor View After Storage Free (`extract_SubView_safe`) | `ggerganov/llama.cpp` `llama_kv_cache_init`, `Root-project/root` `#21681` |
| **4** | Integer Overflow in Allocation Size & Dtype Width Mismatches | `4.1` 32-Bit Element-Count Overflow Before Allocation (`safe_tensor_alloc_bytes`)<br>`4.2` KV-Cache Dtype Byte-Width Mismatch (`update_kv_cache_safe`) | `pytorch/executorch` `#23399` (`QuantizedKVCache` `bf16` vs `fp32` `E_INVALID_ARGUMENT`) |
| **5** | In-Place Mutation vs. Out-of-Place Allocation & Buffer Aliasing | `5.1` In-Place Ring Counter Aliasing (`build_attention_positions`)<br>`5.2` Persistent `nn.Module` Buffer Probability Update (`run_ada_epochs`) | GPT-Neo `StaticCache` (`cumulative_length` in-place mutation), `kornia/kornia` `ADA` `p` buffer |
| **6** | GPU `CUDACachingAllocator`, Fragmentation & Reference Cycles | `6.1` Caching Allocator Best-Fit Pool & OOM Retry (`sim_cuda_caching_alloc`)<br>`6.2` Autograd Computation Graph Leak in Metrics Loop (`accumulate_epoch_loss`) | PyTorch `CUDACachingAllocator`, `Lightning-AI/torchmetrics` `WrapperMetric` |
| **7** | CUDA Streams, Asynchronous Frees & Pinned Memory Races | `7.1` Cross-Stream Use-After-Free & `record_stream` (`sim_cross_stream_alloc`)<br>`7.2` Page-Aligned Pinned Staging Buffer Cleanup (`alloc_pinned_staging`) | `CUDACachingAllocator::recordStream`, `cudaHostAlloc` / `cudaFreeHost` |
| **8** | LLM KV-Cache (`DynamicCache`, `StaticCache`) & Ring-Buffer Lifetimes | `8.1` `DynamicLayer` & `QuantizedLayer` `.reset()` Bug (`reset_and_update_dynamic_cache`)<br>`8.2` Circular Ring-Buffer Checkpoint & Cursor Resume (`resume_ring_buffer`) | `huggingface/transformers` `#48509` / `#48809`, `Lightning-AI/torchmetrics` `Running` `state_dict` |

## Verification

Run the verification suite (`16/16` Python + `16/16` C++ solutions PASS, `16/16` starters FAIL RED, `tree-sitter-cpp` AST clean, `64/64` quiz questions balanced):

```bash
python3 test_ml_memory_ladder.py
```
