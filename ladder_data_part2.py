# -*- coding: utf-8 -*-
"""
Levels 5-8 of the ML Memory & Cache Systems Ladder.
Covers:
  Level 5: Python & PyTorch Memory Internals (Views, In-Place Alias Mutation & Persistent Buffers)
  Level 6: GPU & CUDA Caching Allocator Mechanics (Pools, Fragmentation & Autograd Graph Leaks)
  Level 7: Async GPU Streams & Pinned Host Memory (record_stream Lifetime & mlock Cleanup)
  Level 8: LLM KV-Cache & Ring-Buffer Memory Management (DynamicCache.reset & Rolling Window State)
"""

LEVELS_5_8 = [
    "5. Python Tensors, Views & Persistent Buffers",
    "6. GPU Caching Allocator & Autograd Leaks",
    "7. Async CUDA Streams & Pinned Memory",
    "8. LLM KV-Cache & Ring-Buffer Lifetimes",
]

QUIZ_5_8 = [
    # ---------------- Level 5: Python Tensors, Views & Persistent Buffers ----------------
    {
        "l": 5,
        "q": "In PyTorch, when you slice a tensor `view = x[0:4]`, does `view` get its own new copy of the numbers in RAM/VRAM?",
        "o": [
            "No: `view` is a new header (`TensorImpl`) pointing to the exact same underlying `Storage` buffer",
            "Yes: slicing a tensor in PyTorch always calls `malloc` and copies the 4 numbers",
            "Only on the CPU; on the GPU slicing always makes a full deep copy",
            "It deletes `x` so that only `view` can read those 4 numbers from that point on",
        ],
        "w": "A PyTorch `Tensor` has two parts: a lightweight metadata header (`shape`, `stride`, `storage_offset`, `dtype`) and a heavy `Storage` byte buffer. Slicing `x[0:4]` creates a new header sharing the exact same `Storage`!",
    },
    {
        "l": 5,
        "q": "In your GPT-Neo `StaticCache` local-attention bug fix (`gpt_neo_static_cache_fix.md`), why did `past_len = cache_layer.cumulative_length` BEFORE `cache.update(...)` give the WRONG `past_len` after `update()`?",
        "o": [
            "`cumulative_length` was a 1-element tensor that `update()` incremented in-place (`+=`), mutating `past_len` too because both pointed to the same storage!",
            "`cache.update()` deleted `cumulative_length` and replaced it with a Python string",
            "Python integers are mutable objects that change whenever any variable is added to them",
            "`StaticCache` does not track how many tokens have been seen so `cumulative_length` is always 0",
        ],
        "w": "Because `past_len = cache_layer.cumulative_length` only copied the tensor reference (sticky note), `StaticLayer.update()` doing `self.cumulative_length += q_len` in-place mutated `past_len` behind the caller's back! Calling `.clone()` snapshotted the true pre-update value.",
    },
    {
        "l": 5,
        "q": "What is the difference between `x = x + 1` (out-of-place) and `x += 1` or `x.add_(1)` (in-place) on a PyTorch tensor?",
        "o": [
            "`x = x + 1` allocates a brand-new buffer and rebinds `x`, while `x += 1` overwrites the existing `Storage` buffer shared by all views of `x`",
            "They are 100% identical in memory; `+=` is just shorter to type",
            "`x = x + 1` mutates all other views of `x`, while `x += 1` makes a private copy first",
            "`x += 1` only works on CPU tensors and raises a syntax error on CUDA tensors",
        ],
        "w": "In PyTorch, trailing-underscore methods (`zero_()`, `add_()`, `copy_()`) and `+=` mutate the underlying `Storage` in-place. Every other variable or slice sharing that `Storage` sees its values change immediately!",
    },
    {
        "l": 5,
        "q": "In your `LDAMLoss.m_list` (`pytorch_optimizer`) and `SinusoidalPosEmb` findings, what goes wrong when an `nn.Module` stores a tensor as `self.m_list = torch.tensor(...)` instead of `self.register_buffer('m_list', ...)`?",
        "o": [
            "`model.to('cuda')` and `model.half()` skip `self.m_list` (leaving it on CPU / fp32) and `state_dict()` omits it!",
            "PyTorch immediately deletes any `self.*` tensor attribute that is not registered within 1 second",
            "`self.m_list` gets trained by the optimizer as if it were a learnable `nn.Parameter` weight",
            "`register_buffer` is only needed for strings and integers, never for tensors",
        ],
        "w": "`nn.Module.to()`, `.half()`, `.cuda()`, and `.state_dict()` only walk `_parameters`, `_buffers`, and child `_modules`. A plain `self.t = tensor` attribute is invisible to all of them!",
    },
    {
        "l": 5,
        "q": "In your `Kornia` `AdaptiveDiscriminatorAugmentation` (`ADA`) finding, why did resuming training from a `state_dict()` checkpoint silently break discriminator training?",
        "o": [
            "The probability `p`, EMA stat, and step counter were plain Python floats/ints not saved in `state_dict`, so resuming reset `p` from `0.8` back to `1e-5`",
            "`state_dict()` saved `p` twice and multiplied them together on reload",
            "Kornia's `load_state_dict()` deleted all convolutional weights in the discriminator",
            "The learning rate was set to `NaN` whenever `state_dict()` was called on GPU",
        ],
        "w": "Real T4 finding from your ledger (`20261019_063000_kornia-ada-state-dict-finding.md`): any stateful module whose behavior evolves over training steps (`p`, running EMA, step counters) MUST persist that state in `state_dict` / `register_buffer` or checkpoint resume silently corrupts training.",
    },
    {
        "l": 5,
        "q": "In your `TorchMetrics` `WrapperMetric` finding (`20261019_190000_torchmetrics-wrapper-persistent-finding.md`), why did calling `wrapper.persistent(True)` still return an empty `state_dict()`?",
        "o": [
            "`WrapperMetric.persistent()` only looped over `self._defaults` (empty on the outer wrapper) and forgot to forward `.persistent(mode)` to the inner wrapped metric(s)!",
            "`state_dict()` in PyTorch never saves buffers on subclasses of `nn.Module`",
            "`persistent(True)` actually means 'delete all buffers before saving'",
            "The inner metric's tensors were stored on disk instead of in RAM",
        ],
        "w": "Because `BootStrapper`, `MultioutputWrapper`, `MinMaxMetric`, etc. hold their real metric state on `self.metric` (a child module) and have `self._defaults = {}` on the outer wrapper, `WrapperMetric.persistent(True)` was a complete no-op until forwarded to child metrics!",
    },
    {
        "l": 5,
        "q": "Why does calling `.contiguous()` on a transposed PyTorch tensor `t = x.transpose(0, 1)` sometimes allocate new memory and sometimes return `t` untouched?",
        "o": [
            "`transpose` only swaps the `stride` metadata without moving bytes in RAM; `.contiguous()` allocates and copies bytes only if the strides are no longer in row-major order",
            "`.contiguous()` allocates new memory on odd-numbered calls and reuses memory on even-numbered calls",
            "`.contiguous()` only allocates memory when the tensor contains `NaN` or `Inf` values",
            "`transpose(0, 1)` already copies all bytes in RAM, so `.contiguous()` never allocates",
        ],
        "w": "When a C++/CUDA kernel requires contiguous row-major memory (`ptr[i * cols + j]`), calling `t.contiguous()` checks `t.is_contiguous()`. If `False` (as after `transpose`), it allocates a fresh buffer and copies the elements into row-major order.",
    },
    {
        "l": 5,
        "q": "In your `Project-MONAI` finding (`20261019_180000_monai-first-call-config-mutation.md`), why did calling a transform (`ImageHistogram` / `RandDeformGrid(spacing=None)`) on a second image with a different shape crash?",
        "o": [
            "The first call overwrote `self.spacing` / `self.bins` in-place on `self` with values sized for the first image's shape!",
            "MONAI transforms automatically delete themselves after being called once",
            "Python garbage-collected the second image before the transform started",
            "The transform converted all images into 1D strings on the first call",
        ],
        "w": "Mutating `self.config_attr` lazily on the first forward call turns a reusable stateless transform into a shape-locked trap when the next sample has a different resolution or channel count.",
    },

    # ---------------- Level 6: GPU Caching Allocator & Autograd Leaks ----------------
    {
        "l": 6,
        "q": "Why doesn't PyTorch call NVIDIA's raw `cudaMalloc` and `cudaFree` directly on every tensor creation and destruction?",
        "o": [
            "Raw `cudaMalloc`/`cudaFree` synchronize the entire GPU and stall all streams for hundreds of microseconds, so PyTorch reuses blocks inside `CUDACachingAllocator`",
            "NVIDIA GPUs only allow `cudaMalloc` to be called once per hour by the operating system",
            "`cudaFree` does not exist in CUDA C++; GPU memory can never be freed once allocated",
            "`CUDACachingAllocator` stores all tensors on the CPU hard drive instead of GPU VRAM",
        ],
        "w": "Calling raw OS/driver `cudaMalloc` or `cudaFree` forces a global device synchronization across the GPU. PyTorch's `CUDACachingAllocator` keeps freed GPU blocks in a fast C++ pool so the next tensor of similar size reuses a block in nanoseconds without calling the CUDA driver!",
    },
    {
        "l": 6,
        "q": "In PyTorch CUDA memory stats, what is the difference between `memory_allocated()` and `memory_reserved()`?",
        "o": [
            "`allocated` is VRAM currently held by live tensors; `reserved` is total VRAM held by PyTorch's caching pool (`allocated` + cached free blocks)",
            "`allocated` is CPU RAM and `reserved` is GPU VRAM",
            "`reserved` is always smaller than `allocated` because it excludes model weights",
            "They are two names for the exact same counter in `CUDACachingAllocator`",
        ],
        "w": "When a tensor is deleted in Python (`del x`), `memory_allocated()` drops immediately, but `memory_reserved()` stays unchanged because the block sits in `CUDACachingAllocator`'s free pool ready for the next tensor! `nvidia-smi` shows `memory_reserved()` (plus CUDA context overhead).",
    },
    {
        "l": 6,
        "q": "What does `torch.cuda.empty_cache()` actually do — and what does it NOT do?",
        "o": [
            "It returns unused cached blocks from PyTorch's free pool back to the CUDA driver, but it CANNOT free any memory still held by live tensors",
            "It deletes all live tensors and model weights on the GPU so training starts over with 0 bytes used",
            "It doubles the physical VRAM on the graphics card by compressing float32 tensors into int4",
            "It fixes memory leaks caused by appending live `loss` tensors to a Python list",
        ],
        "w": "`empty_cache()` only releases blocks sitting idle in the caching allocator's free list (`reserved - allocated`). If your training loop leaked live tensor references, `empty_cache()` frees zero bytes of those leaked tensors and just slows the loop down!",
    },
    {
        "l": 6,
        "q": "How does external fragmentation cause a `CUDA out of memory` error even when `reserved - allocated` shows 4 GB of free cached VRAM?",
        "o": [
            "The 4 GB of free pool memory is chopped into many small non-adjacent blocks, so a single contiguous 512 MB tensor request cannot fit in any one block",
            "The GPU refuses to use the last 4 GB of VRAM unless the monitor cable is unplugged",
            "`CUDACachingAllocator` forgets to search its free list whenever more than 2 blocks are free",
            "PyTorch reserves 4 GB exclusively for error messages and stack traces",
        ],
        "w": "Every tensor needs a single contiguous virtual address range. If varying sequence lengths leave 4,000 separate 1 MB free blocks scattered between live blocks, a 512 MB request misses every free block and fails if `cudaMalloc` has no room left on the device!",
    },
    {
        "l": 6,
        "q": "When `CUDACachingAllocator` finds a 10 MB free block in its pool for a 2 MB tensor request, what does it do so the remaining 8 MB isn't wasted?",
        "o": [
            "It splits the 10 MB block into a 2 MB active block (returned to the caller) and an 8 MB free remainder block (put back into the free pool)",
            "It throws away the remaining 8 MB permanently until the process exits",
            "It refuses to use the 10 MB block and calls `cudaMalloc(2 MB)` instead",
            "It shrinks the tensor to 0 bytes so it doesn't have to touch the 10 MB block",
        ],
        "w": "Block splitting (and coalescing adjacent freed blocks back together when both halves are freed) is the core mechanism of `CUDACachingAllocator` (`BlockPool` in `CUDACachingAllocator.cpp`).",
    },
    {
        "l": 6,
        "q": "Why does writing `epoch_losses.append(loss)` inside a PyTorch training loop cause GPU VRAM to climb on every batch until it crashes with OOM?",
        "o": [
            "`loss` is a live autograd Tensor with a `.grad_fn` pointer that keeps the entire step's forward activation graph alive in VRAM!",
            "A Python float inside `epoch_losses` takes 500 MB of GPU memory per number",
            "`append()` copies the model's weights onto a new GPU every time it is called",
            "Lists in Python are stored inside the GPU's L1 cache by default",
        ],
        "w": "One of the most common real-world PyTorch GPU memory leaks: `loss` points to `grad_fn`, which points to the saved activations of every layer in that batch! Appending `loss.detach().item()` (or `float(loss.detach())`) stores a plain 8-byte CPU number and lets the entire GPU graph free immediately.",
    },
    {
        "l": 6,
        "q": "When `CUDACachingAllocator` cannot find a big enough free block in its pool and `cudaMalloc` also fails with OOM, what is its built-in retry step before raising `OutOfMemoryError`?",
        "o": [
            "It calls `release_cached_blocks()` (returning all idle free-pool blocks to CUDA) and retries `cudaMalloc` once",
            "It deletes random model weights until enough VRAM is free for the new tensor",
            "It silently moves the tensor to CPU RAM without telling the program",
            "It waits 60 seconds and asks the user to type a smaller batch size",
        ],
        "w": "Inside `CUDACachingAllocator.cpp`, an OOM from `cudaMalloc` triggers an automatic `release_cached_blocks()` (the C++ body of `empty_cache()`) so fragmented idle blocks are merged back in the CUDA driver before retrying `cudaMalloc`.",
    },
    {
        "l": 6,
        "q": "How do CUDA Virtual Memory Management (`expandable_segments:True` in `PYTORCH_CUDA_ALLOC_CONF`) and PagedAttention (vLLM) solve VRAM fragmentation?",
        "o": [
            "They map physical VRAM pages into contiguous virtual ranges (or fixed-size KV blocks) so non-adjacent physical chunks can back a grown tensor without copying",
            "They disable GPU caching completely and run `cudaMalloc` on every single token",
            "They truncate every prompt to 16 tokens so all tensors have the exact same size",
            "They store the KV cache as JPEG images to avoid floating-point allocations",
        ],
        "w": "`expandable_segments:True` uses `cuMemAddressReserve` + `cuMemMap` so physical 2 MB pages can be stitched onto an existing virtual range without reallocating+copying, while PagedAttention splits KV caches into fixed-size pages (zero external fragmentation!).",
    },

    # ---------------- Level 7: Async CUDA Streams & Pinned Memory ----------------
    {
        "l": 7,
        "q": "What does it mean that CUDA kernel launches and `cudaMemcpyAsync` are 'asynchronous' with respect to the CPU?",
        "o": [
            "The CPU queues the command onto the GPU stream and immediately moves to the next line of CPU code BEFORE the GPU has even started or finished the work",
            "The GPU runs the kernel backwards from the last line to the first line",
            "The CPU freezes and waits until the GPU finishes every calculation before running the next line",
            "The kernel runs on a random GPU in the data center without telling the CPU",
        ],
        "w": "Analogy: the CPU is a restaurant manager putting an order slip on a conveyor belt (`CUDA stream`) for the kitchen (`GPU`). The manager immediately walks to the next line of Python/C++ while the kitchen is still cooking 5 orders back!",
    },
    {
        "l": 7,
        "q": "Why does `cudaMemcpyAsync(gpu_ptr, cpu_ptr, bytes, H2D, stream)` require `cpu_ptr` to be in **pinned (page-locked)** host memory (`pin_memory=True` / `cudaHostAlloc` / `mlock`)?",
        "o": [
            "The GPU's DMA engine copies directly from physical RAM addresses; if the OS paged `cpu_ptr` out to disk mid-copy, the DMA would read garbage or fault!",
            "Pageable CPU memory can only store integers and cannot store floating-point tensors",
            "Pinned memory is stored inside the GPU fan controller instead of system RAM",
            "`cudaMemcpyAsync` deletes `cpu_ptr` before copying it unless it is pinned",
        ],
        "w": "Without page-locking (`mlock` / `cudaHostAlloc`), the OS virtual-memory manager can move or swap a physical RAM page to disk while the GPU's Direct Memory Access (DMA) hardware is reading that physical address in the background!",
    },
    {
        "l": 7,
        "q": "In DeepSpeed's `csrc/pin_memory/page_alloc.cpp` (`ds_page_aligned_alloc`, which you analyzed in Reagent `C02`), what must happen if `mlock(ptr, size)` returns non-zero (fails to pin)?",
        "o": [
            "It must call `free(ptr)` to release the heap buffer and immediately `return nullptr;` so the unpinned buffer is neither leaked nor used after free",
            "It must call `free(ptr)` and then fall through to `return ptr;` so the caller gets the freed address",
            "It must ignore the `mlock` failure and leak `ptr` by allocating a second buffer",
            "It must call `free(ptr)` twice to clear both virtual and physical memory",
        ],
        "w": "In `ds_page_aligned_alloc`, if `mlock(ptr, size) != 0` (e.g., the user hit Linux's `ulimit -l` `RLIMIT_MEMLOCK` cap), the function logs the error, calls `free(ptr)`, and returns `nullptr` immediately.",
    },
    {
        "l": 7,
        "q": "What is the classic **Cross-Stream Use-After-Free** bug in `CUDACachingAllocator` when Tensor `x` is created on `stream_A`, passed to a kernel on `stream_B`, and `del x` runs on the CPU right after?",
        "o": [
            "`del x` marks `x`'s block free on `stream_A` immediately, so a new tensor on `stream_A` can reuse and overwrite `x`'s VRAM while `stream_B`'s kernel is still reading `x`!",
            "The GPU crashes with a blue screen whenever two streams exist in the same process",
            "`del x` waits for every stream on all GPUs to finish by calling `cudaDeviceSynchronize()`",
            "`stream_B` automatically makes a deep copy of `x` on the CPU hard drive",
        ],
        "w": "Because the CPU finishes queueing `kernel<<<..., stream_B>>>(x)` in microseconds and hits `del x` while `stream_B` is still busy, `CUDACachingAllocator` thinks `x`'s block is safe to reuse on `stream_A`! A new tensor on `stream_A` then stomps on `x`'s bytes mid-kernel!",
    },
    {
        "l": 7,
        "q": "How does calling `x.record_stream(stream_B)` in PyTorch prevent that Cross-Stream Use-After-Free bug?",
        "o": [
            "It attaches a CUDA event on `stream_B` to `x`'s block so `CUDACachingAllocator` will not hand that block out again until `stream_B` passes that event",
            "It copies `x` back to the CPU and runs `stream_B`'s kernel in pure Python",
            "It permanently locks `x`'s VRAM block so it can never be freed for the rest of the program",
            "It merges `stream_A` and `stream_B` into a single synchronous CPU thread",
        ],
        "w": "`x.record_stream(stream_B)` tells `CUDACachingAllocator`: 'When `x` is freed, insert a CUDA event on `stream_B` and keep this block in a pending-events list until `cudaEventQuery` confirms `stream_B` has finished using it!'",
    },
    {
        "l": 7,
        "q": "What happens if you allocate a CPU tensor `cpu_buf`, launch an async copy `gpu_buf.copy_(cpu_buf, non_blocking=True)`, and immediately overwrite `cpu_buf[0] = 999` on the next Python line?",
        "o": [
            "A CPU- vs-DMA race condition: `gpu_buf` may receive `999` instead of the original value because the async DMA copy had not read `cpu_buf[0]` yet!",
            "`copy_(non_blocking=True)` always finishes copying before returning to Python",
            "PyTorch raises a `RuntimeError` if you touch `cpu_buf` within 10 minutes of `copy_`",
            "`gpu_buf` and `cpu_buf` become permanently linked so changing one always changes the other",
        ],
        "w": "With `non_blocking=True` on pinned memory, `copy_` returns to Python while the PCIe DMA engine is still queued! Overwriting or freeing `cpu_buf` before recording/waiting on a stream event corrupts the in-flight transfer.",
    },
    {
        "l": 7,
        "q": "Why shouldn't you set `pin_memory=True` on every single tensor in your entire program?",
        "o": [
            "Pinned memory locks physical RAM pages so the OS cannot swap them out; pinning too much starves the OS and other processes of physical RAM",
            "Pinned memory is 100x slower for the CPU to read than normal memory",
            "Only 1 tensor per computer is allowed to use `pin_memory=True`",
            "Pinned memory only works with 8-bit integers and corrupts 32-bit floats",
        ],
        "w": "Page-locked (pinned) host memory is a scarce OS resource (often capped by `RLIMIT_MEMLOCK`) and takes much longer to allocate than regular heap memory. Use a reusable pinned staging buffer (or DataLoader `pin_memory=True` ring) for H2D/D2H transfers.",
    },
    {
        "l": 7,
        "q": "In CUDA Graphs (`torch.cuda.CUDAGraph`), how are memory addresses handled across graph replays?",
        "o": [
            "Every kernel in a captured CUDA Graph has its input/output GPU pointers baked in at capture time, so replaying the graph reads and writes those exact fixed VRAM addresses",
            "A CUDA Graph calls `cudaMalloc` fresh from the OS on every single replay",
            "CUDA Graphs move all tensors to random addresses on each replay for security",
            "CUDA Graphs only capture Python `print()` statements and ignore GPU memory",
        ],
        "w": "Because a captured CUDA Graph bakes raw virtual pointers directly into the recorded command buffer, freeing or reallocating an input/output tensor outside the graph's private memory pool causes a silent GPU Use-After-Free on every replay!",
    },

    # ---------------- Level 8: LLM KV-Cache & Ring-Buffer Lifetimes ----------------
    {
        "l": 8,
        "q": "During LLM text generation, what is the purpose of the KV-Cache (Key/Value Cache)?",
        "o": [
            "It stores the Key and Value vectors of all past tokens so generating token `t+1` only computes K/V for the 1 new token instead of recomputing all `t` previous tokens",
            "It caches the entire English dictionary on the hard drive so the tokenizer runs faster",
            "It backs up the model's weights to a cloud database after every generated word",
            "It compresses the user's prompt by deleting every second word before attention runs",
        ],
        "w": "Without a KV-Cache, generating 1,000 tokens recomputes the past tokens on every step ($O(T^2)$ redundant K/V projections). With a KV-Cache, each step appends 1 new K and V vector to the cache.",
    },
    {
        "l": 8,
        "q": "In HuggingFace `transformers`, what is the structural difference between `StaticLayer` (`StaticCache`) and `DynamicLayer` (`DynamicCache`)?",
        "o": [
            "`StaticLayer` pre-allocates a fixed `[B, H, max_len, D]` buffer and tracks length with `cumulative_length`; `DynamicLayer` grows `self.keys` with `torch.cat` so its length IS `self.keys.shape[-2]`",
            "`StaticLayer` only works on CPU while `DynamicLayer` only works on NVIDIA GPUs",
            "`DynamicLayer` pre-allocates a fixed buffer and `StaticLayer` uses `torch.cat` on every token",
            "There is no difference; they are two aliases for the exact same class",
        ],
        "w": "This storage-model difference is the exact root cause of TWO real bugs in your ledger: the `DynamicCache.reset()` bug (`transformers#48509/#48809`) AND the GPT-Neo `StaticCache` local-window bug (`gpt_neo_static_cache_fix.md`)!",
    },
    {
        "l": 8,
        "q": "In your `DynamicCache.reset()` finding (`transformers#48509` / `#48809`), why did inheriting `CacheLayerMixin.reset()` (`self.keys.zero_()`, `self.values.zero_()`) corrupt `DynamicLayer`?",
        "o": [
            "Zeroing `self.keys` in-place kept its old shape `[B, H, 6, D]` and kept `is_initialized=True`, so `get_seq_length()` still returned `6` and the next prompt `torch.cat`-ed onto 6 stale zero vectors!",
            "`self.keys.zero_()` threw a `TypeError` because PyTorch tensors do not have a `.zero_()` method",
            "Zeroing `self.keys` deleted the model's embedding weights on the GPU",
            "`DynamicLayer.reset()` was never called when `DynamicCache.reset()` ran",
        ],
        "w": "Because `DynamicLayer` has no separate length counter — its sequence length IS `self.keys.shape[-2]` (`0` only when `not self.is_initialized` or `self.keys is None`) — `.zero_()` left 6 phantom zero-tokens in the cache!",
    },
    {
        "l": 8,
        "q": "Even worse: in `QuantizedLayer` (subclass of `DynamicLayer`), what happened after the buggy `.reset()` left `self.is_initialized = True`?",
        "o": [
            "On the next `update()`, seeing `is_initialized == True` took the dequantize branch and RESURRECTED the pre-reset `_quantized_keys` history from the previous conversation!",
            "It crashed immediately with a division-by-zero error inside the quantizer",
            "It returned all `NaN` values because quantized integers cannot be zeroed",
            "It worked fine on `QuantizedLayer` and only failed on unquantized `DynamicLayer`",
        ],
        "w": "Your `d29d19a` verification on `transformers`: `QuantizedLayer` stored past tokens in `self._quantized_keys` (untouched by `self.keys.zero_()`!). Leaving `is_initialized = True` caused the next conversation to dequantize and leak the previous user's prompt history!",
    },
    {
        "l": 8,
        "q": "In your GPT-Neo `StaticCache` bug (`gpt_neo_static_cache_fix.md`), why did `key_length = key.size(-2)` break local sliding-window attention when using `StaticCache`?",
        "o": [
            "With `StaticCache`, `key.size(-2)` is always the full pre-allocated `max_cache_len` (e.g. 256), NOT the number of real tokens written so far, placing the causal mask window at the end of the empty buffer!",
            "`key.size(-2)` returns the head dimension instead of the sequence dimension in GPT-Neo",
            "`StaticCache` sorts the keys in alphabetical order after every token step",
            "Local attention does not use keys or values at all",
        ],
        "w": "In `DynamicCache`, `key.size(-2)` equals the number of tokens seen so far. In `StaticCache`, `key` is pre-allocated to `max_cache_len` (say 256) from step 1! Assuming `key.size(-2)` is the active token count slices the causal mask at `[255:256]` instead of `[0:1]`.",
    },
    {
        "l": 8,
        "q": "In your `TorchMetrics` `Running` finding (`20261019_053000_torchmetrics-running-state-dict-finding.md`), why did a rolling-window metric give wrong answers after `load_state_dict()`?",
        "o": [
            "It saved the ring-buffer tensor `val_list` in `state_dict`, but stored the write cursor / fill count as plain Python attributes that were NOT saved, overwriting the wrong slot on resume!",
            "`load_state_dict()` reversed the order of `val_list` every time a checkpoint was loaded",
            "The window size was multiplied by 2 every time `state_dict()` was called",
            "`Running` deleted `val_list` whenever `compute()` was called",
        ],
        "w": "A ring buffer (circular buffer) is defined by BOTH its backing array AND its write cursor (`window_pos` / `count`). Saving the array without the cursor means resuming after 7 updates resets the cursor to 0 and overwrites the wrong historical entries!",
    },
    {
        "l": 8,
        "q": "What is the general 'Storage-Model Mismatch' pattern that connects `DynamicCache.reset()`, `QuantizedLayer.reset()`, `GPT-Neo StaticCache`, and `TorchMetrics Running`?",
        "o": [
            "Every buffer has a representation invariant (how length, capacity, and valid slots are tracked); a bug happens whenever `reset()`, `update()`, or `state_dict()` updates the buffer but forgets its companion metadata (or vice versa)",
            "All four bugs were caused by using 16-bit floats instead of 64-bit floats",
            "All four bugs only happen when running on Windows without a GPU installed",
            "Every cache bug can be fixed by adding `torch.cuda.empty_cache()` at the end of `forward()`",
        ],
        "w": "That is the master systems lesson: a buffer is NEVER just a pointer or tensor — it is a coupled pair `(storage_buffer, validity_metadata)` such as `(keys, is_initialized)`, `(static_buf, cumulative_length)`, or `(ring_buf, write_cursor, count)`. Whenever an operation mutates one half without the other, silent corruption follows!",
    },
    {
        "l": 8,
        "q": "According to your Standing Protocols (`Rule M-1`, `Rule M-2`, `Rule F-1`), what must you do before calling any of these memory or cache hypotheses a real `T3`/`T4` finding?",
        "o": [
            "Validate the harness with a must-fail control, reproduce end-to-end through the real target API (RED before fix, GREEN after fix), and run the project's test suite",
            "Read the source code carefully by hand and file the issue immediately as a `T1` static read",
            "Ask an LLM whether the code looks buggy and trust its confidence score",
            "Only test the fixed version and assume the unfixed version would have failed",
        ],
        "w": "Rule M-1 & F-1: Reading source is a hypothesis (`T1`). Only an execution-verified reproduction through the real target API with a must-fail control (`T3`) plus verified fix and passing tests (`T4`) is a finding.",
    },
]

EXERCISES_5_8 = [
    # =========================================================================
    # LEVEL 5, EXERCISE 1: In-Place Counter Aliasing (GPT-Neo StaticCache Fix)
    # =========================================================================
    {
        "id": "inplace_counter_alias",
        "l": 5,
        "title": "5.1 In-Place Tensor Counter Aliasing (GPT-Neo StaticCache Window Fix)",
        "real_bug": "huggingface/transformers GPT-Neo StaticCache local attention window (gpt_neo_static_cache_fix.md)",
        "desc": (
            "In `StaticLayer` (HuggingFace `transformers`), `cumulative_length` is a 1-element tensor `[past_tokens]` that `static_cache_update` increments **in-place** (`cumulative_length[0] += new_tokens`).\n"
            "In GPT-Neo's `_attn`, the code snapshotted `past_len_ref = cumulative_length` BEFORE calling `static_cache_update`, and then after `update` built the token positions `[past_len_ref[0] + i for i in range(new_tokens)]`.\n"
            "Because `past_len_ref = cumulative_length` did NOT clone the tensor (both variables pointed to the exact same 1-element list/storage!), the in-place `+= new_tokens` inside `static_cache_update` mutated `past_len_ref[0]` too — shifting every position index forward by `new_tokens`!\n"
            "Fix `gpt_neo_local_window_positions(cumulative_length, new_tokens)` by snapshotting the integer value (or cloning `cumulative_length`) BEFORE `static_cache_update` mutates `cumulative_length` in-place, and return `[positions, cumulative_length[0]]`."
        ),
        "py_name": "gpt_neo_local_window_positions",
        "py_starter": (
            "def static_cache_update(cumulative_length, new_tokens):\n"
            "    # Mutates the 1-element tensor/list IN-PLACE (just like StaticLayer.update)\n"
            "    cumulative_length[0] += new_tokens\n\n"
            "def gpt_neo_local_window_positions(cumulative_length, new_tokens):\n"
            "    # BUG (GPT-Neo StaticCache): aliases cumulative_length without cloning before in-place update!\n"
            "    past_len_box = cumulative_length\n"
            "    static_cache_update(cumulative_length, new_tokens)\n"
            "    positions = [past_len_box[0] + i for i in range(new_tokens)]\n"
            "    return [positions, cumulative_length[0]]\n"
        ),
        "py_sol": (
            "def static_cache_update(cumulative_length, new_tokens):\n"
            "    # Mutates the 1-element tensor/list IN-PLACE (just like StaticLayer.update)\n"
            "    cumulative_length[0] += new_tokens\n\n"
            "def gpt_neo_local_window_positions(cumulative_length, new_tokens):\n"
            "    # FIX: clone/snapshot the pre-update value BEFORE static_cache_update mutates in-place!\n"
            "    past_len = cumulative_length[0]\n"
            "    static_cache_update(cumulative_length, new_tokens)\n"
            "    positions = [past_len + i for i in range(new_tokens)]\n"
            "    return [positions, cumulative_length[0]]\n"
        ),
        "cpp_name": "gpt_neo_local_window_positions",
        "cpp_starter": (
            "void static_cache_update(int* cumulative_length, int new_tokens) {\n"
            "    cumulative_length[0] += new_tokens; // in-place tensor mutation\n"
            "}\n\n"
            "void gpt_neo_local_window_positions(int init_cum_len, int new_tokens, int* out_positions, int* out_new_cum_len) {\n"
            "    int cumulative_length[1] = { init_cum_len };\n"
            "    // BUG: aliases pointer to cumulative_length instead of copying the value before in-place update!\n"
            "    int* past_len_ptr = cumulative_length;\n"
            "    static_cache_update(cumulative_length, new_tokens);\n"
            "    for (int i = 0; i < new_tokens; i++) {\n"
            "        out_positions[i] = past_len_ptr[0] + i;\n"
            "    }\n"
            "    *out_new_cum_len = cumulative_length[0];\n"
            "}\n"
        ),
        "cpp_sol": (
            "void static_cache_update(int* cumulative_length, int new_tokens) {\n"
            "    cumulative_length[0] += new_tokens; // in-place tensor mutation\n"
            "}\n\n"
            "void gpt_neo_local_window_positions(int init_cum_len, int new_tokens, int* out_positions, int* out_new_cum_len) {\n"
            "    int cumulative_length[1] = { init_cum_len };\n"
            "    int past_len = cumulative_length[0];\n"
            "    static_cache_update(cumulative_length, new_tokens);\n"
            "    for (int i = 0; i < new_tokens; i++) {\n"
            "        out_positions[i] = past_len + i;\n"
            "    }\n"
            "    *out_new_cum_len = cumulative_length[0];\n"
            "}\n"
        ),
        "hint": "Copy the value `past_len = cumulative_length[0]` (or `past_len_box = list(cumulative_length)` / `.clone()`) BEFORE calling `static_cache_update(cumulative_length, new_tokens)`.",
        "tests": [
            {"a": "[[0], 4]", "e": "[[0, 1, 2, 3], 4]"},
            {"a": "[[4], 2]", "e": "[[4, 5], 6]"},
            {"a": "[[10], 1]", "e": "[[10], 11]"},
            {"a": "[[256], 3]", "e": "[[256, 257, 258], 259]", "h": 1},
        ],
        "v": [
            ["cumulative_length (list[int] / int*)", "1-element tensor `[count]` owned by `StaticLayer` that tracks how many valid tokens are in the pre-allocated static buffer."],
            ["new_tokens (int)", "Number of query/key tokens arriving in the current forward pass (`query_length`)."],
            ["past_len_box (list[int]) / past_len_ptr (int*)", "BUGGY alias pointing to the exact same memory box as `cumulative_length`."],
            ["past_len (int)", "FIXED independent snapshot (`.clone()`) of the token count taken BEFORE `static_cache_update` mutates `cumulative_length[0]` in place."],
            ["positions (list[int] / int*)", "The absolute token indices `[past_len, past_len + 1, ...]` used to slice the local attention causal mask."],
        ],
        "lines_py": [
            ["def static_cache_update(cumulative_length, new_tokens):", "Models `StaticLayer.update()`, which writes new K/V vectors into the fixed buffer."],
            ["    cumulative_length[0] += new_tokens", "Advances the tensor counter IN-PLACE (`self.cumulative_length += q_len`)."],
            ["def gpt_neo_local_window_positions(cumulative_length, new_tokens):", "Computes the causal mask row indices for the current chunk of `new_tokens`."],
            ["    past_len = cumulative_length[0]", "THE FIX (`gpt_neo_static_cache_fix.md`): Clones/snapshots the integer value BEFORE `static_cache_update` mutates `cumulative_length` in-place!"],
            ["    static_cache_update(cumulative_length, new_tokens)", "Mutates `cumulative_length[0]` in-place from `past_len` to `past_len + new_tokens`."],
            ["    positions = [past_len + i for i in range(new_tokens)]", "Uses the pre-update `past_len` so the first prompt of 4 tokens gets positions `[0, 1, 2, 3]` instead of `[4, 5, 6, 7]`!"],
            ["    return [positions, cumulative_length[0]]", "Returns both the local window positions and the updated `cumulative_length[0]`."],
        ],
        "lines_cpp": [
            ["int past_len = cumulative_length[0];", "THE FIX: Copies the scalar value out of `cumulative_length[0]` before passing the pointer to `static_cache_update`."],
            ["static_cache_update(cumulative_length, new_tokens);", "Increments `cumulative_length[0]` in-place via pointer dereference."],
            ["out_positions[i] = past_len + i;", "Computes each token's true position from the snapshotted `past_len`."],
        ],
    },

    # =========================================================================
    # LEVEL 5, EXERCISE 2: Unregistered State Lost Across state_dict Resume
    # =========================================================================
    {
        "id": "persistent_buffer_state",
        "l": 5,
        "title": "5.2 Stateful Module Buffers Missing From state_dict() Resume (Kornia ADA & TorchMetrics)",
        "real_bug": "kornia/kornia AdaptiveDiscriminatorAugmentation (p reset 0.8 -> 1e-5) & Lightning-AI/torchmetrics WrapperMetric.persistent",
        "desc": (
            "In `kornia`'s `AdaptiveDiscriminatorAugmentation` (`ADA`), the augmentation probability `p` starts at `init_p` (`0.0`) and adapts every `update_interval` steps by `+delta` (clamped to `[0.0, 1.0]`), while `step_count` counts total steps.\n"
            "When training is interrupted after `steps_before_save` steps, the trainer saves `state = get_state_dict(...)` and loads it into a fresh module before running `steps_after_resume` more steps.\n"
            "The buggy starter omits `p` and `step_count` from `state_dict` (leaving it empty `{}`), so resuming from a checkpoint silently resets `p` back to `init_p` and loses the remainder of `step_count % update_interval`!\n"
            "Fix `simulate_ada_resume(init_p, delta, update_interval, steps_before_save, steps_after_resume)` so `state_dict` saves and restores both `'p'` and `'step_count'`, making the resumed run produce the exact same `[round(p, 6), step_count]` as an uninterrupted run of `steps_before_save + steps_after_resume` steps!"
        ),
        "py_name": "simulate_ada_resume",
        "py_starter": (
            "def step_ada(state, delta, update_interval):\n"
            "    state['step_count'] += 1\n"
            "    if state['step_count'] % update_interval == 0:\n"
            "        state['p'] = min(1.0, max(0.0, state['p'] + delta))\n\n"
            "def simulate_ada_resume(init_p, delta, update_interval, steps_before_save, steps_after_resume):\n"
            "    mod1 = {'p': init_p, 'step_count': 0}\n"
            "    for _ in range(steps_before_save):\n"
            "        step_ada(mod1, delta, update_interval)\n"
            "    # BUG (Kornia ADA): state_dict forgets to persist 'p' and 'step_count'!\n"
            "    checkpoint = {}\n"
            "    # Fresh module created on resume:\n"
            "    mod2 = {'p': init_p, 'step_count': 0}\n"
            "    mod2.update(checkpoint)\n"
            "    for _ in range(steps_after_resume):\n"
            "        step_ada(mod2, delta, update_interval)\n"
            "    return [round(mod2['p'], 6), mod2['step_count']]\n"
        ),
        "py_sol": (
            "def step_ada(state, delta, update_interval):\n"
            "    state['step_count'] += 1\n"
            "    if state['step_count'] % update_interval == 0:\n"
            "        state['p'] = min(1.0, max(0.0, state['p'] + delta))\n\n"
            "def simulate_ada_resume(init_p, delta, update_interval, steps_before_save, steps_after_resume):\n"
            "    mod1 = {'p': init_p, 'step_count': 0}\n"
            "    for _ in range(steps_before_save):\n"
            "        step_ada(mod1, delta, update_interval)\n"
            "    checkpoint = {'p': mod1['p'], 'step_count': mod1['step_count']}\n"
            "    mod2 = {'p': init_p, 'step_count': 0}\n"
            "    mod2.update(checkpoint)\n"
            "    for _ in range(steps_after_resume):\n"
            "        step_ada(mod2, delta, update_interval)\n"
            "    return [round(mod2['p'], 6), mod2['step_count']]\n"
        ),
        "cpp_name": "simulate_ada_resume",
        "cpp_starter": (
            "void step_ada(double* p, int* step_count, double delta, int update_interval) {\n"
            "    *step_count += 1;\n"
            "    if ((*step_count % update_interval) == 0) {\n"
            "        double np = *p + delta;\n"
            "        if (np > 1.0) np = 1.0;\n"
            "        if (np < 0.0) np = 0.0;\n"
            "        *p = np;\n"
            "    }\n"
            "}\n\n"
            "void simulate_ada_resume(double init_p, double delta, int update_interval, int steps_before_save, int steps_after_resume, double* out_p, int* out_steps) {\n"
            "    double p1 = init_p;\n"
            "    int sc1 = 0;\n"
            "    for (int i = 0; i < steps_before_save; i++) {\n"
            "        step_ada(&p1, &sc1, delta, update_interval);\n"
            "    }\n"
            "    // Fresh module on resume:\n"
            "    double p2 = init_p;\n"
            "    int sc2 = 0;\n"
            "    // BUG (Kornia ADA): p1 and sc1 were not registered in state_dict, so p2 and sc2 reset to init_p and 0!\n"
            "    for (int j = 0; j < steps_after_resume; j++) {\n"
            "        step_ada(&p2, &sc2, delta, update_interval);\n"
            "    }\n"
            "    *out_p = p2;\n"
            "    *out_steps = sc2;\n"
            "}\n"
        ),
        "cpp_sol": (
            "void step_ada(double* p, int* step_count, double delta, int update_interval) {\n"
            "    *step_count += 1;\n"
            "    if ((*step_count % update_interval) == 0) {\n"
            "        double np = *p + delta;\n"
            "        if (np > 1.0) np = 1.0;\n"
            "        if (np < 0.0) np = 0.0;\n"
            "        *p = np;\n"
            "    }\n"
            "}\n\n"
            "void simulate_ada_resume(double init_p, double delta, int update_interval, int steps_before_save, int steps_after_resume, double* out_p, int* out_steps) {\n"
            "    double p1 = init_p;\n"
            "    int sc1 = 0;\n"
            "    for (int i = 0; i < steps_before_save; i++) {\n"
            "        step_ada(&p1, &sc1, delta, update_interval);\n"
            "    }\n"
            "    double p2 = p1;\n"
            "    int sc2 = sc1;\n"
            "    for (int j = 0; j < steps_after_resume; j++) {\n"
            "        step_ada(&p2, &sc2, delta, update_interval);\n"
            "    }\n"
            "    *out_p = p2;\n"
            "    *out_steps = sc2;\n"
            "}\n"
        ),
        "hint": "Save both `'p'` and `'step_count'` into `checkpoint = {'p': mod1['p'], 'step_count': mod1['step_count']}` (and in C++, initialize `p2 = p1; sc2 = sc1;` from the saved checkpoint).",
        "tests": [
            {"a": "[0.0, 0.1, 4, 8, 4]", "e": "[0.3, 12]"},
            {"a": "[0.0, 0.2, 4, 3, 1]", "e": "[0.2, 4]"},
            {"a": "[0.5, 0.25, 2, 4, 2]", "e": "[1.0, 6]"},
            {"a": "[0.1, 0.05, 5, 7, 3]", "e": "[0.2, 10]", "h": 1},
        ],
        "v": [
            ["init_p (float / double)", "Initial augmentation probability at step 0 (e.g. `1e-5` or `0.0`)."],
            ["delta (float / double)", "Amount added to `p` every `update_interval` steps."],
            ["update_interval (int)", "Number of steps between probability updates."],
            ["steps_before_save (int)", "Training steps run on `mod1` before saving `state_dict`."],
            ["steps_after_resume (int)", "Training steps run on `mod2` after loading `state_dict`."],
            ["checkpoint (dict)", "Serialized `state_dict` dictionary. Must contain both `'p'` and `'step_count'` so resume is lossless!"],
        ],
        "lines_py": [
            ["def step_ada(state, delta, update_interval):", "Advances one step of `AdaptiveDiscriminatorAugmentation`."],
            ["    state['step_count'] += 1", "Increments the step counter."],
            ["    if state['step_count'] % update_interval == 0:", "Triggers probability adjustment every `update_interval` steps."],
            ["        state['p'] = min(1.0, max(0.0, state['p'] + delta))", "Updates and clamps the augmentation probability `p` in `[0.0, 1.0]`."],
            ["    checkpoint = {'p': mod1['p'], 'step_count': mod1['step_count']}", "THE FIX (`Kornia ADA` finding): Registers both `p` and `step_count` as persistent state in `state_dict`!"],
            ["    mod2.update(checkpoint)", "Restores `p` and `step_count` into the resumed module so step `3 + 1 = 4` triggers the interval update seamlessly."],
        ],
        "lines_cpp": [
            ["double p2 = p1; int sc2 = sc1;", "THE FIX: Restores both the probability buffer `p` and counter `step_count` from the checkpoint on resume."],
        ],
    },

    # =========================================================================
    # LEVEL 6, EXERCISE 1: CUDACachingAllocator Best-Fit Split & OOM Eviction
    # =========================================================================
    {
        "id": "cuda_pool_alloc",
        "l": 6,
        "title": "6.1 CUDACachingAllocator Best-Fit Block Splitting & OOM Cache Eviction",
        "real_bug": "PyTorch c10/cuda/CUDACachingAllocator.cpp (BlockPool best-fit search, block splitting, and release_cached_blocks retry)",
        "desc": (
            "Model PyTorch's `CUDACachingAllocator` servicing a single request for `req_bytes` (already rounded to 512B alignment):\n"
            "You are given `free_blocks` (a list of idle cached block sizes currently in the pool), `active_bytes` (VRAM held by live tensors), `req_bytes` (size needed), and `max_vram` (total physical GPU memory).\n"
            "note that `reserved_bytes = active_bytes + sum(free_blocks)`.\n"
            "Implement the 3-step `CUDACachingAllocator` algorithm in `cuda_caching_alloc(free_blocks, active_bytes, req_bytes, max_vram)`:\n"
            "1. **Best-fit pool hit**: If one or more blocks in `free_blocks` have size `>= req_bytes`, pick the **smallest** such block (`best`). Remove `best` from `free_blocks`. If `best > req_bytes`, **split** it by appending the remainder `best - req_bytes` to `free_blocks`. Add `req_bytes` to `active_bytes`.\n"
            "2. **Driver `cudaMalloc` (pool miss)**: If no block in `free_blocks` is `>= req_bytes`, check if `active_bytes + sum(free_blocks) + req_bytes <= max_vram`. If yes, `cudaMalloc` succeeds directly: add `req_bytes` to `active_bytes` (leaving `free_blocks` untouched).\n"
            "3. **OOM Retry (`release_cached_blocks`)**: Otherwise, evict all cached free blocks (`free_blocks = []`, returning their VRAM to the driver) and check if `active_bytes + req_bytes <= max_vram`. If yes, `cudaMalloc` now succeeds (`active_bytes += req_bytes`). If even that exceeds `max_vram`, return `[-1, -1, -1]` (true OOM!).\n"
            "On success (steps 1, 2, or 3), return `[active_bytes, active_bytes + sum(free_blocks), len(free_blocks)]` (which is `[allocated_bytes, reserved_bytes, num_free_blocks]`).\n"
            "(The buggy starter never splits oversized free blocks and never evicts cached free blocks when `cudaMalloc` hits `max_vram`!)"
        ),
        "py_name": "cuda_caching_alloc",
        "py_starter": (
            "def cuda_caching_alloc(free_blocks, active_bytes, req_bytes, max_vram):\n"
            "    pool = list(free_blocks)\n"
            "    # Step 1: look for best-fit block >= req_bytes\n"
            "    candidates = [b for b in pool if b >= req_bytes]\n"
            "    if len(candidates) > 0:\n"
            "        best = min(candidates)\n"
            "        pool.remove(best)\n"
            "        # BUG 1: forgets to split `best` when best > req_bytes (wastes best - req_bytes)!\n"
            "        active_bytes += best\n"
            "        return [active_bytes, active_bytes + sum(pool), len(pool)]\n"
            "    # Step 2: try raw cudaMalloc without evicting pool\n"
            "    if active_bytes + sum(pool) + req_bytes <= max_vram:\n"
            "        active_bytes += req_bytes\n"
            "        return [active_bytes, active_bytes + sum(pool), len(pool)]\n"
            "    # BUG 2: forgets Step 3 (evict `pool = []` via release_cached_blocks and retry before OOM)!\n"
            "    return [-1, -1, -1]\n"
        ),
        "py_sol": (
            "def cuda_caching_alloc(free_blocks, active_bytes, req_bytes, max_vram):\n"
            "    pool = list(free_blocks)\n"
            "    # Step 1: look for best-fit block >= req_bytes\n"
            "    candidates = [b for b in pool if b >= req_bytes]\n"
            "    if len(candidates) > 0:\n"
            "        best = min(candidates)\n"
            "        pool.remove(best)\n"
            "        if best > req_bytes:\n"
            "            pool.append(best - req_bytes)\n"
            "        active_bytes += req_bytes\n"
            "        return [active_bytes, active_bytes + sum(pool), len(pool)]\n"
            "    # Step 2: try raw cudaMalloc without evicting pool\n"
            "    if active_bytes + sum(pool) + req_bytes <= max_vram:\n"
            "        active_bytes += req_bytes\n"
            "        return [active_bytes, active_bytes + sum(pool), len(pool)]\n"
            "    # Step 3: OOM retry — release_cached_blocks (empty_cache) and retry cudaMalloc\n"
            "    pool = []\n"
            "    if active_bytes + req_bytes <= max_vram:\n"
            "        active_bytes += req_bytes\n"
            "        return [active_bytes, active_bytes, 0]\n"
            "    return [-1, -1, -1]\n"
        ),
        "cpp_name": "cuda_caching_alloc",
        "cpp_starter": (
            "void cuda_caching_alloc(const int* free_blocks, int n_free, int active_bytes, int req_bytes, int max_vram, int* out_res) {\n"
            "    int pool_sum = 0;\n"
            "    int best_idx = -1;\n"
            "    for (int i = 0; i < n_free; i++) {\n"
            "        pool_sum += free_blocks[i];\n"
            "        if (free_blocks[i] >= req_bytes) {\n"
            "            if (best_idx == -1 || free_blocks[i] < free_blocks[best_idx]) {\n"
            "                best_idx = i;\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "    if (best_idx != -1) {\n"
            "        int best = free_blocks[best_idx];\n"
            "        // BUG 1: does not split `best` when best > req_bytes!\n"
            "        out_res[0] = active_bytes + best;\n"
            "        out_res[1] = active_bytes + pool_sum;\n"
            "        out_res[2] = n_free - 1;\n"
            "        return;\n"
            "    }\n"
            "    if (active_bytes + pool_sum + req_bytes <= max_vram) {\n"
            "        out_res[0] = active_bytes + req_bytes;\n"
            "        out_res[1] = active_bytes + pool_sum + req_bytes;\n"
            "        out_res[2] = n_free;\n"
            "        return;\n"
            "    }\n"
            "    // BUG 2: forgets OOM retry after releasing cached free blocks!\n"
            "    out_res[0] = -1; out_res[1] = -1; out_res[2] = -1;\n"
            "}\n"
        ),
        "cpp_sol": (
            "void cuda_caching_alloc(const int* free_blocks, int n_free, int active_bytes, int req_bytes, int max_vram, int* out_res) {\n"
            "    int pool_sum = 0;\n"
            "    int best_idx = -1;\n"
            "    for (int i = 0; i < n_free; i++) {\n"
            "        pool_sum += free_blocks[i];\n"
            "        if (free_blocks[i] >= req_bytes) {\n"
            "            if (best_idx == -1 || free_blocks[i] < free_blocks[best_idx]) {\n"
            "                best_idx = i;\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "    if (best_idx != -1) {\n"
            "        int best = free_blocks[best_idx];\n"
            "        int new_n_free = (best > req_bytes) ? n_free : (n_free - 1);\n"
            "        out_res[0] = active_bytes + req_bytes;\n"
            "        out_res[1] = active_bytes + pool_sum;\n"
            "        out_res[2] = new_n_free;\n"
            "        return;\n"
            "    }\n"
            "    if (active_bytes + pool_sum + req_bytes <= max_vram) {\n"
            "        out_res[0] = active_bytes + req_bytes;\n"
            "        out_res[1] = active_bytes + pool_sum + req_bytes;\n"
            "        out_res[2] = n_free;\n"
            "        return;\n"
            "    }\n"
            "    if (active_bytes + req_bytes <= max_vram) {\n"
            "        out_res[0] = active_bytes + req_bytes;\n"
            "        out_res[1] = active_bytes + req_bytes;\n"
            "        out_res[2] = 0;\n"
            "        return;\n"
            "    }\n"
            "    out_res[0] = -1; out_res[1] = -1; out_res[2] = -1;\n"
            "}\n"
        ),
        "hint": "Fix 1 (splitting): in Step 1, if `best > req_bytes`, append `best - req_bytes` back to `pool`, and add `req_bytes` (not `best`) to `active_bytes`. Fix 2 (OOM retry): at the end, clear `pool = []` and check `if active_bytes + req_bytes <= max_vram:`; if so, add `req_bytes` to `active_bytes` and return `[active_bytes, active_bytes, 0]`.",
        "tests": [
            {"a": "[[10, 4], 20, 6, 100]", "e": "[26, 34, 2]"},
            {"a": "[[4, 4, 4], 20, 8, 36]", "e": "[28, 28, 0]"},
            {"a": "[[4, 4], 20, 8, 50]", "e": "[28, 36, 2]"},
            {"a": "[[8], 10, 8, 50]", "e": "[18, 18, 0]", "h": 1},
            {"a": "[[4, 4], 30, 12, 40]", "e": "[-1, -1, -1]", "h": 1},
        ],
        "v": [
            ["free_blocks / pool (list[int])", "Sizes of cached idle VRAM blocks currently held in `CUDACachingAllocator`'s free list."],
            ["active_bytes (int)", "VRAM currently occupied by live tensors (`torch.cuda.memory_allocated()`)."],
            ["req_bytes (int)", "Size of the new tensor allocation request."],
            ["max_vram (int)", "Total physical VRAM available on the GPU."],
            ["candidates (list[int])", "Cached free blocks large enough (`>= req_bytes`) to satisfy the request without calling `cudaMalloc`."],
            ["best (int)", "Smallest cached free block `>= req_bytes` (best-fit policy minimizes fragmentation)."],
        ],
        "lines_py": [
            ["    candidates = [b for b in pool if b >= req_bytes]", "Finds all cached free blocks in the pool that can fit `req_bytes`."],
            ["    if len(candidates) > 0:", "Pool hit! We can satisfy the allocation without calling the slow CUDA driver `cudaMalloc`."],
            ["        best = min(candidates)", "Best-fit lookup (`BlockPool::blocks.lower_bound` in C++): picks the smallest block `>= req_bytes`."],
            ["        pool.remove(best)", "Removes `best` from the free pool."],
            ["        if best > req_bytes: pool.append(best - req_bytes)", "THE FIX #1 (Block splitting): Splits the oversized block and puts the `best - req_bytes` remainder back into the free pool!"],
            ["        active_bytes += req_bytes", "Increases `memory_allocated()` by `req_bytes` (while total `memory_reserved()` stays unchanged!)."],
            ["    if active_bytes + sum(pool) + req_bytes <= max_vram:", "Pool miss: checks if the GPU has enough unreserved VRAM for a fresh `cudaMalloc(req_bytes)`."],
            ["    pool = []", "THE FIX #2 (`release_cached_blocks()`): When fragmented `pool` blocks prevent `cudaMalloc`, evicts idle cached blocks back to the CUDA driver and retries!"],
            ["    if active_bytes + req_bytes <= max_vram:", "Checks if `cudaMalloc` succeeds after releasing cached blocks."],
        ],
        "lines_cpp": [
            ["int new_n_free = (best > req_bytes) ? n_free : (n_free - 1);", "THE FIX #1: When `best > req_bytes`, splitting replaces `best` with `best - req_bytes` so `n_free` stays `n_free`, and `active_bytes` grows by `req_bytes`."],
            ["if (active_bytes + req_bytes <= max_vram) {", "THE FIX #2: After `release_cached_blocks()` frees all `pool_sum` idle bytes, retries `cudaMalloc(req_bytes)`."],
        ],
    },

    # =========================================================================
    # LEVEL 6, EXERCISE 2: Autograd Computation Graph Leak in Training Loop
    # =========================================================================
    {
        "id": "autograd_graph_leak",
        "l": 6,
        "title": "6.2 Autograd Activation Graph Pinning in Metrics History (`loss.detach().item()`)",
        "real_bug": "Classic PyTorch GPU OOM: appending live `loss` Tensor with `grad_fn` to `history` pins forward activations every step",
        "desc": (
            "In PyTorch, each training step produces a `loss` Tensor whose `grad_fn` holds pointers to all intermediate forward activations (`act_bytes` MB of VRAM) needed for `loss.backward()`. "
            "After `backward()` runs, those activations are freed ONLY when no live reference to the `loss` graph node remains!\n"
            "Look at `run_training_epoch_vram(step_losses, act_bytes_per_step)`:\n"
            "- Each step creates a loss node `{'val': val, 'act_bytes': act_bytes_per_step}`.\n"
            "- The buggy starter appends the **entire graph node** `loss_node` into `history`, pinning `act_bytes_per_step` in VRAM on every single step (`leaked_vram` grows by `act_bytes_per_step * num_steps`)!\n"
            "Fix `run_training_epoch_vram` so that `history` appends ONLY the detached scalar value `loss_node['val']` (modeling `loss.detach().item()`), and `leaked_vram` only sums `act_bytes` for objects in `history` that are still graph nodes (`isinstance(x, dict)`), returning `[round(avg_loss, 6), leaked_vram]`."
        ),
        "py_name": "run_training_epoch_vram",
        "py_starter": (
            "def run_training_epoch_vram(step_losses, act_bytes_per_step):\n"
            "    history = []\n"
            "    for val in step_losses:\n"
            "        loss_node = {'val': val, 'act_bytes': act_bytes_per_step}\n"
            "        # BUG: appending live loss_node keeps grad_fn and act_bytes pinned in VRAM forever!\n"
            "        history.append(loss_node)\n"
            "    avg_loss = sum(x['val'] if isinstance(x, dict) else x for x in history) / len(history)\n"
            "    leaked_vram = sum(x['act_bytes'] for x in history if isinstance(x, dict))\n"
            "    return [round(avg_loss, 6), leaked_vram]\n"
        ),
        "py_sol": (
            "def run_training_epoch_vram(step_losses, act_bytes_per_step):\n"
            "    history = []\n"
            "    for val in step_losses:\n"
            "        loss_node = {'val': val, 'act_bytes': act_bytes_per_step}\n"
            "        # FIX: detach scalar value (loss.detach().item()) so the autograd graph node is freed!\n"
            "        history.append(loss_node['val'])\n"
            "    avg_loss = sum(x['val'] if isinstance(x, dict) else x for x in history) / len(history)\n"
            "    leaked_vram = sum(x['act_bytes'] for x in history if isinstance(x, dict))\n"
            "    return [round(avg_loss, 6), leaked_vram]\n"
        ),
        "cpp_name": "run_training_epoch_vram",
        "cpp_starter": (
            "void run_training_epoch_vram(const double* step_losses, int n_steps, int act_bytes_per_step, double* out_avg_loss, int* out_leaked_vram) {\n"
            "    double total_loss = 0.0;\n"
            "    int leaked_vram = 0;\n"
            "    for (int i = 0; i < n_steps; i++) {\n"
            "        double loss_val = step_losses[i];\n"
            "        int keep_grad_fn = 1; // BUG: stores live Autograd Node in history instead of detached scalar!\n"
            "        total_loss += loss_val;\n"
            "        if (keep_grad_fn) {\n"
            "            leaked_vram += act_bytes_per_step;\n"
            "        }\n"
            "    }\n"
            "    *out_avg_loss = total_loss / n_steps;\n"
            "    *out_leaked_vram = leaked_vram;\n"
            "}\n"
        ),
        "cpp_sol": (
            "void run_training_epoch_vram(const double* step_losses, int n_steps, int act_bytes_per_step, double* out_avg_loss, int* out_leaked_vram) {\n"
            "    double total_loss = 0.0;\n"
            "    int leaked_vram = 0;\n"
            "    for (int i = 0; i < n_steps; i++) {\n"
            "        double loss_val = step_losses[i];\n"
            "        int keep_grad_fn = 0; // FIX: loss.detach().item() drops grad_fn so activations free each step!\n"
            "        total_loss += loss_val;\n"
            "        if (keep_grad_fn) {\n"
            "            leaked_vram += act_bytes_per_step;\n"
            "        }\n"
            "    }\n"
            "    *out_avg_loss = total_loss / n_steps;\n"
            "    *out_leaked_vram = leaked_vram;\n"
            "}\n"
        ),
        "hint": "Change `history.append(loss_node)` to `history.append(loss_node['val'])` (and in C++, set `keep_grad_fn = 0;`) so only the detached scalar number is kept in `history`.",
        "tests": [
            {"a": "[[2.0, 1.5, 1.0, 0.5], 256]", "e": "[1.25, 0]"},
            {"a": "[[3.0, 3.0], 512]", "e": "[3.0, 0]"},
            {"a": "[[1.2, 0.8, 0.4], 1024]", "e": "[0.8, 0]", "h": 1},
        ],
        "v": [
            ["step_losses (list[float] / const double*)", "Scalar loss values produced on each training step in the epoch."],
            ["act_bytes_per_step (int)", "VRAM (in MB) pinned by the autograd computation graph (`grad_fn` saved activations) of one step."],
            ["loss_node (dict)", "Models a live PyTorch `Tensor` carrying both its scalar `.item()` (`'val'`) and its `grad_fn` graph (`'act_bytes'`)."],
            ["history (list)", "Epoch logging list. If it stores `loss_node`, every step's graph stays alive in VRAM; if it stores `loss_node['val']`, `loss_node` is garbage-collected at the end of each step!"],
            ["leaked_vram (int)", "Total VRAM still pinned by `history` at the end of the epoch (`0` when properly detached)."],
        ],
        "lines_py": [
            ["    for val in step_losses:", "Iterates over each batch in the training epoch."],
            ["        loss_node = {'val': val, 'act_bytes': act_bytes_per_step}", "Models `loss = criterion(model(x), y)`, which attaches an autograd graph holding `act_bytes_per_step` of activations."],
            ["        history.append(loss_node['val'])", "THE FIX: `history.append(loss.detach().item())` extracts the plain Python float so `loss_node` goes out of scope and frees its GPU activations immediately!"],
            ["    leaked_vram = sum(x['act_bytes'] for x in history if isinstance(x, dict))", "Verifies that zero live autograd graph nodes remain pinned in `history` (`leaked_vram == 0`)."],
        ],
        "lines_cpp": [
            ["int keep_grad_fn = 0;", "THE FIX: Detaches `grad_fn` (`tensor.detach().item<double>()`) before logging the scalar metric."],
        ],
    },

    # =========================================================================
    # LEVEL 7, EXERCISE 1: Cross-Stream Use-After-Free (record_stream Fix)
    # =========================================================================
    {
        "id": "stream_record_lifetime",
        "l": 7,
        "title": "7.1 Cross-Stream GPU Use-After-Free & record_stream() Event Fence",
        "real_bug": "PyTorch CUDACachingAllocator cross-stream race when tensor allocated on stream A is used on stream B without record_stream()",
        "desc": (
            "Suppose a GPU tensor block is allocated on `alloc_stream` (Stream 1), passed to a long-running async kernel on `compute_stream` (Stream 2) that finishes at GPU timestamp `stream2_done_tick`, and then `del tensor` runs on the CPU at timestamp `cpu_free_tick`.\n"
            "Next, at timestamp `next_alloc_tick` on `alloc_stream` (Stream 1), a new tensor requests a block of the same size!\n"
            "- **Without `record_stream(compute_stream)`** (the buggy starter): `CUDACachingAllocator` only checks `alloc_stream`, thinks the block is free at `cpu_free_tick`, and hands the block to the new tensor at `next_alloc_tick` EVEN WHEN `next_alloc_tick < stream2_done_tick` — overwriting the buffer while Stream 2's kernel is still running (silent GPU Use-After-Free)!\n"
            "- **With `record_stream(compute_stream)`**: The block is fenced until `safe_reuse_tick = max(cpu_free_tick, stream2_done_tick)`. At `next_alloc_tick`, the block can be reused (`reused_block = True`) ONLY if `next_alloc_tick >= safe_reuse_tick`; otherwise the allocator MUST allocate a fresh block (`reused_block = False`) so Stream 2 is never corrupted!\n"
            "Fix `can_reuse_stream_block(cpu_free_tick, stream2_done_tick, next_alloc_tick)` to return `[safe_reuse_tick, reused_block]`."
        ),
        "py_name": "can_reuse_stream_block",
        "py_starter": (
            "def can_reuse_stream_block(cpu_free_tick, stream2_done_tick, next_alloc_tick):\n"
            "    # BUG: forgets record_stream(stream2), so it ignores stream2_done_tick!\n"
            "    safe_reuse_tick = cpu_free_tick\n"
            "    reused_block = (next_alloc_tick >= safe_reuse_tick)\n"
            "    return [safe_reuse_tick, reused_block]\n"
        ),
        "py_sol": (
            "def can_reuse_stream_block(cpu_free_tick, stream2_done_tick, next_alloc_tick):\n"
            "    # FIX (record_stream): block cannot be reused until BOTH cpu_free_tick AND stream2_done_tick have passed!\n"
            "    safe_reuse_tick = max(cpu_free_tick, stream2_done_tick)\n"
            "    reused_block = (next_alloc_tick >= safe_reuse_tick)\n"
            "    return [safe_reuse_tick, reused_block]\n"
        ),
        "cpp_name": "can_reuse_stream_block",
        "cpp_starter": (
            "void can_reuse_stream_block(int cpu_free_tick, int stream2_done_tick, int next_alloc_tick, int* out_res) {\n"
            "    // BUG: ignores stream2_done_tick (missing record_stream)!\n"
            "    int safe_reuse_tick = cpu_free_tick;\n"
            "    int reused_block = (next_alloc_tick >= safe_reuse_tick) ? 1 : 0;\n"
            "    out_res[0] = safe_reuse_tick;\n"
            "    out_res[1] = reused_block;\n"
            "}\n"
        ),
        "cpp_sol": (
            "void can_reuse_stream_block(int cpu_free_tick, int stream2_done_tick, int next_alloc_tick, int* out_res) {\n"
            "    int safe_reuse_tick = (cpu_free_tick > stream2_done_tick) ? cpu_free_tick : stream2_done_tick;\n"
            "    int reused_block = (next_alloc_tick >= safe_reuse_tick) ? 1 : 0;\n"
            "    out_res[0] = safe_reuse_tick;\n"
            "    out_res[1] = reused_block;\n"
            "}\n"
        ),
        "hint": "Set `safe_reuse_tick = max(cpu_free_tick, stream2_done_tick)` (or the ternary max in C++) so the block is not reused while `next_alloc_tick < stream2_done_tick`.",
        "tests": [
            {"a": "[10, 50, 25]", "e": "[50, False]"},
            {"a": "[10, 50, 50]", "e": "[50, True]"},
            {"a": "[40, 20, 30]", "e": "[40, False]"},
            {"a": "[40, 20, 45]", "e": "[40, True]", "h": 1},
        ],
        "v": [
            ["cpu_free_tick (int)", "Timestamp when Python `del x` returned the block to `CUDACachingAllocator` on Stream 1."],
            ["stream2_done_tick (int)", "Timestamp when the CUDA event recorded by `x.record_stream(stream_2)` actually finishes executing on the GPU."],
            ["next_alloc_tick (int)", "Timestamp when a subsequent allocation on Stream 1 asks `CUDACachingAllocator` for a block."],
            ["safe_reuse_tick (int)", "Earliest timestamp at which the block's VRAM is guaranteed not to be read/written by either stream (`max(cpu_free_tick, stream2_done_tick)`)."],
            ["reused_block (bool / int)", "`True` (`1`) if `next_alloc_tick >= safe_reuse_tick` (safe to reuse); `False` (`0`) if Stream 2 is still busy and a fresh block must be used instead."],
        ],
        "lines_py": [
            ["def can_reuse_stream_block(cpu_free_tick, stream2_done_tick, next_alloc_tick):", "Determines whether `CUDACachingAllocator` can safely hand out a freed block at `next_alloc_tick`."],
            ["    safe_reuse_tick = max(cpu_free_tick, stream2_done_tick)", "THE FIX (`x.record_stream(stream2)`): Requires both the CPU `free` AND Stream 2's recorded CUDA event (`cudaEventQuery`) to complete before reuse."],
            ["    reused_block = (next_alloc_tick >= safe_reuse_tick)", "Only reuses the block if `next_alloc_tick` is at or after `safe_reuse_tick`, preventing cross-stream memory corruption."],
            ["    return [safe_reuse_tick, reused_block]", "Returns `[safe_reuse_tick, reused_block]`."],
        ],
        "lines_cpp": [
            ["int safe_reuse_tick = (cpu_free_tick > stream2_done_tick) ? cpu_free_tick : stream2_done_tick;", "THE FIX: Models `process_events()` in `CUDACachingAllocator.cpp` waiting until the recorded CUDA event on `stream2` completes."],
        ],
    },

    # =========================================================================
    # LEVEL 7, EXERCISE 2: Pinned Memory (mlock) Early-Return Heap Cleanup
    # =========================================================================
    {
        "id": "pinned_alloc_cleanup",
        "l": 7,
        "title": "7.2 Page-Locked (Pinned) Host Allocation & mlock Failure Cleanup (DeepSpeed page_alloc.cpp)",
        "real_bug": "deepspeedai/DeepSpeed csrc/pin_memory/page_alloc.cpp (ds_page_aligned_alloc mlock failure cleanup)",
        "desc": (
            "In `deepspeed/csrc/pin_memory/page_alloc.cpp`, `ds_page_aligned_alloc(size)` allocates page-aligned host RAM for fast GPU DMA transfers and then calls Linux's `mlock(ptr, size)` to page-lock (pin) it:\n"
            "- First, round `req_bytes` UP to a multiple of `page_size` (4096 bytes): `aligned_bytes = ((req_bytes + page_size - 1) // page_size) * page_size`.\n"
            "- Allocate `aligned_bytes` on the heap (`heap_used += aligned_bytes`).\n"
            "- Next, attempt `mlock`: it succeeds only if `aligned_bytes <= mlock_limit_bytes` (Linux's `RLIMIT_MEMLOCK`).\n"
            "- **If `mlock` fails (`aligned_bytes > mlock_limit_bytes`)**: you MUST `free(ptr)` (`heap_used -= aligned_bytes`) and return `[0, heap_used]` (`0` = `nullptr` in C++)! (The buggy starter forgets to `free(ptr)` when `mlock` fails, leaking `aligned_bytes` of heap RAM!)\n"
            "- **If `mlock` succeeds**: return `[aligned_bytes, heap_used]`."
        ),
        "py_name": "ds_page_aligned_alloc",
        "py_starter": (
            "def ds_page_aligned_alloc(req_bytes, page_size, mlock_limit_bytes, heap_used):\n"
            "    if req_bytes <= 0:\n"
            "        return [0, heap_used]\n"
            "    aligned_bytes = ((req_bytes + page_size - 1) // page_size) * page_size\n"
            "    # Step 1: allocate page-aligned host buffer on the heap\n"
            "    heap_used += aligned_bytes\n"
            "    # Step 2: attempt mlock(ptr, aligned_bytes)\n"
            "    if aligned_bytes > mlock_limit_bytes:\n"
            "        # BUG: mlock failed, but starter returns nullptr (0) without freeing the heap buffer!\n"
            "        return [0, heap_used]\n"
            "    return [aligned_bytes, heap_used]\n"
        ),
        "py_sol": (
            "def ds_page_aligned_alloc(req_bytes, page_size, mlock_limit_bytes, heap_used):\n"
            "    if req_bytes <= 0:\n"
            "        return [0, heap_used]\n"
            "    aligned_bytes = ((req_bytes + page_size - 1) // page_size) * page_size\n"
            "    # Step 1: allocate page-aligned host buffer on the heap\n"
            "    heap_used += aligned_bytes\n"
            "    # Step 2: attempt mlock(ptr, aligned_bytes)\n"
            "    if aligned_bytes > mlock_limit_bytes:\n"
            "        heap_used -= aligned_bytes\n"
            "        return [0, heap_used]\n"
            "    return [aligned_bytes, heap_used]\n"
        ),
        "cpp_name": "ds_page_aligned_alloc",
        "cpp_starter": (
            "void ds_page_aligned_alloc(int req_bytes, int page_size, int mlock_limit_bytes, int heap_used, int* out_res) {\n"
            "    if (req_bytes <= 0) {\n"
            "        out_res[0] = 0; out_res[1] = heap_used; return;\n"
            "    }\n"
            "    int aligned_bytes = ((req_bytes + page_size - 1) / page_size) * page_size;\n"
            "    heap_used += aligned_bytes; // posix_memalign(&ptr, page_size, aligned_bytes)\n"
            "    if (aligned_bytes > mlock_limit_bytes) {\n"
            "        // BUG: mlock(ptr, aligned_bytes) != 0, returns nullptr without free(ptr)!\n"
            "        out_res[0] = 0;\n"
            "        out_res[1] = heap_used;\n"
            "        return;\n"
            "    }\n"
            "    out_res[0] = aligned_bytes;\n"
            "    out_res[1] = heap_used;\n"
            "}\n"
        ),
        "cpp_sol": (
            "void ds_page_aligned_alloc(int req_bytes, int page_size, int mlock_limit_bytes, int heap_used, int* out_res) {\n"
            "    if (req_bytes <= 0) {\n"
            "        out_res[0] = 0; out_res[1] = heap_used; return;\n"
            "    }\n"
            "    int aligned_bytes = ((req_bytes + page_size - 1) / page_size) * page_size;\n"
            "    heap_used += aligned_bytes; // posix_memalign(&ptr, page_size, aligned_bytes)\n"
            "    if (aligned_bytes > mlock_limit_bytes) {\n"
            "        heap_used -= aligned_bytes; // free(ptr) on mlock failure before early return\n"
            "        out_res[0] = 0;\n"
            "        out_res[1] = heap_used;\n"
            "        return;\n"
            "    }\n"
            "    out_res[0] = aligned_bytes;\n"
            "    out_res[1] = heap_used;\n"
            "}\n"
        ),
        "hint": "Inside `if aligned_bytes > mlock_limit_bytes:`, add `heap_used -= aligned_bytes` (`free(ptr)`) right before returning `[0, heap_used]`.",
        "tests": [
            {"a": "[1000, 4096, 8192, 0]", "e": "[4096, 4096]"},
            {"a": "[5000, 4096, 4096, 1024]", "e": "[0, 1024]"},
            {"a": "[4096, 4096, 4096, 0]", "e": "[4096, 4096]"},
            {"a": "[0, 4096, 4096, 512]", "e": "[0, 512]", "h": 1},
        ],
        "v": [
            ["req_bytes (int)", "Unaligned byte count requested for a pinned host staging buffer."],
            ["page_size (int)", "OS virtual memory page size in bytes (typically `4096` bytes on x86_64 Linux)."],
            ["mlock_limit_bytes (int)", "Maximum bytes the process is allowed to page-lock (`RLIMIT_MEMLOCK`)."],
            ["heap_used (int)", "Total heap bytes currently allocated by the process."],
            ["aligned_bytes (int)", "`req_bytes` rounded up to a whole number of OS pages (`((req_bytes + page_size - 1) // page_size) * page_size`)."],
        ],
        "lines_py": [
            ["    aligned_bytes = ((req_bytes + page_size - 1) // page_size) * page_size", "Rounds `req_bytes` up to the next multiple of `page_size` (`4096`) as required by `posix_memalign` / `mlock`."],
            ["    heap_used += aligned_bytes", "Allocates the page-aligned block on the CPU heap."],
            ["    if aligned_bytes > mlock_limit_bytes:", "Models `mlock_ret = mlock(ptr, aligned_bytes)` failing when `aligned_bytes` exceeds `RLIMIT_MEMLOCK`."],
            ["        heap_used -= aligned_bytes", "THE FIX (`ds_page_aligned_alloc` pattern): Calls `free(ptr)` on the early-exit error branch before returning `nullptr`!"],
            ["        return [0, heap_used]", "Returns `0` (`nullptr`) with zero leaked heap bytes."],
        ],
        "lines_cpp": [
            ["if (aligned_bytes > mlock_limit_bytes) {", "Checks if `mlock(ptr, size) != 0`."],
            ["    heap_used -= aligned_bytes; out_res[0] = 0; out_res[1] = heap_used; return;", "Calls `free(ptr); return nullptr;` so the unpinned block is neither leaked nor used after free."],
        ],
    },

    # =========================================================================
    # LEVEL 8, EXERCISE 1: DynamicCache & QuantizedLayer.reset() Bug (#48509/#48809)
    # =========================================================================
    {
        "id": "dynamic_cache_reset",
        "l": 8,
        "title": "8.1 DynamicCache & QuantizedLayer reset() Resurrection Bug (Transformers #48509 / #48809)",
        "real_bug": "huggingface/transformers#48509 & #48809 (DynamicLayer.reset & QuantizedLayer.reset leaving is_initialized=True and stale zero/quantized tokens)",
        "desc": (
            "In `huggingface/transformers`, `DynamicLayer` (and its subclass `QuantizedLayer`) stores past keys in `keys` (and quantized history in `quantized_keys`), with boolean flag `is_initialized`.\n"
            "When `update(new_keys)` runs:\n"
            "- If `not is_initialized`: it initializes the cache fresh (`keys = list(new_keys)`, `quantized_keys = list(new_keys)`, `is_initialized = True`).\n"
            "- If `is_initialized`: it concatenates `new_keys` onto the existing history (`keys = quantized_keys + list(new_keys)` for `QuantizedLayer`, or `keys = keys + list(new_keys)`).\n"
            "Look at what the buggy inherited `CacheLayerMixin.reset()` did on `DynamicLayer` / `QuantizedLayer`:\n"
            "It only zeroed `keys` in-place (`keys = [0] * len(keys)`) and left `quantized_keys` and `is_initialized = True` untouched! "
            "So after `reset()`, calling `update(prompt2_keys)` saw `is_initialized == True` and concatenated `prompt2_keys` onto the old conversation's `quantized_keys` (or stale zeros)!\n"
            "Fix `reset_and_update_dynamic_cache(prompt1_keys, prompt2_keys, is_quantized)` so that the `reset()` step properly sets `keys = []`, `quantized_keys = []`, and `is_initialized = False` before `prompt2_keys` is updated, returning `[len(keys), keys]`."
        ),
        "py_name": "reset_and_update_dynamic_cache",
        "py_starter": (
            "def reset_and_update_dynamic_cache(prompt1_keys, prompt2_keys, is_quantized):\n"
            "    # Conversation 1: update cache with prompt1_keys\n"
            "    keys = list(prompt1_keys)\n"
            "    quantized_keys = list(prompt1_keys) if is_quantized else []\n"
            "    is_initialized = True\n"
            "    # BUG (transformers#48509): inherited reset() zeroes `keys` in-place, leaves is_initialized=True and quantized_keys intact!\n"
            "    keys = [0] * len(keys)\n"
            "    # Conversation 2: update cache with prompt2_keys\n"
            "    if not is_initialized:\n"
            "        keys = list(prompt2_keys)\n"
            "        quantized_keys = list(prompt2_keys) if is_quantized else []\n"
            "        is_initialized = True\n"
            "    else:\n"
            "        base = quantized_keys if is_quantized else keys\n"
            "        keys = base + list(prompt2_keys)\n"
            "    return [len(keys), keys]\n"
        ),
        "py_sol": (
            "def reset_and_update_dynamic_cache(prompt1_keys, prompt2_keys, is_quantized):\n"
            "    # Conversation 1: update cache with prompt1_keys\n"
            "    keys = list(prompt1_keys)\n"
            "    quantized_keys = list(prompt1_keys) if is_quantized else []\n"
            "    is_initialized = True\n"
            "    # FIX (DynamicLayer.reset): drop state and mark is_initialized = False instead of zeroing in-place!\n"
            "    keys = []\n"
            "    quantized_keys = []\n"
            "    is_initialized = False\n"
            "    # Conversation 2: update cache with prompt2_keys\n"
            "    if not is_initialized:\n"
            "        keys = list(prompt2_keys)\n"
            "        quantized_keys = list(prompt2_keys) if is_quantized else []\n"
            "        is_initialized = True\n"
            "    else:\n"
            "        base = quantized_keys if is_quantized else keys\n"
            "        keys = base + list(prompt2_keys)\n"
            "    return [len(keys), keys]\n"
        ),
        "cpp_name": "reset_and_update_dynamic_cache",
        "cpp_starter": (
            "void reset_and_update_dynamic_cache(const int* p1, int len1, const int* p2, int len2, int is_quantized, int* out_len, int* out_keys) {\n"
            "    int keys[32], q_keys[32];\n"
            "    int cur_len = len1;\n"
            "    for (int i = 0; i < len1; i++) { keys[i] = p1[i]; q_keys[i] = p1[i]; }\n"
            "    int is_initialized = 1;\n"
            "    // BUG (transformers#48509): reset() zeroes keys in-place, keeps cur_len and is_initialized=1!\n"
            "    for (int i = 0; i < cur_len; i++) { keys[i] = 0; }\n"
            "    // Conversation 2 update(p2):\n"
            "    if (!is_initialized) {\n"
            "        cur_len = len2;\n"
            "        for (int i = 0; i < len2; i++) { out_keys[i] = p2[i]; }\n"
            "    } else {\n"
            "        for (int i = 0; i < cur_len; i++) { out_keys[i] = is_quantized ? q_keys[i] : keys[i]; }\n"
            "        for (int i = 0; i < len2; i++) { out_keys[cur_len + i] = p2[i]; }\n"
            "        cur_len = cur_len + len2;\n"
            "    }\n"
            "    *out_len = cur_len;\n"
            "}\n"
        ),
        "cpp_sol": (
            "void reset_and_update_dynamic_cache(const int* p1, int len1, const int* p2, int len2, int is_quantized, int* out_len, int* out_keys) {\n"
            "    int keys[32], q_keys[32];\n"
            "    int cur_len = len1;\n"
            "    for (int i = 0; i < len1; i++) { keys[i] = p1[i]; q_keys[i] = p1[i]; }\n"
            "    int is_initialized = 1;\n"
            "    // FIX (DynamicLayer.reset): reset cur_len = 0 and is_initialized = 0!\n"
            "    cur_len = 0;\n"
            "    is_initialized = 0;\n"
            "    // Conversation 2 update(p2):\n"
            "    if (!is_initialized) {\n"
            "        cur_len = len2;\n"
            "        for (int i = 0; i < len2; i++) { out_keys[i] = p2[i]; }\n"
            "    } else {\n"
            "        for (int i = 0; i < cur_len; i++) { out_keys[i] = is_quantized ? q_keys[i] : keys[i]; }\n"
            "        for (int i = 0; i < len2; i++) { out_keys[cur_len + i] = p2[i]; }\n"
            "        cur_len = cur_len + len2;\n"
            "    }\n"
            "    *out_len = cur_len;\n"
            "}\n"
        ),
        "hint": "In the reset step, replace `keys = [0] * len(keys)` with `keys = []`, `quantized_keys = []`, and `is_initialized = False` (and in C++, set `cur_len = 0; is_initialized = 0;`).",
        "tests": [
            {"a": "[[11, 12, 13, 14, 15, 16], [21, 22, 23], False]", "e": "[3, [21, 22, 23]]"},
            {"a": "[[11, 12, 13, 14, 15, 16], [21, 22, 23], True]", "e": "[3, [21, 22, 23]]"},
            {"a": "[[99], [1, 2], True]", "e": "[2, [1, 2]]"},
            {"a": "[[5, 6], [7], False]", "e": "[1, [7]]", "h": 1},
        ],
        "v": [
            ["prompt1_keys (list[int] / const int*)", "Key tokens stored in the KV-cache during Conversation 1 (before `.reset()`)."],
            ["prompt2_keys (list[int] / const int*)", "Key tokens arriving at the start of Conversation 2 (after `.reset()`)."],
            ["is_quantized (bool / int)", "Whether the cache layer is a `QuantizedLayer` (which stores past tokens in `_quantized_keys`) or a plain `DynamicLayer`."],
            ["keys (list[int])", "`self.keys` tensor on `DynamicLayer`. In `DynamicLayer`, `get_seq_length()` is `len(keys)` (`keys.shape[-2]`)!"],
            ["quantized_keys (list[int])", "`self._quantized_keys` on `QuantizedLayer`. Untouched by `self.keys.zero_()`, so leaving `is_initialized=True` resurrected Conversation 1's keys!"],
            ["is_initialized (bool / int)", "Flag telling `update()` whether the layer already holds valid history (`True`) or is empty (`False`)."],
        ],
        "lines_py": [
            ["    keys = list(prompt1_keys)", "Populates the cache with Conversation 1's keys (e.g. 6 tokens)."],
            ["    quantized_keys = list(prompt1_keys) if is_quantized else []", "In `QuantizedLayer`, quantizes and stores Conversation 1's keys in `_quantized_keys`."],
            ["    keys = []", "THE FIX (Part 1): Drops `self.keys` (`self.keys = None`) instead of zeroing in-place so `get_seq_length()` returns `0`."],
            ["    quantized_keys = []", "THE FIX (Part 2): Clears quantized state so pre-reset tokens cannot remain in memory."],
            ["    is_initialized = False", "THE FIX (Part 3): Resets `self.is_initialized = False` so the next `update()` takes the fresh initialization branch instead of concatenating onto stale history!"],
        ],
        "lines_cpp": [
            ["cur_len = 0; is_initialized = 0;", "THE FIX: Resets both the dynamic length and `is_initialized` flag so `update(p2)` starts at index `0`."],
        ],
    },

    # =========================================================================
    # LEVEL 8, EXERCISE 2: Ring-Buffer / Rolling Window state_dict Resume
    # =========================================================================
    {
        "id": "ring_buffer_state_dict",
        "l": 8,
        "title": "8.2 Circular Ring-Buffer Cache Checkpoint & Cursor Resume (TorchMetrics Running Finding)",
        "real_bug": "Lightning-AI/torchmetrics Running / RunningMean / RunningSum (20261019_053000_torchmetrics-running-state-dict-finding.md)",
        "desc": (
            "A fixed-memory rolling-window cache (like `TorchMetrics` `Running` or a sliding-window KV ring buffer of size `window_size`) stores incoming values in a fixed array `buf = [0] * window_size` using a circular write cursor `cursor = step % window_size` and tracks `count` (how many total items have been written so far).\n"
            "When `stream1` writes `len(stream1)` values, we save a checkpoint (`state_dict`), load it into a fresh ring buffer, and write `stream2`.\n"
            "In your `TorchMetrics` `Running` finding, `state_dict` saved `buf` (`val_list`), but **forgot to save the write position / count**, so after loading the checkpoint, `count` reset to `0` (and `cursor` reset to `0`), overwriting `buf[0]` instead of continuing at `len(stream1) % window_size`!\n"
            "Fix `resume_ring_buffer(window_size, stream1, stream2)` so the checkpoint saves and restores BOTH `'buf'` and `'count'`, and return `[buf, sum(buf[:min(count, window_size)])]`."
        ),
        "py_name": "resume_ring_buffer",
        "py_starter": (
            "def write_ring(state, x, window_size):\n"
            "    idx = state['count'] % window_size\n"
            "    state['buf'][idx] = x\n"
            "    state['count'] += 1\n\n"
            "def resume_ring_buffer(window_size, stream1, stream2):\n"
            "    ring1 = {'buf': [0] * window_size, 'count': 0}\n"
            "    for x in stream1:\n"
            "        write_ring(ring1, x, window_size)\n"
            "    # BUG (TorchMetrics Running): saves 'buf' in state_dict, but forgets 'count' (the ring cursor)!\n"
            "    state_dict = {'buf': list(ring1['buf'])}\n"
            "    # Fresh ring buffer on resume:\n"
            "    ring2 = {'buf': [0] * window_size, 'count': 0}\n"
            "    ring2.update(state_dict)\n"
            "    for x in stream2:\n"
            "        write_ring(ring2, x, window_size)\n"
            "    valid = min(ring2['count'], window_size)\n"
            "    return [ring2['buf'], sum(ring2['buf'][:valid])]\n"
        ),
        "py_sol": (
            "def write_ring(state, x, window_size):\n"
            "    idx = state['count'] % window_size\n"
            "    state['buf'][idx] = x\n"
            "    state['count'] += 1\n\n"
            "def resume_ring_buffer(window_size, stream1, stream2):\n"
            "    ring1 = {'buf': [0] * window_size, 'count': 0}\n"
            "    for x in stream1:\n"
            "        write_ring(ring1, x, window_size)\n"
            "    # FIX: persist BOTH the backing array 'buf' AND the write cursor 'count' in state_dict!\n"
            "    state_dict = {'buf': list(ring1['buf']), 'count': ring1['count']}\n"
            "    ring2 = {'buf': [0] * window_size, 'count': 0}\n"
            "    ring2.update(state_dict)\n"
            "    for x in stream2:\n"
            "        write_ring(ring2, x, window_size)\n"
            "    valid = min(ring2['count'], window_size)\n"
            "    return [ring2['buf'], sum(ring2['buf'][:valid])]\n"
        ),
        "cpp_name": "resume_ring_buffer",
        "cpp_starter": (
            "void write_ring(int* buf, int* count, int x, int window_size) {\n"
            "    int idx = (*count) % window_size;\n"
            "    buf[idx] = x;\n"
            "    *count += 1;\n"
            "}\n\n"
            "void resume_ring_buffer(int window_size, const int* s1, int len1, const int* s2, int len2, int* out_buf, int* out_sum) {\n"
            "    int buf1[32];\n"
            "    for (int i = 0; i < window_size; i++) { buf1[i] = 0; }\n"
            "    int count1 = 0;\n"
            "    for (int i = 0; i < len1; i++) { write_ring(buf1, &count1, s1[i], window_size); }\n"
            "    // Fresh ring buffer on resume:\n"
            "    for (int i = 0; i < window_size; i++) { out_buf[i] = buf1[i]; }\n"
            "    // BUG (TorchMetrics Running): forgets to restore count2 = count1 from state_dict!\n"
            "    int count2 = 0;\n"
            "    for (int i = 0; i < len2; i++) { write_ring(out_buf, &count2, s2[i], window_size); }\n"
            "    int valid = (count2 < window_size) ? count2 : window_size;\n"
            "    int total = 0;\n"
            "    for (int i = 0; i < valid; i++) { total += out_buf[i]; }\n"
            "    *out_sum = total;\n"
            "}\n"
        ),
        "cpp_sol": (
            "void write_ring(int* buf, int* count, int x, int window_size) {\n"
            "    int idx = (*count) % window_size;\n"
            "    buf[idx] = x;\n"
            "    *count += 1;\n"
            "}\n\n"
            "void resume_ring_buffer(int window_size, const int* s1, int len1, const int* s2, int len2, int* out_buf, int* out_sum) {\n"
            "    int buf1[32];\n"
            "    for (int i = 0; i < window_size; i++) { buf1[i] = 0; }\n"
            "    int count1 = 0;\n"
            "    for (int i = 0; i < len1; i++) { write_ring(buf1, &count1, s1[i], window_size); }\n"
            "    for (int i = 0; i < window_size; i++) { out_buf[i] = buf1[i]; }\n"
            "    int count2 = count1; // FIX: restore ring-buffer write cursor from state_dict!\n"
            "    for (int i = 0; i < len2; i++) { write_ring(out_buf, &count2, s2[i], window_size); }\n"
            "    int valid = (count2 < window_size) ? count2 : window_size;\n"
            "    int total = 0;\n"
            "    for (int i = 0; i < valid; i++) { total += out_buf[i]; }\n"
            "    *out_sum = total;\n"
            "}\n"
        ),
        "hint": "Include `'count': ring1['count']` in `state_dict` (and in C++, set `int count2 = count1;`) so `stream2` writes to `count1 % window_size` instead of overwriting slot `0`!",
        "tests": [
            {"a": "[3, [10, 20, 30, 40], [50]]", "e": "[[40, 50, 30], 120]"},
            {"a": "[4, [1, 2], [3]]", "e": "[[1, 2, 3, 0], 6]"},
            {"a": "[2, [5, 6, 7], [8]]", "e": "[[7, 8], 15]"},
            {"a": "[3, [1, 2, 3, 4, 5], [6, 7]]", "e": "[[7, 5, 6], 18]", "h": 1},
        ],
        "v": [
            ["window_size (int)", "Fixed capacity of the circular ring buffer (`window` in `TorchMetrics` `Running`)."],
            ["stream1 (list[int] / const int*)", "Values written into the ring buffer before `state_dict()` checkpoint save."],
            ["stream2 (list[int] / const int*)", "Values written into the ring buffer after `load_state_dict()` resume."],
            ["buf (list[int] / int*)", "Fixed-length array of size `window_size` holding the most recent `window_size` items."],
            ["count (int)", "Total items written so far; determines both the next write slot (`count % window_size`) and how many slots are valid (`min(count, window_size)`)."],
            ["valid (int)", "Number of populated slots in `buf` (`min(count, window_size)`)."],
        ],
        "lines_py": [
            ["def write_ring(state, x, window_size):", "Writes one new observation `x` into the circular ring buffer."],
            ["    idx = state['count'] % window_size", "Modulo wraparound selects the oldest slot in `0 .. window_size - 1` to overwrite."],
            ["    state['buf'][idx] = x; state['count'] += 1", "Stores `x` at `idx` and advances the cumulative write counter."],
            ["    state_dict = {'buf': list(ring1['buf']), 'count': ring1['count']}", "THE FIX (`TorchMetrics Running` finding): Persists BOTH the ring array `buf` AND its write counter `count` in `state_dict`!"],
            ["    valid = min(ring2['count'], window_size)", "When fewer than `window_size` items have been seen in total (`[1, 2]` + `[3]` in a size-4 window), `valid` is `3` (not `1`!)."],
        ],
        "lines_cpp": [
            ["int count2 = count1;", "THE FIX: Restores the circular write cursor `count` alongside `out_buf` on checkpoint reload."],
        ],
    },
]
