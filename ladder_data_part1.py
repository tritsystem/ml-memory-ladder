# -*- coding: utf-8 -*-
"""
Levels 1-4 of the ML Memory & Cache Systems Ladder.
Covers:
  Level 1: Pointers, Addresses & Unchecked Allocation (NULL deref & partial-alloc OOM)
  Level 2: Heap Lifetimes & Memory Leaks (Pointer overwrite & loop scratch churn)
  Level 3: Dangling Pointers, Use-After-Free & Double-Free (Early-exit fallthrough & storage resize)
  Level 4: Allocation Math & Raw Byte Traps (IO2BO integer overflow & element_size vs scalar_type)
"""

LEVELS_1_4 = [
    "1. Pointers, Addresses & NULL Checks",
    "2. Heap Lifetimes & Memory Leaks",
    "3. Use-After-Free & Double-Free",
    "4. Allocation Math & Raw Byte Traps",
]

QUIZ_1_4 = [
    # ---------------- Level 1: Pointers, Addresses & NULL Checks ----------------
    {
        "l": 1,
        "q": "In C and C++, what is a pointer variable (like `float *workspace`) in plain words?",
        "o": [
            "A small box that holds the memory address (numbered slot) where the real data lives",
            "A giant container that holds all the floating-point numbers directly inside its own name",
            "A special GPU core that multiplies matrices automatically whenever the variable is read",
            "A backup copy of the entire array that the compiler saves on the hard drive for safety",
        ],
        "w": "Analogy: a pointer is a sticky note with a warehouse room number written on it, not the warehouse room itself. Copying the pointer only copies the sticky note.",
    },
    {
        "l": 1,
        "q": "What do `malloc(bytes)` and `calloc(count, size)` return when the system runs out of memory (OOM)?",
        "o": [
            "They return `NULL` (address 0) to say no memory block could be reserved",
            "They automatically pause the program until the user closes another window",
            "They return a valid pointer to a tiny 1-byte emergency buffer on the stack",
            "They throw a Python `MemoryError` that skips the rest of the C++ function",
        ],
        "w": "In C (`malloc`/`calloc`), allocation failure does NOT throw an exception; it quietly returns `NULL` (address `0`). If you don't check for `NULL`, the very next write crashes the process.",
    },
    {
        "l": 1,
        "q": "What happens in C++ if `workspace = (float**)calloc(n, sizeof(float*))` returns `NULL` and the next line runs `workspace[0] = buf`?",
        "o": [
            "Undefined behavior and an immediate `SIGSEGV` crash from writing to address 0",
            "The array automatically resizes itself to fit item 0 and continues normally",
            "The write is safely ignored by the CPU and `workspace` stays `NULL` without error",
            "The value `buf` is stored in a temporary CPU register until memory frees up",
        ],
        "w": "This is CWE-476 (NULL Pointer Dereference) — the exact bug you found and filed in `deepspeedai/DeepSpeed#8565` (`shm.cpp:390`) and `eclipse-mosquitto/mosquitto#3762` (`ctrl_shell.c`).",
    },
    {
        "l": 1,
        "q": "How does `calloc(count, size)` differ from `malloc(count * size)` when allocation succeeds?",
        "o": [
            "`calloc` zeroes every byte of the new block and checks `count * size` for overflow",
            "`calloc` allocates memory on the GPU while `malloc` allocates memory on the CPU",
            "`calloc` leaves old garbage bytes in the block so it finishes twice as fast",
            "`calloc` never returns `NULL` even when all physical RAM is completely full",
        ],
        "w": "`calloc` both zero-initializes the memory AND checks if `count * size` would overflow `size_t`, whereas `malloc` leaves uninitialized garbage bytes and takes a single already-multiplied size.",
    },
    {
        "l": 1,
        "q": "Suppose a function allocates 3 buffers in a row (`a = calloc(...)`, `b = calloc(...)`, `c = calloc(...)`). What goes wrong if `c` is `NULL` and you just `return -1;`?",
        "o": [
            "Buffers `a` and `b` are leaked because nobody freed them before returning",
            "Buffers `a` and `b` are freed automatically because the function returned `-1`",
            "Nothing goes wrong because `calloc` rolls back earlier calls when one fails",
            "Calling `free(c)` is required or the `NULL` pointer itself will leak memory",
        ],
        "w": "C++ has no automatic rollback for raw `calloc` calls. If allocation #3 fails after #1 and #2 succeeded, you must `free(a)` and `free(b)` before returning (`free(NULL)` is a safe no-op). You hit this exact compound-guard pattern in `DeepSpeed#8565`.",
    },
    {
        "l": 1,
        "q": "Is calling `free(NULL)` in C/C++ a crash, or is it safe?",
        "o": [
            "It is guaranteed by the C/C++ standard to be a safe no-op (it does nothing)",
            "It immediately crashes the program with a double-free segmentation fault",
            "It corrupts the heap allocator's free-list header at address zero",
            "It frees the most recently allocated block on the current thread by mistake",
        ],
        "w": "Because `free(NULL)` is guaranteed to do nothing safely, a single cleanup block can safely run `free(a); free(b); free(c);` when any one of them failed and is `NULL`!",
    },
    {
        "l": 1,
        "q": "Why do C++ static analyzers sometimes miss `if ((p = calloc(n, sz)) == NULL)` if written carelessly?",
        "o": [
            "The assignment `p = calloc(...)` is wrapped inside `(...)` on the left of `== NULL`",
            "The C compiler rewrites `calloc` into `malloc` before the parser reads the file",
            "`NULL` is only defined in Python files and never appears in C or C++ code",
            "Static analyzers only inspect comments and never look inside `if` conditions",
        ],
        "w": "Real lesson from your Reagent `C07` detector (`httpd` `suexec.c`): the left-hand side of `== NULL` is the parenthesized assignment `(p = calloc(...))`, so the checker must unwrap the assignment to see that `p` IS guarded.",
    },
    {
        "l": 1,
        "q": "Why do C and C++ parse `if (!a || !b)` differently inside tree-sitter AST tools?",
        "o": [
            "C wraps the `if` test in `parenthesized_expression`, while C++ uses `condition_clause`",
            "C++ does not allow the `||` operator inside an `if` statement at all",
            "C evaluates `||` from right to left while C++ evaluates it from left to right",
            "C++ turns every pointer into a boolean at compile time and deletes the `if`",
        ],
        "w": "Real lesson #7 from your Reagent `C07` work on `DeepSpeed/csrc/cpu/comm/shm.cpp`: `.c` and `.cpp` grammars wrap `if` conditions in different AST node types (`parenthesized_expression` vs `condition_clause`).",
    },

    # ---------------- Level 2: Heap Lifetimes & Memory Leaks ----------------
    {
        "l": 2,
        "q": "What is a memory leak (CWE-401) in plain everyday terms?",
        "o": [
            "Rented memory whose last address pointer was lost or overwritten before `free()` was called",
            "Memory that physically drips electricity out of the RAM stick when the CPU gets hot",
            "A buffer that was freed twice in the same function, corrupting the allocator",
            "A variable on the CPU stack that disappears as soon as its function returns",
        ],
        "w": "Analogy: you rent a storage locker (`malloc`), write the locker number on a single sticky note (`ptr`), and then erase the sticky note to write a different number (`ptr = other`) without ever returning the first locker's key (`free`).",
    },
    {
        "l": 2,
        "q": "In `DeepSpeed#8565` (`shm.cpp`), `workspace_buf = malloc(sz)` was passed to `shared_create(workspace_buf)` (which copied its bytes to a file) and then `workspace_buf = mmap_ptr`. Why did that leak?",
        "o": [
            "`workspace_buf` was overwritten with `mmap_ptr` before `free(workspace_buf)` was called",
            "`shared_create` automatically freed `workspace_buf` twice inside the kernel",
            "`malloc(sz)` allocated memory on the stack instead of the heap in `shm.cpp`",
            "`mmap_ptr` was smaller than `sz`, leaving the tail bytes of `workspace_buf` dangling",
        ],
        "w": "In `DeepSpeed#8565`, `workspace_buf` held the only pointer to the heap block from line 362. Line 365 reassigned `workspace_buf = (allreduce_workspace*)allreduce_buffer.bytes`, permanently orphaning the heap block! Inserting `free(workspace_buf);` at line 364 fixed it.",
    },
    {
        "l": 2,
        "q": "What happens when a C++ inference loop calls `float *scratch = (float*)malloc(1048576)` on every token step and forgets `free(scratch)`?",
        "o": [
            "Resident memory (RSS) grows by 1 MB per token until the OS OOM-kills the process",
            "The compiler reuses the exact same 1 MB address on every loop iteration automatically",
            "Only the last token's 1 MB stays in RAM because loop variables auto-free `malloc`",
            "The program slows down on step 1 but memory usage stays completely flat forever",
        ],
        "w": "Notice the distinction: the pointer variable `scratch` lives on the stack and resets each iteration, but the 1 MB block it pointed to lives on the HEAP and stays allocated forever unless `free(scratch)` is called.",
    },
    {
        "l": 2,
        "q": "Even if you DO call `free(scratch)` at the end of every token step, why do high-speed ML kernels pre-allocate one `workspace` buffer outside the loop instead?",
        "o": [
            "Calling `malloc`/`free` every step takes locks, fragments the heap, and thrashes CPU caches",
            "`free()` only works once per program and fails if called inside a `for` loop",
            "Heap memory cannot hold floating-point numbers if allocated more than once",
            "Pre-allocating outside the loop uses more total RAM than leaking inside the loop",
        ],
        "w": "Allocating and freeing megabytes on every token step forces OS page faults, allocator lock contention, and cold L1/L2 caches. Pre-allocating a reusable workspace once (and growing it only when `needed > capacity`) is standard in PyTorch/DeepSpeed.",
    },
    {
        "l": 2,
        "q": "Why did your Reagent `C05` leak detector initially miss `T *p = (T*)malloc(n)` before you added `_semantic_parent()`?",
        "o": [
            "The `(T*)` cast wrapped the `malloc` call, hiding the outer assignment node",
            "Casting `malloc` to `(T*)` automatically registers the pointer with a garbage collector",
            "`malloc` inside a C++ file returns a reference instead of a raw pointer",
            "Tree-sitter deletes all cast expressions when parsing C and C++ source files",
        ],
        "w": "In C/C++ ASTs, `(T*)malloc(n)` makes the parent of `malloc(n)` a `cast_expression`, not the `init_declarator` or `assignment_expression`. Walking up through casts (`_semantic_parent`) caught the real DeepSpeed `shm.cpp` leak!",
    },
    {
        "l": 2,
        "q": "In C++, what is RAII (`std::unique_ptr` / `std::vector`) and why does it prevent early-return memory leaks?",
        "o": [
            "It ties heap memory to a stack object whose destructor automatically frees it on any return or throw",
            "It disables the heap completely and forces all tensors of any size onto the CPU stack",
            "It runs a background garbage-collection thread that scans RAM every 10 seconds",
            "It ignores allocation failures so the function never has to return early on error",
        ],
        "w": "When a stack object (`std::unique_ptr<float[]>`) goes out of scope — whether by normal `return`, early error `return`, or an exception — the C++ compiler automatically emits the destructor call that frees the heap block.",
    },
    {
        "l": 2,
        "q": "When a function transfers ownership of a newly allocated buffer to its caller (`return buf;`), should the function call `free(buf)` before returning on the success path?",
        "o": [
            "No: on the success path the caller now owns `buf`; only error paths inside the function should `free(buf)`",
            "Yes: every function that calls `malloc` must always call `free` before `return buf`",
            "Yes, because `free(buf)` only marks the buffer as read-only for the caller",
            "No, and error paths inside the function should never call `free(buf)` either",
        ],
        "w": "If you `free(buf)` and then `return buf;`, you hand the caller a dangling (already-freed) pointer! Clean pattern: free `buf` on early error exits, and return `buf` intact on the success path.",
    },
    {
        "l": 2,
        "q": "In Python, CPython uses reference counting (`ob_refcnt`) plus a cycle detector. Why can Python objects still 'leak' memory in a long-running ML server?",
        "o": [
            "Appending tensors or caches to a global/module list or dict keeps their reference count > 0 forever",
            "Python never frees any variable until the entire computer is rebooted",
            "Reference counting only works for strings and never frees lists or numbers",
            "The garbage collector deletes objects that are still in use if RAM gets above 50%",
        ],
        "w": "If an unbounded dict, cache, or history list (`self.history.append(tensor)`) keeps a live reference to an object, its reference count never hits 0, so Python is forbidden from freeing it.",
    },

    # ---------------- Level 3: Use-After-Free & Double-Free ----------------
    {
        "l": 3,
        "q": "What actually happens to a pointer variable `p` when you call `free(p)` in C/C++?",
        "o": [
            "The heap block is returned to the allocator, but `p` still holds the old address (now a dangling pointer)",
            "`free(p)` automatically sets `p = NULL` in every function across the whole program",
            "The CPU physically erases the variable `p` from the stack so it cannot be read",
            "Nothing happens until the end of `main()`, when all `free()` calls run together",
        ],
        "w": "Because `p` is passed by value to `free(p)`, `free` cannot change `p` itself! `p` still holds the exact same address number on your sticky note, even though the warehouse room now belongs to someone else.",
    },
    {
        "l": 3,
        "q": "In your CFEngine `RemoveFilament` fix (`cfengine/core#6358`), what went wrong after `if (StringSetSize(filament->items) == 0) { StringSetDestroy(filament->items); free(filament); }`?",
        "o": [
            "There was no `return;`, so execution fell through to read `filament->db_id` (UAF) and `free(filament)` again (double-free)",
            "The `if` condition checked `== 0` instead of `< 0`, so the branch never ran",
            "`free(filament)` was called before `StringSetDestroy(filament->items)` inside the `if` block",
            "`OpenDB` allocated a new `filament` pointer without checking for `NULL` first",
        ],
        "w": "That single missing `return;` after the early-exit cleanup caused BOTH a CWE-416 Use-After-Free (`OpenDB(&db, filament->db_id)`) AND a CWE-415 Double-Free at the end of `RemoveFilament`! Adding `return;` fixed both.",
    },
    {
        "l": 3,
        "q": "Why is a Double-Free (`free(p)` followed later by `free(p)` again, CWE-415) so dangerous in C/C++?",
        "o": [
            "It corrupts the allocator's free-list metadata or frees a block already handed out to a new object",
            "It simply returns `false` on the second `free(p)` and wastes one CPU cycle",
            "It zeroes out the memory block a second time, which is safe but slightly slower",
            "It only affects 16-bit microcontrollers and is ignored on 64-bit Linux systems",
        ],
        "w": "When `free(p)` runs the first time, the allocator puts that block onto its free-list (and may immediately hand it to another `malloc`). Calling `free(p)` a second time either corrupts the free-list links or yanks memory out from under the new owner!",
    },
    {
        "l": 3,
        "q": "Why can a Use-After-Free (CWE-416) appear to 'pass all unit tests' on a developer's laptop and then crash or corrupt weights in production?",
        "o": [
            "Freed memory often keeps its old bytes briefly until another thread or allocation overwrites that slot",
            "Unit tests always run on the GPU where `free()` is completely disabled",
            "Compilers automatically delete `free()` calls when compiling in debug mode",
            "Production servers use 32-bit floats while laptops only use 64-bit floats",
        ],
        "w": "Reading `filament->db_id` right after `free(filament)` might read the old bytes still sitting in RAM during a quiet single-threaded test — which is why running under AddressSanitizer (`-fsanitize=address`, ASan) is critical to catch UAF immediately!",
    },
    {
        "l": 3,
        "q": "How does AddressSanitizer (ASan) catch a Use-After-Free the instant it happens, even if no other `malloc` has reused the memory yet?",
        "o": [
            "It places freed blocks in a quarantine queue and marks their shadow-memory bytes as poisoned",
            "It runs the program 1,000 times with random inputs until a segmentation fault occurs",
            "It replaces every pointer with a string containing the variable's source-code name",
            "It disables `free()` completely so no memory is ever returned during the test",
        ],
        "w": "ASan maintains 1 byte of 'shadow memory' per 8 bytes of real RAM and quarantines freed blocks. Every pointer dereference checks the shadow byte first; touching a poisoned (freed) address aborts immediately with a full stack trace.",
    },
    {
        "l": 3,
        "q": "In PyTorch / C++ tensor internals, what happens if you grab a raw pointer `float *p = tensor.data_ptr<float>()` and then call `tensor.resize_(bigger_shape)`?",
        "o": [
            "`resize_` allocates a new larger buffer and frees the old one, leaving `p` as a dangling pointer (UAF)",
            "`p` automatically updates its own address to point at the newly allocated buffer",
            "`resize_` refuses to run whenever any raw pointer variable exists in the function",
            "The new elements are stored in a linked list so the old pointer `p` stays valid forever",
        ],
        "w": "A raw pointer (`float *p`) or cached byte-offset view does NOT know when `Storage` reallocates. If `resize_` grows past current capacity, the old backing buffer is freed and `p` now points to freed memory!",
    },
    {
        "l": 3,
        "q": "What defensive habit after `free(p)` prevents a later cleanup path in the same struct or function from double-freeing `p`?",
        "o": [
            "Immediately set `p = NULL;` right after `free(p);` (and return if it is an early-exit branch)",
            "Call `malloc(0)` and assign the result to `p` so it points to a fresh block",
            "Add `p++` after `free(p)` so the pointer points to the next byte instead",
            "Wrap the second `free(p)` in a `try / catch` block to ignore the crash",
        ],
        "w": "Setting `p = NULL;` after `free(p);` turns any later `free(p)` into `free(NULL)` (a safe no-op) and turns any accidental dereference `*p` into an immediate, deterministic NULL fault instead of silent heap corruption.",
    },
    {
        "l": 3,
        "q": "In `cf-agent/abstract_dir.c`, why wasn't `if (local) { free(dir); } else { RemoteDirClose(dir); }` a Use-After-Free?",
        "o": [
            "`free(dir)` and `RemoteDirClose(dir)` sit in mutually exclusive `if` / `else` branches, so both never run on the same call",
            "`RemoteDirClose(dir)` does not actually read `dir` and only logs a message",
            "`free(dir)` in C only takes effect after the `else` branch finishes running",
            "`local` is always `false` in C programs so `free(dir)` is dead code",
        ],
        "w": "Real lesson from your Reagent `C02` branch-exclusivity fix: a static scanner that only checks line order without checking whether two statements sit in mutually exclusive `if`/`else` branches will false-positive on safe code.",
    },

    # ---------------- Level 4: Allocation Math & Raw Byte Traps ----------------
    {
        "l": 4,
        "q": "What is an Integer-Overflow-to-Buffer-Overflow (`IO2BO`, CWE-190 -> CWE-787) bug in `malloc(n * sizeof(float))`?",
        "o": [
            "A huge `n` makes `n * 4` wrap around past the integer limit to a tiny number, allocating a tiny buffer that the loop then overflows",
            "`sizeof(float)` changes from 4 to 8 at runtime when `n` is larger than 1,000",
            "`malloc` automatically converts negative numbers into `NULL` without overflowing",
            "Multiplying inside `malloc` is a syntax error in C++ unless cast to `double`",
        ],
        "w": "Suppose 32-bit `size_t` max is `4,294,967,295` (`0xFFFFFFFF`). If `n = 1,073,741,825`, then `n * 4 = 4,294,967,300`, which wraps around modulo $2^{32}$ to just **4 bytes**! `malloc(4)` succeeds, and writing `n` floats smashes the heap.",
    },
    {
        "l": 4,
        "q": "Why is the overflow check `if (n * sizeof(float) < n)` BROKEN in C/C++ for guarding `malloc(n * sizeof(float))`?",
        "o": [
            "Multiplication by 4 can wrap around multiple times and still land on a number `>= n`, or trigger signed undefined behavior",
            "The `<` operator in C/C++ only compares the lowest 8 bits of each number",
            "`sizeof(float)` is a floating-point number so `<` rounds both sides to zero",
            "The check is too strict and rejects every `n` greater than 100",
        ],
        "w": "The safe, non-overflowing guard divides the constant limit BEFORE multiplying: `if (elem_size != 0 && n > SIZE_MAX / elem_size) return NULL;` — or simply use `calloc(n, elem_size)`, which checks this internally!",
    },
    {
        "l": 4,
        "q": "In your Reagent `C06` triage on `cfengine/core` (`StrList_Insert`), why was `new_cap = old_cap * 2` NOT an exploitable integer overflow on real 64-bit systems?",
        "o": [
            "`old_cap` came from a prior real allocation that already succeeded in RAM, so OOM happens long before `old_cap * 2` can approach `SIZE_MAX`",
            "Multiplying by 2 in C can never overflow any integer type under any circumstance",
            "`realloc` ignores its size argument and always doubles the buffer safely",
            "64-bit integers have infinite range and no maximum value",
        ],
        "w": "Great systems reasoning from your ledger: to wrap `old_cap * 2` near `SIZE_MAX` ($2^{64}-1$), a prior allocation of $2^{63}$ bytes (8 exabytes!) would have had to succeed first, which is physically impossible.",
    },
    {
        "l": 4,
        "q": "In your `pytorch/executorch#23399` finding (`QuantizedKVCache`), why did passing `bfloat16` `k_val` into `torch.ops.llama.update_cache` fatally abort the whole Python process?",
        "o": [
            "The cache buffer was hardcoded to `float32` (4 bytes), so the C++ op's `element_size()` check failed `ET_CHECK_MSG` and called `abort()`",
            "Python does not support `bfloat16` tensors on CPU under any version of PyTorch",
            "`update_cache` ran out of GPU memory and threw a catchable `torch.cuda.OutOfMemoryError`",
            "The model weights were deleted when `QuantizedKVCache.from_float()` was called",
        ],
        "w": "In `executorch#23399`, `self.cache_fp_type` was hardcoded to `torch.float32` (4 bytes/elem), while on-device LLM inference passed `bf16`/`fp16` (2 bytes/elem). The C++ `ET_CHECK_MSG` on `element_size()` aborted the entire process (not catchable by `try/except`)!",
    },
    {
        "l": 4,
        "q": "What is the SECOND, even sneakier bug when a C++ custom op checks ONLY `src.element_size() == dst.element_size()` before calling `memcpy(dst_ptr, src_ptr, nbytes)`?",
        "o": [
            "`float16` and `bfloat16` both have `element_size() == 2`, so `memcpy` passes the check and silently reinterprets `bf16` bits as bogus `fp16` numbers!",
            "`memcpy` refuses to copy 2-byte elements and raises a hardware alignment fault",
            "`float16` and `bfloat16` have identical bit layouts so `memcpy` works fine between them",
            "`element_size()` returns the total tensor byte count instead of bytes per element",
        ],
        "w": "Both `fp16` (1 sign, 5 exponent, 10 mantissa bits) and `bf16` (1 sign, 8 exponent, 7 mantissa bits) are 2 bytes (`element_size() == 2`)! A raw `memcpy` gated only on byte width silently scrambles every number with zero error unless `scalar_type()` (dtype) is also checked or cast first.",
    },
    {
        "l": 4,
        "q": "Why did existing unit tests in `executorch` never catch the `QuantizedKVCache` `bf16`/`fp16` crash before your audit?",
        "o": [
            "Every unit test in CI fed `float32` inputs, even though real on-device LLM deployment runs in `bf16` or `fp16`",
            "The unit tests were marked `@unittest.skip` on all platforms",
            "`QuantizedKVCache` was never imported anywhere in the repository",
            "The bug only happens when the sequence length is greater than 1,000,000 tokens",
        ],
        "w": "Classic training/CI vs. deployment blind spot: CI tests defaulted to `torch.float32` (which happened to match the hardcoded `self.cache_fp_type = torch.float32`), while real quantized LLM runs use `bf16`/`fp16`.",
    },
    {
        "l": 4,
        "q": "What was the clean 2-line fix in `QuantizedKVCache._update_and_return_float_values()` (`executorch#23399`) that fixed BOTH the custom C++ `memcpy` path and the fallback `index_put_` path?",
        "o": [
            "Cast `k_val = k_val.to(k_out.dtype)` and `v_val = v_val.to(v_out.dtype)` once up front before either update branch runs",
            "Wrap `torch.ops.llama.update_cache` in a Python `try / except Exception:` block",
            "Delete the `ET_CHECK_MSG` assertion in C++ so `memcpy` copies 2-byte values into 4-byte slots",
            "Force the entire LLM model to run in `float64` on mobile devices",
        ],
        "w": "Casting `k_val` and `v_val` to `k_out.dtype` / `v_out.dtype` up front guarantees that both byte-width (`element_size()`) AND bit-format (`scalar_type()`) match the dequantized destination buffer on both paths!",
    },
    {
        "l": 4,
        "q": "In C++, why is `memcpy(dst, src, count)` a common bug when `dst` and `src` are `float*` arrays of length `count`?",
        "o": [
            "`memcpy`'s third argument is `n_bytes`, so passing `count` instead of `count * sizeof(float)` copies only the first quarter of the array!",
            "`memcpy` only works on `char*` strings and crashes if given `float*` pointers",
            "`memcpy` automatically multiplies its third argument by `sizeof(float)` twice",
            "`memcpy` swaps `dst` and `src` whenever `count` is an odd number",
        ],
        "w": "`memcpy(void *dst, const void *src, size_t nbytes)` always counts raw BYTES, never elements. For `count` floats (4 bytes each), passing `count` copies only `count` bytes (`count / 4` floats) and leaves the rest uncopied!",
    },
]

EXERCISES_1_4 = [
    # =========================================================================
    # LEVEL 1, EXERCISE 1: Unchecked calloc NULL Dereference (CWE-476)
    # =========================================================================
    {
        "id": "null_guard",
        "l": 1,
        "title": "1.1 Unchecked calloc() NULL Dereference (DeepSpeed #8565 & Mosquitto #3762)",
        "real_bug": "deepspeedai/DeepSpeed#8565 (csrc/cpu/comm/shm.cpp:375-390) & eclipse-mosquitto/mosquitto#3762 (apps/mosquitto_ctrl/ctrl_shell.c:830)",
        "desc": (
            "In C/C++, `calloc(num_workers, elem_bytes)` asks the operating system for a zeroed heap block. "
            "When memory is exhausted (`avail_bytes < num_workers * elem_bytes`), `calloc` returns `NULL` (in our Python model, `None`). "
            "The buggy starter code never checks for `NULL` and immediately indexes `workspace[i] = worker_addrs[i]`, crashing with a `SIGSEGV` (`TypeError` in Python). "
            "Fix `init_shm_workspace(num_workers, elem_bytes, avail_bytes, worker_addrs)` so that:\n"
            "1. If `num_workers <= 0` or `calloc` returns `NULL`/`None` (which happens when `num_workers * elem_bytes > avail_bytes`), it immediately returns `None` (`NULL` in C++).\n"
            "2. Otherwise, it copies each `worker_addrs[i]` into `workspace[i]` and returns `workspace`."
        ),
        "py_name": "init_shm_workspace",
        "py_starter": (
            "def sim_calloc(count, elem_bytes, avail_bytes):\n"
            "    total_bytes = count * elem_bytes\n"
            "    if count <= 0 or total_bytes > avail_bytes:\n"
            "        return None\n"
            "    return [0] * count\n\n"
            "def init_shm_workspace(num_workers, elem_bytes, avail_bytes, worker_addrs):\n"
            "    workspace = sim_calloc(num_workers, elem_bytes, avail_bytes)\n"
            "    # BUG (DeepSpeed shm.cpp:390): no NULL/None check before writing workspace[i]!\n"
            "    for i in range(num_workers):\n"
            "        workspace[i] = worker_addrs[i]\n"
            "    return workspace\n"
        ),
        "py_sol": (
            "def sim_calloc(count, elem_bytes, avail_bytes):\n"
            "    total_bytes = count * elem_bytes\n"
            "    if count <= 0 or total_bytes > avail_bytes:\n"
            "        return None\n"
            "    return [0] * count\n\n"
            "def init_shm_workspace(num_workers, elem_bytes, avail_bytes, worker_addrs):\n"
            "    workspace = sim_calloc(num_workers, elem_bytes, avail_bytes)\n"
            "    if workspace is None:\n"
            "        return None\n"
            "    for i in range(num_workers):\n"
            "        workspace[i] = worker_addrs[i]\n"
            "    return workspace\n"
        ),
        "cpp_name": "init_shm_workspace",
        "cpp_starter": (
            "// Simulated C++ heap allocator for in-browser execution\n"
            "int* sim_calloc(int count, int elem_bytes, int avail_bytes, int* out_buf) {\n"
            "    long long total_bytes = (long long)count * (long long)elem_bytes;\n"
            "    if (count <= 0 || total_bytes > (long long)avail_bytes) {\n"
            "        return NULL;\n"
            "    }\n"
            "    for (int k = 0; k < count; k++) { out_buf[k] = 0; }\n"
            "    return out_buf;\n"
            "}\n\n"
            "int init_shm_workspace(int num_workers, int elem_bytes, int avail_bytes, const int* worker_addrs, int* backing_buf) {\n"
            "    int* workspace = sim_calloc(num_workers, elem_bytes, avail_bytes, backing_buf);\n"
            "    // BUG (DeepSpeed shm.cpp:390): workspace is dereferenced without checking if (!workspace)!\n"
            "    for (int i = 0; i < num_workers; i++) {\n"
            "        workspace[i] = worker_addrs[i];\n"
            "    }\n"
            "    return 1; // 1 = success, -1 = NULL allocation failure\n"
            "}\n"
        ),
        "cpp_sol": (
            "// Simulated C++ heap allocator for in-browser execution\n"
            "int* sim_calloc(int count, int elem_bytes, int avail_bytes, int* out_buf) {\n"
            "    long long total_bytes = (long long)count * (long long)elem_bytes;\n"
            "    if (count <= 0 || total_bytes > (long long)avail_bytes) {\n"
            "        return NULL;\n"
            "    }\n"
            "    for (int k = 0; k < count; k++) { out_buf[k] = 0; }\n"
            "    return out_buf;\n"
            "}\n\n"
            "int init_shm_workspace(int num_workers, int elem_bytes, int avail_bytes, const int* worker_addrs, int* backing_buf) {\n"
            "    int* workspace = sim_calloc(num_workers, elem_bytes, avail_bytes, backing_buf);\n"
            "    if (workspace == NULL) {\n"
            "        return -1;\n"
            "    }\n"
            "    for (int i = 0; i < num_workers; i++) {\n"
            "        workspace[i] = worker_addrs[i];\n"
            "    }\n"
            "    return 1;\n"
            "}\n"
        ),
        "hint": "Check `if workspace is None: return None` (in Python) or `if (workspace == NULL) { return -1; }` (in C++) immediately after `sim_calloc(...)`, BEFORE the `for` loop.",
        "tests": [
            {"a": "[3, 8, 64, [101, 102, 103]]", "e": "[101, 102, 103]"},
            {"a": "[4, 8, 16, [10, 20, 30, 40]]", "e": "None"},
            {"a": "[0, 8, 64, []]", "e": "None"},
            {"a": "[2, 16, 32, [777, 888]]", "e": "[777, 888]", "h": 1},
            {"a": "[2, 16, 31, [777, 888]]", "e": "None", "h": 1},
        ],
        "v": [
            ["num_workers (int)", "How many distributed worker processes need a slot in the `workspace` pointer table (`size` in DeepSpeed's `shm_initialize`)."],
            ["elem_bytes (int)", "Byte size of a single slot in the table (in C++, `sizeof(struct allreduce_workspace*)`, which is 8 bytes on a 64-bit CPU)."],
            ["avail_bytes (int)", "How many free bytes remain in system heap memory right now. If `num_workers * elem_bytes > avail_bytes`, allocation fails."],
            ["worker_addrs (list / const int*)", "The shared-memory buffer addresses (`workspace_buf_other` in DeepSpeed) to store into each slot `workspace[i]`."],
            ["total_bytes (int / long long)", "Total bytes requested from the heap: `count * elem_bytes`. Cast to `long long` in C++ so the multiplication does not overflow 32-bit `int`."],
            ["workspace (list or None / int*)", "The pointer returned by `sim_calloc`. Holds a valid zeroed array on success, or `None` / `NULL` (address 0) when allocation fails."],
            ["i (int)", "Loop index stepping through each worker slot from `0` to `num_workers - 1`."],
        ],
        "lines_py": [
            ["def sim_calloc(count, elem_bytes, avail_bytes):", "Defines a helper that models C's `calloc(count, elem_bytes)` against a finite heap budget `avail_bytes`."],
            ["    total_bytes = count * elem_bytes", "Multiplies element count by per-element byte size to get the total heap reservation needed."],
            ["    if count <= 0 or total_bytes > avail_bytes:", "Checks if the request is invalid (`<= 0`) or exceeds remaining RAM (`> avail_bytes`)."],
            ["        return None", "Models C's `return NULL;`: hands back `None` (no memory block) instead of raising an exception."],
            ["    return [0] * count", "Models `calloc` zero-filling the newly allocated block of `count` slots."],
            ["def init_shm_workspace(num_workers, elem_bytes, avail_bytes, worker_addrs):", "Entry point mirroring `shm_initialize` in `deepspeed/csrc/cpu/comm/shm.cpp`."],
            ["    workspace = sim_calloc(num_workers, elem_bytes, avail_bytes)", "Calls the allocator. CRITICAL: `workspace` is now EITHER a list of `num_workers` zeros OR `None`!"],
            ["    if workspace is None:", "THE FIX: Guards against allocation failure before any `[i]` indexing can touch `workspace`."],
            ["        return None", "Cleanly aborts initialization and signals allocation failure to the caller instead of crashing."],
            ["    for i in range(num_workers):", "Safe to loop now because we proved `workspace` is not `None` and has `num_workers` slots."],
            ["        workspace[i] = worker_addrs[i]", "Writes each worker's shared-memory address into slot `i` (line 390 of `shm.cpp`). Without the guard above, this line crashes when `workspace` is `None`."],
            ["    return workspace", "Returns the populated workspace table to the caller."],
        ],
        "lines_cpp": [
            ["int* workspace = sim_calloc(num_workers, elem_bytes, avail_bytes, backing_buf);", "In C++, `calloc` returns a raw pointer `int*`. If the OS is out of memory, `workspace` is `NULL` (`0x0`)."],
            ["if (workspace == NULL) { return -1; }", "THE FIX (CWE-476 guard): Checks `workspace == NULL` (or `if (!workspace)`) and returns an error code `-1` before dereferencing."],
            ["for (int i = 0; i < num_workers; i++) {", "Loops `i` from `0` up to `num_workers - 1`."],
            ["    workspace[i] = worker_addrs[i];", "Pointer arithmetic `*(workspace + i) = worker_addrs[i]`. If `workspace` were `NULL` (`0x0`), this would write to address `0x0 + i*4`, triggering a fatal hardware `SIGSEGV`."],
            ["return 1;", "Returns `1` to indicate every slot was allocated and populated safely."],
        ],
    },

    # =========================================================================
    # LEVEL 1, EXERCISE 2: Compound Allocation Guard & Partial-OOM Cleanup
    # =========================================================================
    {
        "id": "compound_guard",
        "l": 1,
        "title": "1.2 Compound Multi-Buffer Allocation & Partial-OOM Cleanup (DeepSpeed #8565 Commit 2)",
        "real_bug": "deepspeedai/DeepSpeed#8565 (shm.cpp:375-384 — 5 sibling calloc() calls needing a compound guard + cleanup)",
        "desc": (
            "In `DeepSpeed#8565`, `shm_initialize` actually made 5 sibling `calloc()` calls in one block (`workspace`, `symmetric_buffer[0]`, `symmetric_buffer[1]`, etc.). "
            "When a function allocates multiple buffers in sequence (`buf_sizes = [sz0, sz1, ...]`) from a heap with `avail_bytes` free:\n"
            "- Allocation `i` succeeds if `sz_i > 0` and `used_bytes + sz_i <= avail_bytes` (consuming `sz_i` bytes from the heap).\n"
            "- If ANY allocation in `buf_sizes` fails (say buffer #2 exceeds remaining `avail_bytes`), you cannot just return early! Buffers #0 and #1 were already allocated and MUST be freed, or you leak them!\n"
            "Write `alloc_all_or_cleanup(buf_sizes, avail_bytes)` that returns `[ok, active_allocs, bytes_leaked]`:\n"
            "- If every buffer in `buf_sizes` fits within `avail_bytes` (and every `sz > 0`), return `[True, len(buf_sizes), 0]`.\n"
            "- If any buffer fails (`sz <= 0` or not enough `avail_bytes` left), free all previously allocated buffers in that call and return `[False, 0, 0]`. "
            "(The buggy starter forgets to free earlier buffers on partial failure, returning non-zero `bytes_leaked`!)"
        ),
        "py_name": "alloc_all_or_cleanup",
        "py_starter": (
            "def alloc_all_or_cleanup(buf_sizes, avail_bytes):\n"
            "    allocated = []\n"
            "    used_bytes = 0\n"
            "    for sz in buf_sizes:\n"
            "        if sz <= 0 or used_bytes + sz > avail_bytes:\n"
            "            # BUG: returns False on partial OOM without freeing earlier buffers in `allocated`!\n"
            "            return [False, len(allocated), used_bytes]\n"
            "        allocated.append(sz)\n"
            "        used_bytes += sz\n"
            "    return [True, len(allocated), 0]\n"
        ),
        "py_sol": (
            "def alloc_all_or_cleanup(buf_sizes, avail_bytes):\n"
            "    allocated = []\n"
            "    used_bytes = 0\n"
            "    for sz in buf_sizes:\n"
            "        if sz <= 0 or used_bytes + sz > avail_bytes:\n"
            "            while len(allocated) > 0:\n"
            "                freed_sz = allocated.pop()\n"
            "                used_bytes -= freed_sz\n"
            "            return [False, len(allocated), used_bytes]\n"
            "        allocated.append(sz)\n"
            "        used_bytes += sz\n"
            "    return [True, len(allocated), 0]\n"
        ),
        "cpp_name": "alloc_all_or_cleanup",
        "cpp_starter": (
            "void alloc_all_or_cleanup(const int* buf_sizes, int n_bufs, int avail_bytes, int* out_status) {\n"
            "    int active_count = 0;\n"
            "    int used_bytes = 0;\n"
            "    for (int i = 0; i < n_bufs; i++) {\n"
            "        int sz = buf_sizes[i];\n"
            "        if (sz <= 0 || used_bytes + sz > avail_bytes) {\n"
            "            // BUG: returns on partial OOM without freeing the `active_count` buffers already allocated!\n"
            "            out_status[0] = 0;\n"
            "            out_status[1] = active_count;\n"
            "            out_status[2] = used_bytes;\n"
            "            return;\n"
            "        }\n"
            "        active_count += 1;\n"
            "        used_bytes += sz;\n"
            "    }\n"
            "    out_status[0] = 1;\n"
            "    out_status[1] = active_count;\n"
            "    out_status[2] = 0;\n"
            "}\n"
        ),
        "cpp_sol": (
            "void alloc_all_or_cleanup(const int* buf_sizes, int n_bufs, int avail_bytes, int* out_status) {\n"
            "    int active_count = 0;\n"
            "    int used_bytes = 0;\n"
            "    for (int i = 0; i < n_bufs; i++) {\n"
            "        int sz = buf_sizes[i];\n"
            "        if (sz <= 0 || used_bytes + sz > avail_bytes) {\n"
            "            for (int j = 0; j < i; j++) {\n"
            "                active_count -= 1;\n"
            "                used_bytes -= buf_sizes[j];\n"
            "            }\n"
            "            out_status[0] = 0;\n"
            "            out_status[1] = active_count;\n"
            "            out_status[2] = used_bytes;\n"
            "            return;\n"
            "        }\n"
            "        active_count += 1;\n"
            "        used_bytes += sz;\n"
            "    }\n"
            "    out_status[0] = 1;\n"
            "    out_status[1] = active_count;\n"
            "    out_status[2] = 0;\n"
            "}\n"
        ),
        "hint": "Inside the failure `if` branch (`sz <= 0 or used_bytes + sz > avail_bytes`), unwind/clear `allocated` and subtract each freed size from `used_bytes` (so `len(allocated)` and `used_bytes` both become `0`) before returning `[False, len(allocated), used_bytes]`.",
        "tests": [
            {"a": "[[16, 16, 16], 64]", "e": "[True, 3, 0]"},
            {"a": "[[16, 16, 32], 40]", "e": "[False, 0, 0]"},
            {"a": "[[32, 0, 16], 64]", "e": "[False, 0, 0]"},
            {"a": "[[10, 20, 30, 40], 100]", "e": "[True, 4, 0]", "h": 1},
            {"a": "[[10, 20, 30, 40], 99]", "e": "[False, 0, 0]", "h": 1},
        ],
        "v": [
            ["buf_sizes (list[int] / const int*)", "Byte sizes of the sibling buffers that must all be allocated together (like `workspace`, `symmetric_buffer[0..1]`, `ernie_buffer[0..1]` in DeepSpeed)."],
            ["avail_bytes (int)", "Total free heap bytes available before the first allocation starts."],
            ["allocated (list[int]) / active_count (int)", "Tracks each buffer that has already been allocated so far during this call."],
            ["used_bytes (int)", "Running sum of heap bytes currently held by `allocated` buffers."],
            ["sz (int)", "Size in bytes of the current buffer being requested in the loop."],
            ["freed_sz (int)", "Size of a previously allocated buffer being released during partial-OOM rollback."],
        ],
        "lines_py": [
            ["def alloc_all_or_cleanup(buf_sizes, avail_bytes):", "Function that must either allocate every buffer in `buf_sizes` or leave zero leaked bytes behind."],
            ["    allocated = []", "List holding the sizes of buffers currently live on the heap."],
            ["    used_bytes = 0", "Tracks total heap bytes consumed by `allocated` so far."],
            ["    for sz in buf_sizes:", "Attempts to allocate each sibling buffer in order."],
            ["        if sz <= 0 or used_bytes + sz > avail_bytes:", "Detects when the current buffer request `sz` is invalid or overflows remaining heap `avail_bytes - used_bytes`."],
            ["            while len(allocated) > 0:", "THE FIX: Loops over all earlier buffers that succeeded before this failure."],
            ["                freed_sz = allocated.pop()", "Models `free(buf_j)`: removes the buffer from active allocations."],
            ["                used_bytes -= freed_sz", "Returns `freed_sz` bytes back to the heap so `used_bytes` drops back to `0`."],
            ["            return [False, len(allocated), used_bytes]", "Returns `[False, 0, 0]`: call failed cleanly with 0 active buffers and 0 leaked bytes."],
            ["        allocated.append(sz)", "Records that this buffer succeeded and is now live on the heap."],
            ["        used_bytes += sz", "Deducts `sz` from available heap capacity."],
            ["    return [True, len(allocated), 0]", "All sibling allocations succeeded; caller now owns all `len(allocated)` buffers."],
        ],
        "lines_cpp": [
            ["if (sz <= 0 || used_bytes + sz > avail_bytes) {", "Checks if allocation `i` fails (`calloc` returned `NULL`)."],
            ["    for (int j = 0; j < i; j++) {", "THE FIX: Walks over every prior buffer `0 .. i-1` that already succeeded before `i` failed."],
            ["        active_count -= 1; used_bytes -= buf_sizes[j];", "Models calling `free(buf[j])` on each prior buffer so zero heap bytes leak on error return."],
            ["    out_status[0] = 0; out_status[1] = active_count; out_status[2] = used_bytes; return;", "Writes `[0 (False), 0, 0]` and returns cleanly."],
        ],
    },

    # =========================================================================
    # LEVEL 2, EXERCISE 1: Pointer Overwrite Memory Leak (CWE-401)
    # =========================================================================
    {
        "id": "reassigned_leak",
        "l": 2,
        "title": "2.1 Pointer Reassigned Before free() (DeepSpeed #8565 Commit 1)",
        "real_bug": "deepspeedai/DeepSpeed#8565 (csrc/cpu/comm/shm.cpp:362-365 — workspace_buf malloc overwritten by mmap pointer)",
        "desc": (
            "In `deepspeedai/DeepSpeed#8565` (`csrc/cpu/comm/shm.cpp:362-365`), `shm_initialize` did:\n"
            "```cpp\n"
            "workspace_buf = (struct allreduce_workspace*)malloc(sizeof(struct allreduce_workspace)); // line 362\n"
            "shared_create(&allreduce_buffer, ..., workspace_buf, ...); // line 364: writes bytes to shm fd\n"
            "workspace_buf = (struct allreduce_workspace*)allreduce_buffer.bytes; // line 365: OVERWRITES pointer!\n"
            "```\n"
            "Because `workspace_buf` was reassigned at line 365 without calling `free(workspace_buf)` first, the heap block from line 362 was permanently leaked! "
            "In `setup_shared_workspace(init_val, shm_mapped_val, heap_bytes_used, block_bytes)`, we allocate a temporary staging block of `block_bytes` (adding `block_bytes` to `heap_bytes_used`), copy `init_val` into shared memory, and then point `workspace_buf` at the shared-memory value `shm_mapped_val`. "
            "Fix the function so the temporary heap staging block is freed (`heap_bytes_used` decremented by `block_bytes`) BEFORE `workspace_buf` is reassigned to `shm_mapped_val`, returning `[workspace_buf, synced_to_shm, heap_bytes_used]`."
        ),
        "py_name": "setup_shared_workspace",
        "py_starter": (
            "def setup_shared_workspace(init_val, shm_mapped_val, heap_bytes_used, block_bytes):\n"
            "    # Line 362: malloc temporary staging block on the heap\n"
            "    workspace_buf = init_val\n"
            "    heap_bytes_used += block_bytes\n"
            "    # Line 364: shared_create copies staging block contents into shared memory\n"
            "    synced_to_shm = workspace_buf\n"
            "    # BUG (Line 365 of shm.cpp): reassigns workspace_buf without freeing the staging block first!\n"
            "    workspace_buf = shm_mapped_val\n"
            "    return [workspace_buf, synced_to_shm, heap_bytes_used]\n"
        ),
        "py_sol": (
            "def setup_shared_workspace(init_val, shm_mapped_val, heap_bytes_used, block_bytes):\n"
            "    # Line 362: malloc temporary staging block on the heap\n"
            "    workspace_buf = init_val\n"
            "    heap_bytes_used += block_bytes\n"
            "    # Line 364: shared_create copies staging block contents into shared memory\n"
            "    synced_to_shm = workspace_buf\n"
            "    # FIX: free(workspace_buf) before reassigning workspace_buf to the shared-memory mapping!\n"
            "    heap_bytes_used -= block_bytes\n"
            "    workspace_buf = shm_mapped_val\n"
            "    return [workspace_buf, synced_to_shm, heap_bytes_used]\n"
        ),
        "cpp_name": "setup_shared_workspace",
        "cpp_starter": (
            "void setup_shared_workspace(int init_val, int shm_mapped_val, int heap_bytes_used, int block_bytes, int* out_res) {\n"
            "    int workspace_buf = init_val;\n"
            "    heap_bytes_used += block_bytes; // malloc(sizeof(allreduce_workspace))\n"
            "    int synced_to_shm = workspace_buf; // shared_create(..., workspace_buf, ...)\n"
            "    // BUG (shm.cpp:365): overwrites workspace_buf without free(workspace_buf)!\n"
            "    workspace_buf = shm_mapped_val;\n"
            "    out_res[0] = workspace_buf;\n"
            "    out_res[1] = synced_to_shm;\n"
            "    out_res[2] = heap_bytes_used;\n"
            "}\n"
        ),
        "cpp_sol": (
            "void setup_shared_workspace(int init_val, int shm_mapped_val, int heap_bytes_used, int block_bytes, int* out_res) {\n"
            "    int workspace_buf = init_val;\n"
            "    heap_bytes_used += block_bytes; // malloc(sizeof(allreduce_workspace))\n"
            "    int synced_to_shm = workspace_buf; // shared_create(..., workspace_buf, ...)\n"
            "    // FIX: free(workspace_buf) right after shared_create, before reassigning workspace_buf\n"
            "    heap_bytes_used -= block_bytes;\n"
            "    workspace_buf = shm_mapped_val;\n"
            "    out_res[0] = workspace_buf;\n"
            "    out_res[1] = synced_to_shm;\n"
            "    out_res[2] = heap_bytes_used;\n"
            "}\n"
        ),
        "hint": "Right after `synced_to_shm = workspace_buf` (and before `workspace_buf = shm_mapped_val`), free the temporary heap staging block by subtracting `block_bytes`: `heap_bytes_used -= block_bytes`.",
        "tests": [
            {"a": "[42, 99, 0, 128]", "e": "[99, 42, 0]"},
            {"a": "[7, 500, 1024, 256]", "e": "[500, 7, 1024]"},
            {"a": "[1, 2, 64, 64]", "e": "[2, 1, 64]", "h": 1},
        ],
        "v": [
            ["init_val (int)", "Initial value written into the temporary heap staging buffer (`workspace_buf` at line 362 of `shm.cpp`)."],
            ["shm_mapped_val (int)", "Value living inside the `mmap`-ed shared memory buffer (`allreduce_buffer.bytes` at line 365 of `shm.cpp`)."],
            ["heap_bytes_used (int)", "Total heap bytes currently allocated by the process before and after this function runs."],
            ["block_bytes (int)", "Size in bytes of `struct allreduce_workspace` allocated by `malloc`."],
            ["workspace_buf (int)", "The pointer variable that first holds the `malloc` staging buffer and is then reassigned to point at `allreduce_buffer.bytes`."],
            ["synced_to_shm (int)", "Captures the staging value that `shared_create` wrote into the shared-memory file descriptor."],
        ],
        "lines_py": [
            ["def setup_shared_workspace(init_val, shm_mapped_val, heap_bytes_used, block_bytes):", "Models `shm_initialize` in `deepspeed/csrc/cpu/comm/shm.cpp`."],
            ["    workspace_buf = init_val", "Line 362: `workspace_buf = (struct allreduce_workspace*)malloc(...)`."],
            ["    heap_bytes_used += block_bytes", "The heap allocator reserves `block_bytes` for the temporary staging buffer."],
            ["    synced_to_shm = workspace_buf", "Line 364: `shared_create` copies the bytes from `workspace_buf` into the shared-memory file descriptor."],
            ["    heap_bytes_used -= block_bytes", "THE FIX (DeepSpeed #8565): `free(workspace_buf);` releases the temporary heap block now that `shared_create` is done with it."],
            ["    workspace_buf = shm_mapped_val", "Line 365: safely repoints `workspace_buf` to the `mmap` shared-memory address without leaking the old heap block."],
            ["    return [workspace_buf, synced_to_shm, heap_bytes_used]", "Returns the active `mmap` value, the synced staging value, and net `heap_bytes_used` (unchanged!)."],
        ],
        "lines_cpp": [
            ["int workspace_buf = init_val; heap_bytes_used += block_bytes;", "Allocates the initial staging struct on the heap via `malloc`."],
            ["int synced_to_shm = workspace_buf;", "Passes `workspace_buf` into `shared_create`, which `write()`s its contents to the shm file descriptor."],
            ["heap_bytes_used -= block_bytes;", "THE FIX: `free(workspace_buf);` before `workspace_buf` is overwritten on the next line."],
            ["workspace_buf = shm_mapped_val;", "Reassigns `workspace_buf = (struct allreduce_workspace*)allreduce_buffer.bytes;`."],
        ],
    },

    # =========================================================================
    # LEVEL 2, EXERCISE 2: Reusable Workspace vs. Per-Step Loop Leak
    # =========================================================================
    {
        "id": "loop_scratch_leak",
        "l": 2,
        "title": "2.2 Per-Step Scratch Allocation Leak vs. Reusable Workspace Growth",
        "real_bug": "Reagent C05 loop-reassignment leak pattern & inference workspace pooling",
        "desc": (
            "During autoregressive LLM decoding, each step `i` requests a scratch buffer of `step_bytes[i]` bytes. "
            "The buggy starter allocates a brand-new buffer on every step and overwrites `workspace_cap = req` without freeing the previous step's buffer, accumulating `total_allocated += req` on every single token! "
            "Fix `run_step_workspace(step_bytes)` to implement a **reusable high-water-mark workspace**:\n"
            "- Maintain `workspace_cap` (initially `0`) and `live_bytes` (initially `0`, the currently allocated workspace size) and `realloc_count` (initially `0`).\n"
            "- On each step `req` in `step_bytes`: if `req > workspace_cap`, free the old workspace (if any), allocate a new one of size `req` (so `workspace_cap = req`, `live_bytes = req`, and `realloc_count += 1`). If `req <= workspace_cap`, reuse the existing workspace without allocating (`realloc_count` unchanged, `live_bytes` stays `workspace_cap`).\n"
            "- Return `[live_bytes, realloc_count]` after all steps finish."
        ),
        "py_name": "run_step_workspace",
        "py_starter": (
            "def run_step_workspace(step_bytes):\n"
            "    workspace_cap = 0\n"
            "    live_bytes = 0\n"
            "    realloc_count = 0\n"
            "    for req in step_bytes:\n"
            "        # BUG: allocates a new buffer every step and leaks the old one!\n"
            "        workspace_cap = req\n"
            "        live_bytes += req\n"
            "        realloc_count += 1\n"
            "    return [live_bytes, realloc_count]\n"
        ),
        "py_sol": (
            "def run_step_workspace(step_bytes):\n"
            "    workspace_cap = 0\n"
            "    live_bytes = 0\n"
            "    realloc_count = 0\n"
            "    for req in step_bytes:\n"
            "        if req > workspace_cap:\n"
            "            live_bytes -= workspace_cap\n"
            "            workspace_cap = req\n"
            "            live_bytes += workspace_cap\n"
            "            realloc_count += 1\n"
            "    return [live_bytes, realloc_count]\n"
        ),
        "cpp_name": "run_step_workspace",
        "cpp_starter": (
            "void run_step_workspace(const int* step_bytes, int n_steps, int* out_res) {\n"
            "    int workspace_cap = 0;\n"
            "    int live_bytes = 0;\n"
            "    int realloc_count = 0;\n"
            "    for (int i = 0; i < n_steps; i++) {\n"
            "        int req = step_bytes[i];\n"
            "        // BUG: allocates on every step without freeing or reusing workspace!\n"
            "        workspace_cap = req;\n"
            "        live_bytes += req;\n"
            "        realloc_count += 1;\n"
            "    }\n"
            "    out_res[0] = live_bytes;\n"
            "    out_res[1] = realloc_count;\n"
            "}\n"
        ),
        "cpp_sol": (
            "void run_step_workspace(const int* step_bytes, int n_steps, int* out_res) {\n"
            "    int workspace_cap = 0;\n"
            "    int live_bytes = 0;\n"
            "    int realloc_count = 0;\n"
            "    for (int i = 0; i < n_steps; i++) {\n"
            "        int req = step_bytes[i];\n"
            "        if (req > workspace_cap) {\n"
            "            live_bytes -= workspace_cap;\n"
            "            workspace_cap = req;\n"
            "            live_bytes += workspace_cap;\n"
            "            realloc_count += 1;\n"
            "        }\n"
            "    }\n"
            "    out_res[0] = live_bytes;\n"
            "    out_res[1] = realloc_count;\n"
            "}\n"
        ),
        "hint": "Only reallocate when `req > workspace_cap`. Inside `if req > workspace_cap:`, subtract the old `workspace_cap` from `live_bytes` (freeing the old block), set `workspace_cap = req`, add the new `workspace_cap` to `live_bytes`, and increment `realloc_count`.",
        "tests": [
            {"a": "[[128, 128, 128, 128]]", "e": "[128, 1]"},
            {"a": "[[64, 128, 96, 256, 128]]", "e": "[256, 3]"},
            {"a": "[[]]", "e": "[0, 0]"},
            {"a": "[[512, 256, 128]]", "e": "[512, 1]", "h": 1},
        ],
        "v": [
            ["step_bytes (list[int] / const int*)", "Scratch buffer byte sizes needed on each inference step `i`."],
            ["workspace_cap (int)", "Current capacity (in bytes) of the reusable scratch buffer."],
            ["live_bytes (int)", "Total heap bytes currently held by our process (must equal `workspace_cap`, never the sum of all steps!)."],
            ["realloc_count (int)", "How many times we had to call `free` + `malloc` to grow the workspace."],
            ["req (int)", "Scratch bytes required by the current step."],
        ],
        "lines_py": [
            ["def run_step_workspace(step_bytes):", "Simulates scratch workspace management across `len(step_bytes)` inference steps."],
            ["    workspace_cap = 0", "Starts with no workspace allocated yet (`capacity = 0`)."],
            ["    live_bytes = 0", "Starts with `0` heap bytes in use."],
            ["    realloc_count = 0", "Counts how many actual heap allocations occur."],
            ["    for req in step_bytes:", "Visits each step's scratch size requirement `req`."],
            ["        if req > workspace_cap:", "THE FIX: Only reallocate when the current workspace is too small (`req > workspace_cap`)."],
            ["            live_bytes -= workspace_cap", "Frees the previous workspace buffer before replacing the pointer."],
            ["            workspace_cap = req", "Updates workspace capacity to the new high-water mark `req`."],
            ["            live_bytes += workspace_cap", "Allocates the new `workspace_cap`-byte buffer."],
            ["            realloc_count += 1", "Records one growth allocation (steps with `req <= workspace_cap` do zero allocations!)."],
            ["    return [live_bytes, realloc_count]", "Returns `[live_bytes, realloc_count]`."],
        ],
        "lines_cpp": [
            ["if (req > workspace_cap) {", "Checks whether the existing `workspace` buffer is smaller than `req`."],
            ["    live_bytes -= workspace_cap;", "Calls `free(workspace)` on the old buffer so it does not leak when `workspace` is reassigned."],
            ["    workspace_cap = req; live_bytes += workspace_cap; realloc_count += 1;", "Allocates the new larger workspace and increments `realloc_count`."],
        ],
    },

    # =========================================================================
    # LEVEL 3, EXERCISE 1: Missing return After Early-Exit free() (CFEngine #6358)
    # =========================================================================
    {
        "id": "early_exit_uaf",
        "l": 3,
        "title": "3.1 Missing return After Early-Exit free() -> UAF + Double-Free (CFEngine #6358)",
        "real_bug": "cfengine/core#6358 (libpromises/dbm_test_api.c — RemoveFilament missing return after early-exit free)",
        "desc": (
            "In `cfengine/core#6358` (`libpromises/dbm_test_api.c`), `RemoveFilament` had an early-exit check for an empty filament:\n"
            "```c\n"
            "if (StringSetSize(filament->items) == 0) {\n"
            "    StringSetDestroy(filament->items);\n"
            "    free(filament);\n"
            "    // BUG: missing `return;` here!\n"
            "}\n"
            "bool success = OpenDB(&db, filament->db_id); // USE-AFTER-FREE read of filament->db_id!\n"
            "...\n"
            "StringSetDestroy(filament->items);\n"
            "free(filament); // DOUBLE-FREE of filament and filament->items!\n"
            "```\n"
            "Write `remove_filament(item_count, db_id)` that tracks `[freed_count, uaf_reads, db_opened]`:\n"
            "- When `item_count == 0`, destroy/free the filament (`freed_count += 1`) and **return immediately** `[freed_count, uaf_reads, db_opened]` (which is `[1, 0, 0]`).\n"
            "- When `item_count > 0`, open the DB (if `freed_count > 0`, increment `uaf_reads += 1`; set `db_opened = db_id`), then free the filament (`freed_count += 1`) and return `[freed_count, uaf_reads, db_opened]`."
        ),
        "py_name": "remove_filament",
        "py_starter": (
            "def remove_filament(item_count, db_id):\n"
            "    freed_count = 0\n"
            "    uaf_reads = 0\n"
            "    db_opened = 0\n"
            "    if item_count == 0:\n"
            "        freed_count += 1\n"
            "        # BUG (cfengine/core#6358): missing return after early-exit free!\n"
            "    if freed_count > 0:\n"
            "        uaf_reads += 1\n"
            "    db_opened = db_id\n"
            "    freed_count += 1\n"
            "    return [freed_count, uaf_reads, db_opened]\n"
        ),
        "py_sol": (
            "def remove_filament(item_count, db_id):\n"
            "    freed_count = 0\n"
            "    uaf_reads = 0\n"
            "    db_opened = 0\n"
            "    if item_count == 0:\n"
            "        freed_count += 1\n"
            "        return [freed_count, uaf_reads, db_opened]\n"
            "    if freed_count > 0:\n"
            "        uaf_reads += 1\n"
            "    db_opened = db_id\n"
            "    freed_count += 1\n"
            "    return [freed_count, uaf_reads, db_opened]\n"
        ),
        "cpp_name": "remove_filament",
        "cpp_starter": (
            "void remove_filament(int item_count, int db_id, int* out_res) {\n"
            "    int freed_count = 0;\n"
            "    int uaf_reads = 0;\n"
            "    int db_opened = 0;\n"
            "    if (item_count == 0) {\n"
            "        freed_count += 1;\n"
            "        // BUG (cfengine/core#6358): missing return after early-exit free!\n"
            "    }\n"
            "    if (freed_count > 0) {\n"
            "        uaf_reads += 1;\n"
            "    }\n"
            "    db_opened = db_id;\n"
            "    freed_count += 1;\n"
            "    out_res[0] = freed_count;\n"
            "    out_res[1] = uaf_reads;\n"
            "    out_res[2] = db_opened;\n"
            "}\n"
        ),
        "cpp_sol": (
            "void remove_filament(int item_count, int db_id, int* out_res) {\n"
            "    int freed_count = 0;\n"
            "    int uaf_reads = 0;\n"
            "    int db_opened = 0;\n"
            "    if (item_count == 0) {\n"
            "        freed_count += 1;\n"
            "        out_res[0] = freed_count;\n"
            "        out_res[1] = uaf_reads;\n"
            "        out_res[2] = db_opened;\n"
            "        return;\n"
            "    }\n"
            "    if (freed_count > 0) {\n"
            "        uaf_reads += 1;\n"
            "    }\n"
            "    db_opened = db_id;\n"
            "    freed_count += 1;\n"
            "    out_res[0] = freed_count;\n"
            "    out_res[1] = uaf_reads;\n"
            "    out_res[2] = db_opened;\n"
            "}\n"
        ),
        "hint": "Inside `if item_count == 0:`, right after `freed_count += 1`, return `[freed_count, uaf_reads, db_opened]` (or write `out_res` and `return;` in C++) so execution does not fall through to the rest of the function.",
        "tests": [
            {"a": "[0, 7]", "e": "[1, 0, 0]"},
            {"a": "[5, 7]", "e": "[1, 0, 7]"},
            {"a": "[0, 99]", "e": "[1, 0, 0]", "h": 1},
            {"a": "[1, 42]", "e": "[1, 0, 42]", "h": 1},
        ],
        "v": [
            ["item_count (int)", "Number of items in `filament->items` (`StringSetSize(filament->items)` in CFEngine)."],
            ["db_id (int)", "Database identifier field `filament->db_id` stored inside the heap struct `filament`."],
            ["freed_count (int)", "How many times `free(filament)` has been called (`1` = clean single free; `2` = CWE-415 double-free!)."],
            ["uaf_reads (int)", "How many times `filament->db_id` was read AFTER `filament` was already freed (`> 0` = CWE-416 use-after-free!)."],
            ["db_opened (int)", "Which `db_id` was passed to `OpenDB` (`0` if the empty filament exited early as intended)."],
        ],
        "lines_py": [
            ["def remove_filament(item_count, db_id):", "Models `RemoveFilament` from `cfengine/core`'s `libpromises/dbm_test_api.c`."],
            ["    freed_count = 0", "Tracks how many times `free(filament)` has executed."],
            ["    uaf_reads = 0", "Tracks any read of `filament->db_id` after `free(filament)`."],
            ["    db_opened = 0", "Records the DB opened on the non-empty path."],
            ["    if item_count == 0:", "Early-exit branch when `StringSetSize(filament->items) == 0`."],
            ["        freed_count += 1", "Calls `StringSetDestroy(filament->items); free(filament);`."],
            ["        return [freed_count, uaf_reads, db_opened]", "THE ONE-LINE FIX (`cfengine/core#6358`): returns immediately so execution cannot fall through to `OpenDB` and the second `free(filament)`!"],
            ["    if freed_count > 0:", "Detects if `filament` was already freed before reaching `OpenDB(&db, filament->db_id)`."],
            ["        uaf_reads += 1", "Flags the Use-After-Free read that happened when `return` was missing."],
            ["    db_opened = db_id", "Opens `filament->db_id` on the normal `item_count > 0` path."],
            ["    freed_count += 1", "Normal end-of-function `free(filament)` (which became a Double-Free when the early-exit branch fell through!)."],
            ["    return [freed_count, uaf_reads, db_opened]", "Returns `[1, 0, db_id]` on the normal path and `[1, 0, 0]` on the empty path."],
        ],
        "lines_cpp": [
            ["if (item_count == 0) {", "Checks if the filament has zero items (`StringSetSize(filament->items) == 0`)."],
            ["    freed_count += 1;", "Frees the struct on the early-exit branch."],
            ["    out_res[0] = freed_count; out_res[1] = uaf_reads; out_res[2] = db_opened; return;", "THE FIX: Exits the function immediately after freeing on the early-exit branch."],
        ],
    },

    # =========================================================================
    # LEVEL 3, EXERCISE 2: Dangling Tensor View Across Storage Resize
    # =========================================================================
    {
        "id": "tensor_view_uaf",
        "l": 3,
        "title": "3.2 Dangling Tensor View Pointer After Backing Storage Reallocation",
        "real_bug": "PyTorch C++ TensorImpl / Storage resize_() dangling data_ptr view hazard",
        "desc": (
            "In C++ tensor libraries (like ATen / PyTorch), a tensor `view` does not own its own array; it stores a pointer to the backing `Storage` buffer plus an `offset` and `length`. "
            "When `storage` is resized to `new_total_len` (because a cache or tensor grew), the allocator allocates a brand-new buffer of length `new_total_len` (copying old values and zero-filling new slots) and **frees the old buffer**! "
            "If `view` still reads from its saved `old_storage_ref` instead of rebinding to the newly reallocated `storage`, reading the view accesses freed memory (or stale values written before an update to the new storage). "
            "Fix `resize_and_read_view(init_data, new_total_len, write_idx, write_val, view_offset, view_len)` so that after `storage` is reallocated to `new_total_len` and `storage[write_idx] = write_val` is written, the slice is read from the **new** `storage` at `[view_offset : view_offset + view_len]` (and not from the freed `old_storage`!)."
        ),
        "py_name": "resize_and_read_view",
        "py_starter": (
            "def resize_and_read_view(init_data, new_total_len, write_idx, write_val, view_offset, view_len):\n"
            "    storage = list(init_data)\n"
            "    # A view caches a reference to the backing storage buffer BEFORE resize\n"
            "    view_backing = storage\n"
            "    if new_total_len > len(storage):\n"
            "        # Reallocates a new buffer and frees the old one (marked with -999999 poison)\n"
            "        new_storage = storage + [0] * (new_total_len - len(storage))\n"
            "        for k in range(len(storage)):\n"
            "            storage[k] = -999999  # ASan poison on freed old storage!\n"
            "        storage = new_storage\n"
            "    storage[write_idx] = write_val\n"
            "    # BUG: view_backing still points at the old freed (poisoned) storage buffer!\n"
            "    return view_backing[view_offset : view_offset + view_len]\n"
        ),
        "py_sol": (
            "def resize_and_read_view(init_data, new_total_len, write_idx, write_val, view_offset, view_len):\n"
            "    storage = list(init_data)\n"
            "    view_backing = storage\n"
            "    if new_total_len > len(storage):\n"
            "        new_storage = storage + [0] * (new_total_len - len(storage))\n"
            "        for k in range(len(storage)):\n"
            "            storage[k] = -999999  # ASan poison on freed old storage!\n"
            "        storage = new_storage\n"
            "        view_backing = storage\n"
            "    storage[write_idx] = write_val\n"
            "    return view_backing[view_offset : view_offset + view_len]\n"
        ),
        "cpp_name": "resize_and_read_view",
        "cpp_starter": (
            "void resize_and_read_view(const int* init_data, int init_len, int new_total_len, int write_idx, int write_val, int view_offset, int view_len, int* out_slice) {\n"
            "    int old_buf[64];\n"
            "    int new_buf[64];\n"
            "    for (int i = 0; i < init_len; i++) { old_buf[i] = init_data[i]; }\n"
            "    int* storage = old_buf;\n"
            "    int* view_backing = storage; // raw data_ptr cached before resize\n"
            "    if (new_total_len > init_len) {\n"
            "        for (int i = 0; i < init_len; i++) { new_buf[i] = storage[i]; }\n"
            "        for (int i = init_len; i < new_total_len; i++) { new_buf[i] = 0; }\n"
            "        for (int i = 0; i < init_len; i++) { old_buf[i] = -999999; } // ASan poison freed old_buf\n"
            "        storage = new_buf;\n"
            "        // BUG: view_backing still points to freed old_buf!\n"
            "    }\n"
            "    storage[write_idx] = write_val;\n"
            "    for (int j = 0; j < view_len; j++) {\n"
            "        out_slice[j] = view_backing[view_offset + j];\n"
            "    }\n"
            "}\n"
        ),
        "cpp_sol": (
            "void resize_and_read_view(const int* init_data, int init_len, int new_total_len, int write_idx, int write_val, int view_offset, int view_len, int* out_slice) {\n"
            "    int old_buf[64];\n"
            "    int new_buf[64];\n"
            "    for (int i = 0; i < init_len; i++) { old_buf[i] = init_data[i]; }\n"
            "    int* storage = old_buf;\n"
            "    int* view_backing = storage;\n"
            "    if (new_total_len > init_len) {\n"
            "        for (int i = 0; i < init_len; i++) { new_buf[i] = storage[i]; }\n"
            "        for (int i = init_len; i < new_total_len; i++) { new_buf[i] = 0; }\n"
            "        for (int i = 0; i < init_len; i++) { old_buf[i] = -999999; }\n"
            "        storage = new_buf;\n"
            "        view_backing = storage;\n"
            "    }\n"
            "    storage[write_idx] = write_val;\n"
            "    for (int j = 0; j < view_len; j++) {\n"
            "        out_slice[j] = view_backing[view_offset + j];\n"
            "    }\n"
            "}\n"
        ),
        "hint": "Whenever `storage` is reallocated (`storage = new_storage` / `storage = new_buf`), rebind `view_backing = storage` so the view does not point to the poisoned/freed old buffer.",
        "tests": [
            {"a": "[[10, 20], 4, 2, 30, 0, 3]", "e": "[10, 20, 30]"},
            {"a": "[[5, 6, 7], 3, 1, 99, 0, 3]", "e": "[5, 99, 7]"},
            {"a": "[[1, 2], 6, 5, 42, 1, 5]", "e": "[2, 0, 0, 0, 42]", "h": 1},
        ],
        "v": [
            ["init_data (list[int] / const int*)", "Initial elements inside the tensor's backing storage before `resize_`."],
            ["new_total_len (int)", "New capacity requested for `storage`. If larger than `len(init_data)`, a reallocation occurs."],
            ["write_idx (int)", "Element index updated in `storage` after the resize."],
            ["write_val (int)", "New value written to `storage[write_idx]`."],
            ["view_offset (int)", "Starting element index of the tensor view (`storage_offset()`)."],
            ["view_len (int)", "Number of elements visible through the tensor view."],
            ["storage (list[int] / int*)", "Current backing buffer owned by the tensor's `StorageImpl`."],
            ["view_backing (list[int] / int*)", "Raw pointer (`data_ptr`) held by the view. Becomes a dangling pointer if `storage` reallocates without rebinding `view_backing`!"],
        ],
        "lines_py": [
            ["def resize_and_read_view(init_data, new_total_len, write_idx, write_val, view_offset, view_len):", "Simulates a tensor view across a storage reallocation (`resize_`)."],
            ["    storage = list(init_data)", "Allocates the initial backing storage array."],
            ["    view_backing = storage", "The view caches a raw pointer to `storage` (like `float *p = tensor.data_ptr<float>()`)."],
            ["    if new_total_len > len(storage):", "Checks if `resize_` needs more capacity than the current block holds."],
            ["        new_storage = storage + [0] * (new_total_len - len(storage))", "Allocates a new larger block and copies existing elements over."],
            ["        for k in range(len(storage)): storage[k] = -999999", "Models `free(old_storage)` + AddressSanitizer poisoning (`-999999`) of the freed memory."],
            ["        storage = new_storage", "Updates the storage owner to point at the newly allocated block."],
            ["        view_backing = storage", "THE FIX: Rebinds `view_backing` to the new `storage` so it does not read freed memory!"],
            ["    storage[write_idx] = write_val", "Writes `write_val` into the active storage buffer."],
            ["    return view_backing[view_offset : view_offset + view_len]", "Reads the view slice safely from the live buffer."],
        ],
        "lines_cpp": [
            ["int* view_backing = storage;", "Caches the raw pointer before `resize_`."],
            ["for (int i = 0; i < init_len; i++) { old_buf[i] = -999999; }", "Poisons the old buffer on free, just as ASan (`-fsanitize=address`) does."],
            ["storage = new_buf; view_backing = storage;", "THE FIX: Re-queries `tensor.data_ptr()` (`view_backing = storage`) after reallocation."],
        ],
    },

    # =========================================================================
    # LEVEL 4, EXERCISE 1: Integer Overflow in Allocation Size (IO2BO, CWE-190)
    # =========================================================================
    {
        "id": "alloc_overflow",
        "l": 4,
        "title": "4.1 Integer Overflow in Allocation Byte Size (Reagent C06 IO2BO)",
        "real_bug": "Reagent C06 (c06_alloc_size_overflow/alloc_size_overflow.py — CWE-190/680 malloc(n * sizeof(T)) wraparound)",
        "desc": (
            "In 32-bit unsigned arithmetic (or 32-bit GPU index math), integers wrap around modulo $2^{32} = 4,294,967,296$ (`UINT32_MAX = 4,294,967,295`). "
            "Look at what the buggy starter does when `n_elems = 1,073,741,825` and `elem_bytes = 4`:\n"
            "`(1073741825 * 4) & 0xFFFFFFFF` wraps around to just **`4` bytes**! "
            "So the allocator thinks you only asked for `4` bytes (which is `<= budget_bytes`), allocates a tiny 4-byte buffer, and then the caller writes 1 billion floats past the end of the buffer (Heap Buffer Overflow, CWE-787)! "
            "Fix `safe_tensor_alloc_bytes(n_elems, elem_bytes, uint32_max, budget_bytes)` using the exact division guard from your Reagent `C06` detector:\n"
            "- If `n_elems < 0` or `elem_bytes <= 0`, return `-1`.\n"
            "- Check `if n_elems > uint32_max // elem_bytes:` BEFORE multiplying! If it would overflow `uint32_max`, return `-1`.\n"
            "- Compute `total_bytes = n_elems * elem_bytes`. If `total_bytes > budget_bytes`, return `-1`.\n"
            "- Otherwise return `total_bytes`."
        ),
        "py_name": "safe_tensor_alloc_bytes",
        "py_starter": (
            "def safe_tensor_alloc_bytes(n_elems, elem_bytes, uint32_max, budget_bytes):\n"
            "    if n_elems < 0 or elem_bytes <= 0:\n"
            "        return -1\n"
            "    # BUG (CWE-190 IO2BO): 32-bit multiplication wraps around BEFORE checking budget_bytes!\n"
            "    total_bytes = (n_elems * elem_bytes) & uint32_max\n"
            "    if total_bytes > budget_bytes:\n"
            "        return -1\n"
            "    return total_bytes\n"
        ),
        "py_sol": (
            "def safe_tensor_alloc_bytes(n_elems, elem_bytes, uint32_max, budget_bytes):\n"
            "    if n_elems < 0 or elem_bytes <= 0:\n"
            "        return -1\n"
            "    if n_elems > uint32_max // elem_bytes:\n"
            "        return -1\n"
            "    total_bytes = n_elems * elem_bytes\n"
            "    if total_bytes > budget_bytes:\n"
            "        return -1\n"
            "    return total_bytes\n"
        ),
        "cpp_name": "safe_tensor_alloc_bytes",
        "cpp_starter": (
            "long long safe_tensor_alloc_bytes(long long n_elems, long long elem_bytes, long long uint32_max, long long budget_bytes) {\n"
            "    if (n_elems < 0 || elem_bytes <= 0) {\n"
            "        return -1;\n"
            "    }\n"
            "    // BUG (CWE-190 IO2BO): 32-bit multiplication wraps around BEFORE checking budget_bytes!\n"
            "    long long total_bytes = (n_elems * elem_bytes) & uint32_max;\n"
            "    if (total_bytes > budget_bytes) {\n"
            "        return -1;\n"
            "    }\n"
            "    return total_bytes;\n"
            "}\n"
        ),
        "cpp_sol": (
            "long long safe_tensor_alloc_bytes(long long n_elems, long long elem_bytes, long long uint32_max, long long budget_bytes) {\n"
            "    if (n_elems < 0 || elem_bytes <= 0) {\n"
            "        return -1;\n"
            "    }\n"
            "    if (n_elems > uint32_max / elem_bytes) {\n"
            "        return -1;\n"
            "    }\n"
            "    long long total_bytes = n_elems * elem_bytes;\n"
            "    if (total_bytes > budget_bytes) {\n"
            "        return -1;\n"
            "    }\n"
            "    return total_bytes;\n"
            "}\n"
        ),
        "hint": "Add `if n_elems > uint32_max // elem_bytes: return -1` (in Python) or `if (n_elems > uint32_max / elem_bytes) return -1;` (in C++) BEFORE computing `total_bytes = n_elems * elem_bytes`.",
        "tests": [
            {"a": "[100, 4, 4294967295, 1024]", "e": "400"},
            {"a": "[1073741825, 4, 4294967295, 1024]", "e": "-1"},
            {"a": "[300, 4, 4294967295, 1024]", "e": "-1"},
            {"a": "[0, 4, 4294967295, 1024]", "e": "0", "h": 1},
            {"a": "[1073741824, 4, 4294967295, 4294967295]", "e": "-1", "h": 1},
        ],
        "v": [
            ["n_elems (int / long long)", "Number of tensor elements requested (potentially untrusted input from a model file or network packet)."],
            ["elem_bytes (int / long long)", "Byte width of one element (`sizeof(float) == 4`, `sizeof(double) == 8`)."],
            ["uint32_max (int / long long)", "Maximum value representable by a 32-bit unsigned size (`4294967295`, i.e. `0xFFFFFFFF`)."],
            ["budget_bytes (int / long long)", "Maximum bytes the allocator is allowed to hand out."],
            ["total_bytes (int / long long)", "Computed byte count `n_elems * elem_bytes`. In the buggy version, `& uint32_max` models 32-bit unsigned wraparound (`1073741825 * 4 -> 4`!)."],
        ],
        "lines_py": [
            ["def safe_tensor_alloc_bytes(n_elems, elem_bytes, uint32_max, budget_bytes):", "Computes the validated byte size for a tensor allocation or returns `-1` on overflow/OOM."],
            ["    if n_elems < 0 or elem_bytes <= 0:", "Rejects negative element counts or non-positive element byte widths."],
            ["        return -1", "Returns `-1` on invalid dimensions."],
            ["    if n_elems > uint32_max // elem_bytes:", "THE FIX (Reagent C06 guard): Divides `uint32_max` by `elem_bytes` first. If `n_elems` exceeds that quotient, `n_elems * elem_bytes` is guaranteed to overflow `uint32_max`!"],
            ["        return -1", "Rejects the overflowing size BEFORE multiplying (`1073741825 > 1073741823 -> -1`)."],
            ["    total_bytes = n_elems * elem_bytes", "Safe to multiply now: guaranteed not to wrap around `uint32_max`."],
            ["    if total_bytes > budget_bytes:", "Checks whether the true, un-wrapped byte size fits within `budget_bytes`."],
            ["        return -1", "Rejects requests that exceed the memory budget."],
            ["    return total_bytes", "Returns the verified byte count."],
        ],
        "lines_cpp": [
            ["if (n_elems > uint32_max / elem_bytes) { return -1; }", "THE FIX: Standard C/C++ overflow guard (`if (n > SIZE_MAX / sizeof(T))`) that checks the bound via division before multiplying."],
            ["long long total_bytes = n_elems * elem_bytes;", "Computes exact byte count without 32-bit wraparound."],
        ],
    },

    # =========================================================================
    # LEVEL 4, EXERCISE 2: ExecuTorch QuantizedKVCache Dtype vs Byte-Width memcpy
    # =========================================================================
    {
        "id": "kv_dtype_memcpy",
        "l": 4,
        "title": "4.2 Raw C++ memcpy Gated Only on Byte-Width vs. Dtype (ExecuTorch #23399)",
        "real_bug": "pytorch/executorch#23399 (QuantizedKVCache._update_and_return_float_values fatal ET_CHECK_MSG abort on bf16/fp16)",
        "desc": (
            "In `pytorch/executorch#23399`, `QuantizedKVCache` dequantized its cache into a buffer of dtype `cache_dtype` (e.g. `'fp32'`, 4 bytes) and then updated slot `input_pos` with incoming `k_val` of dtype `in_dtype` (e.g. `'bf16'`, 2 bytes) using a raw C++ `memcpy` gated ONLY on byte width (`element_size()`):\n"
            "- When `in_dtype` was `'bf16'` (2 bytes) and `cache_dtype` was `'fp32'` (4 bytes), the C++ op saw `2 != 4` and called `abort()` (crashing the entire process)!\n"
            "- Even worse: if `in_dtype` was `'bf16'` (2 bytes) and `cache_dtype` was `'fp16'` (2 bytes), a byte-width-only `memcpy` saw `2 == 2` and copied the raw bits without converting, corrupting the numbers!\n"
            "In `update_kv_cache_slot(cache_vals, cache_dtype, k_val, in_dtype, input_pos)`, the simulated C++ op `raw_cpp_memcpy_slot` returns `'ABORT_BYTE_WIDTH_MISMATCH'` if the byte sizes differ, or `'CORRUPT_RAW_BITS'` if the byte sizes match (`'bf16'` vs `'fp16'`) but the dtypes differ! "
            "Fix `update_kv_cache_slot` by casting `in_dtype` to `cache_dtype` (setting `cast_dtype = cache_dtype` and rounding `k_val` with `cast_value(k_val, cache_dtype)`) BEFORE calling `raw_cpp_memcpy_slot`."
        ),
        "py_name": "update_kv_cache_slot",
        "py_starter": (
            "def dtype_bytes(dt):\n"
            "    return 4 if dt == 'fp32' else 2\n\n"
            "def cast_value(val, target_dtype):\n"
            "    # Models PyTorch's k_val.to(k_out.dtype) format conversion\n"
            "    return round(val, 2) if target_dtype != 'fp32' else round(val, 5)\n\n"
            "def raw_cpp_memcpy_slot(cache_vals, cache_dtype, val, val_dtype, pos):\n"
            "    if dtype_bytes(val_dtype) != dtype_bytes(cache_dtype):\n"
            "        return 'ABORT_BYTE_WIDTH_MISMATCH'\n"
            "    if val_dtype != cache_dtype:\n"
            "        return 'CORRUPT_RAW_BITS'\n"
            "    out = list(cache_vals)\n"
            "    out[pos] = val\n"
            "    return out\n\n"
            "def update_kv_cache_slot(cache_vals, cache_dtype, k_val, in_dtype, input_pos):\n"
            "    # BUG (executorch#23399): passes k_val with un-cast in_dtype straight into raw C++ memcpy!\n"
            "    return raw_cpp_memcpy_slot(cache_vals, cache_dtype, k_val, in_dtype, input_pos)\n"
        ),
        "py_sol": (
            "def dtype_bytes(dt):\n"
            "    return 4 if dt == 'fp32' else 2\n\n"
            "def cast_value(val, target_dtype):\n"
            "    # Models PyTorch's k_val.to(k_out.dtype) format conversion\n"
            "    return round(val, 2) if target_dtype != 'fp32' else round(val, 5)\n\n"
            "def raw_cpp_memcpy_slot(cache_vals, cache_dtype, val, val_dtype, pos):\n"
            "    if dtype_bytes(val_dtype) != dtype_bytes(cache_dtype):\n"
            "        return 'ABORT_BYTE_WIDTH_MISMATCH'\n"
            "    if val_dtype != cache_dtype:\n"
            "        return 'CORRUPT_RAW_BITS'\n"
            "    out = list(cache_vals)\n"
            "    out[pos] = val\n"
            "    return out\n\n"
            "def update_kv_cache_slot(cache_vals, cache_dtype, k_val, in_dtype, input_pos):\n"
            "    k_cast = cast_value(k_val, cache_dtype)\n"
            "    return raw_cpp_memcpy_slot(cache_vals, cache_dtype, k_cast, cache_dtype, input_pos)\n"
        ),
        "cpp_name": "update_kv_cache_slot",
        "cpp_starter": (
            "// dtype codes: 0 = fp32 (4B), 1 = fp16 (2B), 2 = bf16 (2B)\n"
            "int dtype_bytes(int dt) { return (dt == 0) ? 4 : 2; }\n\n"
            "int raw_cpp_memcpy_slot(int* cache_vals, int cache_dtype, int val, int val_dtype, int pos) {\n"
            "    if (dtype_bytes(val_dtype) != dtype_bytes(cache_dtype)) {\n"
            "        return -999; // ET_CHECK_MSG FATAL ABORT (byte width mismatch)\n"
            "    }\n"
            "    if (val_dtype != cache_dtype) {\n"
            "        return -888; // Silent bit reinterpretation (bf16 bits copied into fp16 slot)\n"
            "    }\n"
            "    cache_vals[pos] = val;\n"
            "    return 1;\n"
            "}\n\n"
            "int update_kv_cache_slot(int* cache_vals, int cache_dtype, int k_val, int in_dtype, int input_pos) {\n"
            "    // BUG (executorch#23399): passes un-cast in_dtype straight to raw C++ memcpy op!\n"
            "    return raw_cpp_memcpy_slot(cache_vals, cache_dtype, k_val, in_dtype, input_pos);\n"
            "}\n"
        ),
        "cpp_sol": (
            "// dtype codes: 0 = fp32 (4B), 1 = fp16 (2B), 2 = bf16 (2B)\n"
            "int dtype_bytes(int dt) { return (dt == 0) ? 4 : 2; }\n\n"
            "int raw_cpp_memcpy_slot(int* cache_vals, int cache_dtype, int val, int val_dtype, int pos) {\n"
            "    if (dtype_bytes(val_dtype) != dtype_bytes(cache_dtype)) {\n"
            "        return -999; // ET_CHECK_MSG FATAL ABORT (byte width mismatch)\n"
            "    }\n"
            "    if (val_dtype != cache_dtype) {\n"
            "        return -888; // Silent bit reinterpretation (bf16 bits copied into fp16 slot)\n"
            "    }\n"
            "    cache_vals[pos] = val;\n"
            "    return 1;\n"
            "}\n\n"
            "int update_kv_cache_slot(int* cache_vals, int cache_dtype, int k_val, int in_dtype, int input_pos) {\n"
            "    int cast_dtype = cache_dtype;\n"
            "    return raw_cpp_memcpy_slot(cache_vals, cache_dtype, k_val, cast_dtype, input_pos);\n"
            "}\n"
        ),
        "hint": "In `update_kv_cache_slot`, convert `k_val` first with `k_cast = cast_value(k_val, cache_dtype)` and pass `k_cast, cache_dtype` (instead of `k_val, in_dtype`) into `raw_cpp_memcpy_slot`.",
        "tests": [
            {"a": "[[1.0, 2.0, 3.0], 'fp32', 9.5, 'bf16', 1]", "e": "[1.0, 9.5, 3.0]"},
            {"a": "[[1.0, 2.0, 3.0], 'fp16', 4.25, 'bf16', 2]", "e": "[1.0, 2.0, 4.25]"},
            {"a": "[[1.0, 2.0], 'fp32', 7.125, 'fp32', 0]", "e": "[7.125, 2.0]"},
            {"a": "[[0.0, 0.0], 'fp32', 3.14159, 'fp16', 0]", "e": "[3.14159, 0.0]", "h": 1},
        ],
        "v": [
            ["cache_vals (list[float] / int*)", "The dequantized KV-cache buffer (`k_out` in `QuantizedKVCache`)."],
            ["cache_dtype (str / int)", "The dtype of `cache_vals` (`self.cache_fp_type`, hardcoded to `'fp32'` in ExecuTorch)."],
            ["k_val (float / int)", "Incoming live key activation produced by the current attention layer."],
            ["in_dtype (str / int)", "Runtime dtype of `k_val` (often `'bf16'` or `'fp16'` during on-device LLM inference)."],
            ["input_pos (int)", "Token sequence position where `k_val` should be written inside `cache_vals`."],
            ["k_cast (float)", "Value of `k_val` after explicit dtype conversion `k_val.to(k_out.dtype)`."],
        ],
        "lines_py": [
            ["def dtype_bytes(dt): return 4 if dt == 'fp32' else 2", "Returns `element_size()`: 4 bytes for `fp32`, 2 bytes for both `fp16` and `bf16`."],
            ["def raw_cpp_memcpy_slot(cache_vals, cache_dtype, val, val_dtype, pos):", "Models `torch.ops.llama.update_cache`, which does a raw C++ `memcpy` after checking `element_size()`."],
            ["    if dtype_bytes(val_dtype) != dtype_bytes(cache_dtype): return 'ABORT_BYTE_WIDTH_MISMATCH'", "Models `ET_CHECK_MSG(k_val.element_size() == k_out.element_size())`, which fatally aborts the Python process on `bf16` (2B) vs `fp32` (4B)!"],
            ["    if val_dtype != cache_dtype: return 'CORRUPT_RAW_BITS'", "Models what happens when `bf16` (2B) is `memcpy`'d into `fp16` (2B): byte widths match, so `memcpy` silently scrambles the exponent/mantissa bits!"],
            ["def update_kv_cache_slot(cache_vals, cache_dtype, k_val, in_dtype, input_pos):", "Models `QuantizedKVCache._update_and_return_float_values()` in `pytorch/executorch#23399`."],
            ["    k_cast = cast_value(k_val, cache_dtype)", "THE FIX (`executorch#23399`): `k_val = k_val.to(k_out.dtype)` converts both the byte width AND bit format to match `cache_dtype` up front."],
            ["    return raw_cpp_memcpy_slot(cache_vals, cache_dtype, k_cast, cache_dtype, input_pos)", "Now `raw_cpp_memcpy_slot` receives matching `cache_dtype`, succeeding on `fp32`, `fp16`, and `bf16` alike."],
        ],
        "lines_cpp": [
            ["if (dtype_bytes(val_dtype) != dtype_bytes(cache_dtype)) return -999;", "In C++, `ET_CHECK_MSG` checks `element_size()` and calls `abort()` when `2 != 4`."],
            ["int cast_dtype = cache_dtype;", "THE FIX: Casts the incoming tensor to `k_out.scalar_type()` before invoking the raw `memcpy` kernel."],
        ],
    },
]
