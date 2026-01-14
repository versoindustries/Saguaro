# Saguaro Tokenizer Build Fix - Action Required

## Overview
The Saguaro native tokenizer (`_saguaro_core.so`) has a critical buffer overflow bug that causes segmentation faults during indexing and baseline training. A fix has been implemented but **cannot be compiled** due to a build system pollution issue.

---

## Problem 1: Buffer Overflow (FIXED in source, NOT compiled)

### Location
`saguaro/native/ops/fused_text_tokenizer_op.h`, function `text_tokenize_batch_parallel` (lines 689-715)

### Root Cause
When processing text files larger than `max_length` (default 4096), the function writes directly to the output tensor without bounds checking. A 100KB file produces ~100,000 tokens, overflowing a 4096-element buffer.

### Fix Applied
Added a temporary scratch buffer and safe copy with truncation:
```cpp
// Allocate scratch buffer large enough for input
size_t required_size = text_lens[i] + 2;
std::vector<int32_t> temp_tokens(required_size);

// Tokenize into temporary buffer
int len = text_tokenize_utf8_simd(..., temp_tokens.data(), ...);

// Safe Copy with Truncation
int copy_len = std::min(len, max_length);
std::memcpy(out_ptr, temp_tokens.data(), copy_len * sizeof(int32_t));
```

### Status
✅ Source code fixed in `fused_text_tokenizer_op.h`
❌ NOT compiled into `_saguaro_core.so`

---

## Problem 2: Build System Pollution (BLOCKING)

### Location
- `CMakeLists.txt` (lines 24-42)
- `build_secure.sh` (line 226)

### Root Cause
The `cv2` (OpenCV) package in the virtual environment has a startup hook that prints `sys.path` to stdout when Python is invoked. This output pollutes the CMake `execute_process()` capture of TensorFlow include paths.

### Symptom
```
-- TF Include: ['', '/home/.../site-packages/cv2', ...]
/home/.../tensorflow/include
```
This multi-line output corrupts `flags.make`:
```
CXX_INCLUDES = -I"/home/...['', ...]
/home/.../tensorflow/include"
```
Leading to: `flags.make:8: *** missing separator. Stop.`

### Attempted Fixes (Failed)
1. Added `-I` (isolated mode) to Python calls in CMake
2. Added `-I` to `build_secure.sh`
3. Filtered `cv2` from `sys.path` before import

None worked because the pollution happens from a `.pth` or config file loaded before any user code.

---

## Required Actions

### Option A: Remove cv2 from venv (Quick)
```bash
./venv/bin/pip uninstall opencv-python opencv-python-headless
```
Then rebuild:
```bash
source ./venv/bin/activate && bash ./build_secure.sh
```

### Option B: Fix cv2 Pollution (Robust)
Find and patch the offending file in `venv/lib/python3.12/site-packages/cv2/`:
- Check `config.py`, `config-3.py`, or `load_config_py3.py` for `print(sys.path)` statements
- Remove or comment out the print

### Option C: Rewrite CMake to Filter Output (Complex)
Modify CMake to regex-filter the TensorFlow path extraction:
```cmake
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import tensorflow as tf; print(tf.sysconfig.get_include())"
    OUTPUT_VARIABLE RAW_TF_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
# Extract last line only (actual TF path)
string(REGEX REPLACE ".*\n" "" TF_INCLUDE_DIR "${RAW_TF_INCLUDE_DIR}")
```

---

## Verification After Fix

1. Rebuild native ops:
   ```bash
   source ./venv/bin/activate && rm -rf build && bash ./build_secure.sh
   ```

2. Reinstall package:
   ```bash
   ./venv/bin/pip install .
   ```

3. Test training (should not crash):
   ```bash
   ./venv/bin/saguaro train-baseline --corpus .
   ```

4. Test indexing (should not crash):
   ```bash
   ./venv/bin/saguaro index --path .
   ```

---

## Files Modified

| File | Change |
|------|--------|
| `saguaro/native/ops/fused_text_tokenizer_op.h` | Buffer overflow fix |
| `saguaro/tokenization/train_baseline.py` | Created (HuggingFace curriculum support) |
| `CMakeLists.txt` | Attempted `-I` fix (partial) |
| `build_secure.sh` | Attempted `-I` fix (partial) |

---

## Priority
**CRITICAL** - Saguaro cannot index or train until this is resolved.
