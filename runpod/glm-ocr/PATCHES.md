# GLM-OCR Server Patches

Two runtime patches are applied to the installed `glmocr` package after `pip install`:

---

## Patch 1 — PDF data URI support (`page_loader.py`)

**File:** `/venv/main/lib/python3.11/site-packages/glmocr/dataloader/page_loader.py`

**Problem:** `glmocr` only accepts PDFs via file paths or raw bytes. Sending a `data:application/pdf;base64,...` URI caused it to fall through to `_load_image()` and silently skip the document.

**Fix:** Added a branch at the top of both `_load_source()` and `_iter_source()` that detects `data:application/pdf` URIs, base64-decodes them, and routes them to `_load_pdf_bytes()` / `_iter_pdf_bytes()` respectively.

**Applied via:**
```bash
python3 - <<'PYEOF'
import pathlib

pkg = pathlib.Path("/venv/main/lib/python3.11/site-packages/glmocr/dataloader/page_loader.py")
src = pkg.read_text()

OLD_LOAD = '''        if source.startswith("file://"):
            file_path = source[7:]
        else:
            file_path = source

        # Detect PDF
        if os.path.isfile(file_path) and file_path.lower().endswith(".pdf"):
            return self._load_pdf(file_path)

        # Otherwise load as a single image page
        return [self._load_image(source)]'''

NEW_LOAD = '''        # Handle PDF data URIs (data:application/pdf;base64,...)
        if source.startswith("data:application/pdf"):
            _, b64data = source.split(",", 1)
            import base64 as _b64
            return self._load_pdf_bytes(_b64.b64decode(b64data))

        if source.startswith("file://"):
            file_path = source[7:]
        else:
            file_path = source

        # Detect PDF
        if os.path.isfile(file_path) and file_path.lower().endswith(".pdf"):
            return self._load_pdf(file_path)

        # Otherwise load as a single image page
        return [self._load_image(source)]'''

OLD_ITER = '''        if source.startswith("file://"):
            file_path = source[7:]
        else:
            file_path = source

        if os.path.isfile(file_path) and file_path.lower().endswith(".pdf"):
            yield from self._iter_pdf(file_path)
        else:
            yield self._load_image(source)'''

NEW_ITER = '''        # Handle PDF data URIs (data:application/pdf;base64,...)
        if source.startswith("data:application/pdf"):
            _, b64data = source.split(",", 1)
            import base64 as _b64
            yield from self._iter_pdf_bytes(_b64.b64decode(b64data))
            return

        if source.startswith("file://"):
            file_path = source[7:]
        else:
            file_path = source

        if os.path.isfile(file_path) and file_path.lower().endswith(".pdf"):
            yield from self._iter_pdf(file_path)
        else:
            yield self._load_image(source)'''

patched = src
assert OLD_LOAD in patched, "patch target _load_source not found — glmocr version mismatch"
assert OLD_ITER in patched, "patch target _iter_source not found — glmocr version mismatch"
patched = patched.replace(OLD_LOAD, NEW_LOAD, 1)
patched = patched.replace(OLD_ITER, NEW_ITER, 1)
pkg.write_text(patched)
print("[patch] page_loader.py: data:application/pdf URI support added")
PYEOF
```

---

## Patch 2 — Layout GPU semaphore (`_workers.py`)

**File:** `/venv/main/lib/python3.11/site-packages/glmocr/pipeline/_workers.py`

**Problem:** Concurrent HTTP requests each spawn their own `layout_worker` thread, all sharing a single `PPDocLayoutDetector` on GPU. Without locking, simultaneous `layout_detector.process()` calls cause CUDA OOM errors.

**Fix:** Added a module-level `threading.Semaphore(1)` (`_LAYOUT_GPU_SEMAPHORE`) and wrapped the `layout_detector.process()` call with it, serialising GPU layout forward passes while keeping page loading and vLLM OCR calls fully concurrent.

**Applied via:**
```bash
python3 - <<'PYEOF'
import pathlib

pkg = pathlib.Path("/venv/main/lib/python3.11/site-packages/glmocr/pipeline/_workers.py")
src = pkg.read_text()

if "_LAYOUT_GPU_SEMAPHORE" in src:
    print("[patch] _workers.py: layout semaphore already present, skipping")
else:
    OLD_IMPORT = '''import queue
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed'''

    NEW_IMPORT = '''import queue
import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# One GPU layout forward pass at a time across all concurrent requests.
_LAYOUT_GPU_SEMAPHORE = threading.Semaphore(1)'''

    OLD_PROCESS = '''    try:
        layout_results, vis_images = layout_detector.process(
            batch_images,
            save_visualization=save_visualization,
            global_start_idx=global_start_idx,
            use_polygon=use_polygon,
        )
        if vis_images:
            state.layout_vis_images.update(vis_images)'''

    NEW_PROCESS = '''    try:
        with _LAYOUT_GPU_SEMAPHORE:
            layout_results, vis_images = layout_detector.process(
                batch_images,
                save_visualization=save_visualization,
                global_start_idx=global_start_idx,
                use_polygon=use_polygon,
            )
        if vis_images:
            state.layout_vis_images.update(vis_images)'''

    assert OLD_IMPORT in src, "patch target (imports) not found in _workers.py — glmocr version mismatch"
    assert OLD_PROCESS in src, "patch target (_flush_layout_batch) not found in _workers.py — glmocr version mismatch"
    patched = src.replace(OLD_IMPORT, NEW_IMPORT, 1)
    patched = patched.replace(OLD_PROCESS, NEW_PROCESS, 1)
    pkg.write_text(patched)
    print("[patch] _workers.py: layout GPU semaphore(1) applied")
PYEOF
```

---

---

## Patch 3 — vLLM submission semaphore (`_workers.py`)

**File:** `/venv/main/lib/python3.11/site-packages/glmocr/pipeline/_workers.py`

**Problem:** After Patch 2 serialised layout, all concurrent requests still flood vLLM simultaneously. A 10-request load test with a 31-page PDF sends ~4 200 regions to vLLM at once; vLLM batches them all together and every request finishes at roughly the same time (~105 s), so no request benefits from being "first".

**Fix:** Added a module-level `threading.Semaphore(1)` (`_VLLM_SEMAPHORE`) that is acquired at the start of `recognition_worker` and released in its `finally` block. Only one request's OCR stage runs at a time. Layout for queued requests continues to run freely (the existing `_LAYOUT_GPU_SEMAPHORE` is released between page batches) and buffers its regions into `region_queue` (capacity 2 000, large enough for a full document). The result is a pipelined assembly line:

```
Req 1: [layout]──[vLLM ~100s]──done at ~103s
Req 2:    [layout during Req 1 vLLM]──[wait vllm_sem]──[vLLM ~100s]──done at ~203s
Req 3:          [layout during Req 2 vLLM]──────────────────────────[vLLM]──done at ~303s
```

**Trade-off:** Total throughput for a burst of N requests is N × single-request time. Earlier requests return faster; later requests wait longer than under the "flood vLLM" approach.

**Applied via:**
```bash
python3 - <<'PYEOF'
import pathlib, glmocr, os

pkg = pathlib.Path(os.path.dirname(glmocr.__file__)) / "pipeline" / "_workers.py"
src = pkg.read_text()

if "_VLLM_SEMAPHORE" in src:
    print("[patch] _workers.py: vLLM semaphore already present, skipping")
else:
    OLD_SEM = '''# One GPU layout forward pass at a time across all concurrent requests.
_LAYOUT_GPU_SEMAPHORE = threading.Semaphore(1)'''

    NEW_SEM = '''# One GPU layout forward pass at a time across all concurrent requests.
_LAYOUT_GPU_SEMAPHORE = threading.Semaphore(1)

# One vLLM submission batch at a time across all concurrent requests.
_VLLM_SEMAPHORE = threading.Semaphore(1)'''

    OLD_RECOG_START = '''    """Consume regions, run parallel OCR, store results."""
    executor = None
    try:'''

    NEW_RECOG_START = '''    """Consume regions, run parallel OCR, store results."""
    _VLLM_SEMAPHORE.acquire()
    executor = None
    try:'''

    OLD_RECOG_FINALLY = '''    finally:
        state.drain_queue(state.region_queue)'''

    NEW_RECOG_FINALLY = '''    finally:
        _VLLM_SEMAPHORE.release()
        state.drain_queue(state.region_queue)'''

    assert OLD_SEM in src, "patch target (_LAYOUT_GPU_SEMAPHORE) not found — glmocr version mismatch"
    assert OLD_RECOG_START in src, "patch target (recognition_worker start) not found — glmocr version mismatch"
    assert OLD_RECOG_FINALLY in src, "patch target (recognition_worker finally) not found — glmocr version mismatch"
    patched = src.replace(OLD_SEM, NEW_SEM, 1)
    patched = patched.replace(OLD_RECOG_START, NEW_RECOG_START, 1)
    patched = patched.replace(OLD_RECOG_FINALLY, NEW_RECOG_FINALLY, 1)
    pkg.write_text(patched)
    print("[patch] _workers.py: vLLM semaphore(1) applied")
PYEOF
```

---

All patches include assertion guards that fail loudly if the patch targets are not found, protecting against silent failures on glmocr version changes.
