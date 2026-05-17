# GLM-OCR vast.ai Deployment — Critical Findings

## 1. Wrong model ID

**Error:** `Repository Not Found` / model download fails  
**Cause:** Model is `zai-org/GLM-OCR`, not `THUDM/GLM-OCR`.  
**Fix:** `MODEL="zai-org/GLM-OCR"` in `start.sh`.

---

## 2. supervisor xmlrpc conflict with venv

**Error:**
```
ImportError: cannot import name 'xmlrpc' from 'supervisor'
  File "/venv/main/lib/python3.11/site-packages/supervisor/xmlrpc.py"
```
**Cause:** `glmocr[server]` pulls in the `supervisor` PyPI package into `/venv/main`, which shadows the system-installed supervisord when the venv is active.  
**Fix:** `deactivate` before any `supervisorctl` or `supervisord` call.

---

## 3. supervisord not running during manual provisioning

**Error:** `unix:///var/run/supervisor.sock no such file`  
**Cause:** On vast.ai the provisioning script runs before supervisord starts, so `supervisorctl` has nothing to talk to.  
**Fix:**
```bash
if ! pgrep -x supervisord > /dev/null; then
    supervisord -c /etc/supervisor/supervisord.conf
    sleep 2
fi
supervisorctl reread && supervisorctl update
```

---

## 4. HF_TOKEN not inherited by supervisord

**Error:** vLLM crashes with `401 Unauthorized` when downloading model weights.  
**Cause:** `export HF_TOKEN=...` in the provisioning shell is not forwarded to an already-running supervisord process.  
**Fix:** Inject the token directly into the supervisor program conf:
```ini
environment=PROC_NAME="%(program_name)s",HF_TOKEN="hf_xxx"
```
Use `sed -i` to write it before `supervisorctl update`.

---

## 5. glmocr config — wrong nesting / shallow merge strips defaults

**Error:** Pipeline crashes with `KeyError: 'threshold'` / layout model fails to load.  
**Cause:** Writing a minimal YAML with only a few overrides caused a shallow merge at the `pipeline.layout` level, silently dropping `threshold`, `id2label`, `label_task_mapping`, and other required fields.  
**Fix:** Copy the package default config and sed-patch only the values that differ:
```bash
cp /venv/main/lib/python3.11/site-packages/glmocr/config.yaml /etc/glmocr_config.yaml
sed -i "s/port: 5002/port: ${GLMOCR_PORT}/"     /etc/glmocr_config.yaml
sed -i 's/enabled: true/enabled: false/'         /etc/glmocr_config.yaml
sed -i "s/api_port: 8080/api_port: ${OCR_PORT}/" /etc/glmocr_config.yaml
sed -i 's/# device: null/device: "cuda:0"/'      /etc/glmocr_config.yaml
sed -i "s/batch_size: 1/batch_size: 4/"          /etc/glmocr_config.yaml
sed -i "s/max_tokens: 8192/max_tokens: 2048/"    /etc/glmocr_config.yaml
```

---

## 6. `data:application/pdf;base64,...` URIs silently ignored

**Error:** Server returns `{"json_result": [], "layout_details": [], "markdown_result": ""}` — 0 regions, no exception.  
**Log:** `Skipping source (unit 0): Error loading image 'data:application/pdf;base64,...'`

**Root cause** (`page_loader.py`):
```python
def _load_source(self, source):
    # raw bytes branch handles b"%PDF-" correctly ✓
    # string branch falls through to _load_image() for data: URIs
    if source.startswith("file://"):
        file_path = source[7:]
    else:
        file_path = source
    if os.path.isfile(file_path) and file_path.lower().endswith(".pdf"):
        return self._load_pdf(file_path)          # ← only file paths work
    return [self._load_image(source)]             # ← data:application/pdf lands here

def _load_image(self, source):
    if source.startswith("data:image"):           # ← only data:image/* handled
        ...
    else:
        raise ValueError(f"Invalid image source: {source}")   # ← raised, then swallowed
```
`iter_pages_with_unit_indices` catches the RuntimeError as `logger.warning("Skipping source …")` — no HTTP error, silent empty result.

**Fix:** Patch the installed package in `start.sh` to add a `data:application/pdf` branch in both `_load_source()` and `_iter_source()`:
```python
if source.startswith("data:application/pdf"):
    _, b64data = source.split(",", 1)
    import base64 as _b64
    return self._load_pdf_bytes(_b64.b64decode(b64data))
```
See the `python3 - <<'PYEOF'` heredoc in `start.sh` for the full patch.

**Note:** GLM-OCR **does** support PDFs — but only via file paths ending in `.pdf` or raw bytes starting with `b"%PDF-"`. The `data:application/pdf;base64,...` MIME type was simply not implemented.

---

## 7. max_tokens vs max_model_len mismatch

**Error:**
```
max_tokens=8192 cannot be greater than max_model_len=4096
```
**Cause:** glmocr's default config has `max_tokens: 8192`; vLLM was launched with `--max-model-len 4096`.  
**Fix:** The official vLLM recipe uses `max_tokens: 2048`. Keep `MAX_MODEL_LEN=4096` and add a sed patch to cap glmocr's `max_tokens` at 2048.

---

## 8. supervisorctl `restart` fails on FATAL/STOPPED state

**Error:** `ERROR (not running)` when running `supervisorctl restart <program>` after a crash.  
**Cause:** `restart` = `stop` + `start`; it fails if the program is not currently running.  
**Fix:** Use `supervisorctl start <program>` to bring up a stopped/fatal process.

---

## 9. supervisorctl tail shows nothing with /dev/stdout log

**Cause:** `stdout_logfile=/dev/stdout` in supervisor conf routes output to supervisord's own stdout, not a file. `supervisorctl tail` needs a real file path.  
**Workaround:**
```bash
tail -f /var/log/supervisor/supervisord.log
# or redirect to a file:
sed -i 's|stdout_logfile=/dev/stdout|stdout_logfile=/tmp/vllm-0.log|' \
    /etc/supervisor/conf.d/vllm-0.conf
```

---

## 10. Port 5002 not externally mapped on vast.ai

**Cause:** The vast.ai template only exposes ports declared at instance creation. Port 5002 (glmocr) was not in the template.  
**Workaround (existing instance):** SSH tunnel:
```bash
ssh -L 5002:localhost:5002 root@<host> -p <ssh-port>
```
**Fix (template):** Add port 5002 to the vast.ai template's exposed port list.

---

## 11. MTP speculative tokens: use 1 not 3

**Source:** Official vLLM GLM-OCR recipe.  
**Finding:** The recipe specifies `--speculative-config.num_speculative_tokens 1`. Using 3 is untested against the official guidance and may hurt rather than help on this model.  
**Fix:** `MTP_JSON='{"method":"mtp","num_speculative_tokens":1}'` in `start.sh`.

---

## 12. OOM during layout detection at higher concurrency

**Error:** `Layout detection failed for pages [N], skipping batch: CUDA out of memory. Tried to allocate 276.00 MiB.`  
**Cause:** vLLM and glmocr's layout detection model share GPU VRAM. At 75% GPU util, vLLM takes ~18.2 GiB; the layout model takes ~5.2 GiB — leaving < 200 MiB free. Concurrent requests trigger parallel layout batches, each needing 276 MiB.  
**Fix:**
- Reduce `--gpu-memory-utilization` from `0.75` → `0.65` (frees ~2.4 GiB for glmocr)
- Add `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to glmocr's supervisor environment to reduce allocator fragmentation

**Counterintuitive finding:** higher GPU util (80–83%) gives *better* throughput than lower (65%) despite more OOM retries. A larger KV cache reduces vLLM evictions during concurrent long-sequence OCR decoding, which matters more than eliminating layout retries.

| GPU util | Concurrency | p/s |
|----------|-------------|-----|
| 75% | 8 | 1.46 |
| 65% | 8 | 1.11 |
| 83% | 8 | **1.77** ← best |

**Recommended setting:** `--gpu-memory-utilization 0.80` on RTX 3090.  
Add `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to glmocr's supervisor env to reduce fragmentation and cut retry frequency.

**Memory budget on RTX 3090 (23.56 GiB) at 80%:**
| Component | VRAM |
|-----------|------|
| vLLM @ 80% | ~18.8 GiB |
| glmocr layout model | ~3.7 GiB |
| Available for layout batches | ~1.0 GiB |

---

## 13. 1-page chunks are slower than 4-page chunks

**Finding:** Reducing chunk size to 1 page to improve vLLM batching actually decreases throughput.

| Chunk size | Concurrency | p/s |
|------------|-------------|-----|
| 4 pages | 8 | 1.46 |
| 1 page | 8 | 0.66 |
| 1 page | 16 | 0.87 |
| 1 page | 32 | 0.60 (OOM) |

**Reason:** glmocr runs layout detection once per request (not per page). With 1-page chunks, layout detection overhead is paid for every single page. With 4-page chunks it is amortized across 4 pages. Additionally, higher concurrency with 1-page chunks causes concurrent layout detection OOMs, returning silent empty (0 KB) HTTP 200 responses.

**Recommendation:** Use 4-page chunks. The optimal concurrency is 8–12 on RTX 3090.

---

## Environment

| Item | Value |
|------|-------|
| Base image | `vastai/base-image:cuda-12.8.1-cudnn-devel-ubuntu22.04-py311` |
| GPU | RTX 3090 24 GB |
| Model | `zai-org/GLM-OCR` |
| vLLM | `>=0.9.0` |
| glmocr | `glmocr[selfhosted,server]` |
| Python venv | `/venv/main` |
| glmocr config | `/etc/glmocr_config.yaml` |
| Supervisor scripts | `/opt/supervisor-scripts/` |
| Ports | 5002 → glmocr, 8000 → vLLM |
