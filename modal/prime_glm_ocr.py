"""Prime the deployed GLM-OCR worker — triggers snapshot creation and waits.

Run once after every `modal deploy` that rebuilds the image/snapshot:
    python modal/prime_glm_ocr.py

The first call takes 3-10 min (vLLM startup + CUDA warmup + snapshot creation).
Once it returns "ready", all future cold starts restore from snapshot in ~5s.
"""

import time
import requests

ENDPOINT = "https://ringkasan-net--glm-ocr-ocrfrontend-prime.modal.run"


def main():
    print(f"Priming GLM-OCR worker via {ENDPOINT}")
    print("This takes 3-10 min on first deploy (snapshot creation). Please wait...\n")

    t0 = time.time()
    try:
        resp = requests.get(ENDPOINT, timeout=900)
        resp.raise_for_status()
        data = resp.json()
    except requests.Timeout:
        print(f"TIMEOUT after {round(time.time() - t0)}s — server may still be starting.")
        return
    except Exception as e:
        print(f"ERROR: {e}")
        return

    wall = round(time.time() - t0, 1)
    status  = data.get("status", "?")
    elapsed = data.get("elapsed_s", "?")
    chars   = data.get("chars", "?")

    print(f"Status   : {status}")
    print(f"Server   : {elapsed}s")
    print(f"Wall     : {wall}s")
    print(f"OCR chars: {chars}")
    print("\nSnapshot ready — cold starts will now restore in ~5s.")


if __name__ == "__main__":
    main()
