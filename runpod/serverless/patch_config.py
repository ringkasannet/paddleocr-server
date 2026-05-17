"""
Patches PaddleOCR-VL.yaml for single-container RunPod deployment.
Reads target values from env vars so the same script works for any model/port.
"""
import yaml, sys, os

CONFIG_PATH = os.environ.get("PIPELINE_CONFIG", "/workspace/PaddleOCR-VL.yaml")
MODEL_NAME  = os.environ.get("MODEL_NAME",  "PaddleOCR-VL-1.5-0.9B")
SERVER_URL  = os.environ.get("VLLM_SERVER_URL", "http://localhost:8118/v1")

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

vl = cfg.get("SubModules", {}).get("VLRecognition")
if vl is None:
    print("ERROR: SubModules.VLRecognition not found in config", file=sys.stderr)
    sys.exit(1)

vl["model_name"] = MODEL_NAME
vl.setdefault("genai_config", {})
vl["genai_config"]["backend"]    = "vllm-server"
vl["genai_config"]["server_url"] = SERVER_URL

print(f"Patched SubModules.VLRecognition:")
print(f"  model_name = {vl['model_name']}")
print(f"  backend    = {vl['genai_config']['backend']}")
print(f"  server_url = {vl['genai_config']['server_url']}")

with open(CONFIG_PATH, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

print(f"Done: {CONFIG_PATH}")
