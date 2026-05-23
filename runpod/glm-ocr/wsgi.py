"""Gunicorn WSGI entry-point for glmocr server.

Config path is read from the GLMOCR_CONFIG environment variable
(default: /etc/glmocr_config.yaml).
"""
import os
import multiprocessing

from glmocr.config import load_config
from glmocr.server import create_app

multiprocessing.set_start_method("spawn", force=True)

_config_path = os.environ.get("GLMOCR_CONFIG", "/etc/glmocr_config.yaml")
config = load_config(_config_path)
app = create_app(config)
app.config["pipeline"].start()
