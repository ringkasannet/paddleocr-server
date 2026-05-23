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

# Route glmocr's logger through Gunicorn's error log handler so that
# request-level logs appear in the supervisor log file.
# glmocr sets propagate=False and holds a StreamHandler to a stale sys.stdout
# reference after Gunicorn forks — borrowing gunicorn.error's handlers fixes this.
import logging as _logging
_gunicorn_logger = _logging.getLogger("gunicorn.error")
_glmocr_logger   = _logging.getLogger("glmocr")
_glmocr_logger.handlers  = _gunicorn_logger.handlers
_glmocr_logger.setLevel(_gunicorn_logger.level or _logging.INFO)
