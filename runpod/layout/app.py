"""Pod-mode HTTP server — wraps the same handler logic as the serverless worker.

POST /
  body: {"file": "<base64>", "fileType": 0, "dpi": 150}
  returns: same JSON as the serverless handler

Run: python app.py  (or via Dockerfile.pod CMD)
"""

import uvicorn
from fastapi import FastAPI, Request

from handler import handler  # model loads here at import time

app = FastAPI()


@app.post("/")
async def process(request: Request):
    body = await request.json()
    return handler({"input": body})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
