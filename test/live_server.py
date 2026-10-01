"""
live_server.py -- serves dashboard.html and a JSON API, refreshing the data in the background.

    uvicorn live_server:app --host 0.0.0.0 --port 8000

Endpoints
    GET  /                 the dashboard
    GET  /api/dashboard    latest model output (what the dashboard polls)
    GET  /api/status       last run time / last error
    POST /api/refresh      trigger a refresh now
Env
    REFRESH_SECONDS   seconds between refreshes (default 300)
    CORS_ORIGINS      comma-separated origins if the page is NOT served by this server
"""
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

import Intellisenz.test.live_server as live_server
import live_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s  %(message)s")
log = logging.getLogger("live_server")

REFRESH_SECONDS = int(os.getenv("REFRESH_SECONDS", "300"))
DASHBOARD = Path(__file__).with_name("dashboard.html")

state = {"payload": None, "last_ok": None, "last_error": None, "running": False}
_run_lock = threading.Lock()


def run_once() -> None:
    if not _run_lock.acquire(blocking=False):      # a refresh is already in progress
        return
    state["running"] = True
    try:
        state["payload"] = live_pipeline.refresh()
        state["last_ok"] = time.time()
        state["last_error"] = None
        log.info("Refresh OK")
    except Exception as e:                          # keep serving the previous payload
        state["last_error"] = f"{type(e).__name__}: {e}"
        log.exception("Refresh failed")
    finally:
        state["running"] = False
        _run_lock.release()


def _loop(stop: threading.Event) -> None:
    while not stop.is_set():
        run_once()
        stop.wait(REFRESH_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    stop = threading.Event()
    threading.Thread(target=_loop, args=(stop,), daemon=True).start()
    yield
    stop.set()


app = FastAPI(title="SHM live API", lifespan=lifespan)
if os.getenv("CORS_ORIGINS"):
    app.add_middleware(CORSMiddleware, allow_origins=os.environ["CORS_ORIGINS"].split(","),
                       allow_methods=["GET", "POST"], allow_headers=["*"])


@app.get("/")
def index():
    return FileResponse(DASHBOARD)


@app.get("/api/dashboard")
def dashboard_data():
    if state["payload"] is None:
        raise HTTPException(503, state["last_error"] or "First refresh still running")
    return JSONResponse(state["payload"], headers={"Cache-Control": "no-store"})


@app.get("/api/status")
def status():
    return {"running": state["running"], "last_ok_epoch": state["last_ok"],
            "last_error": state["last_error"], "refresh_seconds": REFRESH_SECONDS}


@app.post("/api/refresh")
def refresh_now():
    threading.Thread(target=run_once, daemon=True).start()
    return {"started": True}





# python -m uvicorn live_server:app --host 127.0.0.1 --port 8000