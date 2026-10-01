"""
refresh_server.py -- local backend for the SHM dashboard's "Refresh live data" button.

WHAT IT DOES, IN ORDER, when the dashboard's Refresh button is clicked (POST /api/refresh):
  1. runs data.py           (pulls the last window from InfluxDB -> influx_data.csv)
  2. runs load_timescale.py (archives the parsed data into TimescaleDB -- informational only;
                              nothing downstream reads it back, so a failure here is reported
                              but does NOT stop the refresh)
  3. runs SHM_modelv4 as a module, on the FRESH influx_data.csv just pulled in step 1, and
     writes runs_v4/dashboard_data.json -- the one file with every real (non-fabricated) number
     the dashboard shows.

If step 1 fails, the refresh is ABORTED and reported as failed: recomputing on a stale CSV while
telling you it's "live" would be exactly the kind of fabrication this rebuild was meant to remove.

This script also serves the dashboard.html and its JSON over plain HTTP from one origin, so the
browser never hits a CORS wall talking to itself.

SETUP (edit the three constants below once)
  INTELLISENZ_ROOT -- folder containing data.py, load_timescale.py, and the intellisenz package
  WEB_DIR          -- folder containing dashboard.html (your "test" folder)
  CSV_NAME         -- the CSV filename data.py writes and SHM_modelv4 reads (must match data.py's
                      OUTPUT_FILE and load_timescale.py's CSV_FILE -- both currently "influx_data.csv")

USAGE
    python refresh_server.py
    -> open http://localhost:8787/dashboard.html
"""

import http.server
import json
import logging
import shutil
import socketserver
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("refresh_server")

# ---- EDIT THESE THREE TO MATCH YOUR MACHINE -------------------------------
INTELLISENZ_ROOT = Path("/Users/shaurya/Intellisenz")
WEB_DIR = INTELLISENZ_ROOT / "test"          # where dashboard.html lives
CSV_NAME = "influx_data.csv"                 # must match data.py's OUTPUT_FILE
# ----------------------------------------------------------------------------

OUT_DIR = INTELLISENZ_ROOT / "runs_v4"
PORT = 8787
STEP_TIMEOUT_S = 600


def run_step(label: str, cmd: list[str], cwd: Path, fatal: bool) -> dict:
    logger.info("Running %s: %s", label, " ".join(cmd))
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=STEP_TIMEOUT_S)
        ok = p.returncode == 0
        log = (p.stdout or "") + (("\n" + p.stderr) if p.stderr else "")
        if not ok:
            logger.error("%s FAILED (exit %d)\n%s", label, p.returncode, log[-4000:])
        return {"step": label, "ok": ok, "fatal": fatal, "seconds": round(time.time() - t0, 1),
                "log": log[-4000:]}
    except subprocess.TimeoutExpired:
        return {"step": label, "ok": False, "fatal": fatal, "seconds": STEP_TIMEOUT_S,
                "log": f"{label} timed out after {STEP_TIMEOUT_S}s"}
    except Exception as e:
        return {"step": label, "ok": False, "fatal": fatal, "seconds": round(time.time() - t0, 1), "log": str(e)}


def do_refresh() -> dict:
    steps = []
    py = sys.executable

    s1 = run_step("1/3 fetch live data (data.py)", [py, "data.py"], INTELLISENZ_ROOT, fatal=True)
    steps.append(s1)
    if not s1["ok"]:
        return {"ok": False, "steps": steps,
                "message": "Live fetch failed -- refresh aborted so the dashboard is never shown "
                           "stale data labelled as fresh. See the log for this step."}

    s2 = run_step("2/3 archive to TimescaleDB (load_timescale.py)", [py, "load_timescale.py"], INTELLISENZ_ROOT, fatal=False)
    steps.append(s2)
    # non-fatal: nothing downstream reads TimescaleDB back, this is archival only

    s3 = run_step("3/3 recompute SHM model + export dashboard data",
                  [py, "-m", "intellisenz.models.SHM_modelv4", "--csv", CSV_NAME, "--out", str(OUT_DIR)],
                  INTELLISENZ_ROOT, fatal=True)
    steps.append(s3)
    if not s3["ok"]:
        return {"ok": False, "steps": steps, "message": "Model recompute failed -- see log for this step."}

    dash_json = OUT_DIR / "dashboard_data.json"
    if not dash_json.exists():
        return {"ok": False, "steps": steps, "message": "Recompute finished but dashboard_data.json was not produced."}

    return {"ok": True, "steps": steps, "message": "Refresh complete.", "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def log_message(self, fmt, *args):
        logger.info("%s - %s", self.address_string(), fmt % args)

    def _json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/dashboard_data"):
            p = OUT_DIR / "dashboard_data.json"
            if not p.exists():
                return self._json({"error": "No data yet -- click Refresh to run the pipeline once."}, 404)
            return self._json(json.loads(p.read_text()))
        return super().do_GET()

    def do_POST(self):
        if self.path == "/api/refresh":
            result = do_refresh()
            return self._json(result, 200 if result["ok"] else 500)
        self.send_response(404)
        self.end_headers()


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


def main():
    if not (INTELLISENZ_ROOT / "data.py").exists():
        logger.warning("data.py not found at %s -- check INTELLISENZ_ROOT at the top of this file.", INTELLISENZ_ROOT)
    if not WEB_DIR.exists():
        logger.warning("WEB_DIR %s does not exist -- check the constant at the top of this file.", WEB_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with Server(("", PORT), Handler) as httpd:
        logger.info("Serving %s on http://localhost:%d/dashboard.html", WEB_DIR, PORT)
        logger.info("POST http://localhost:%d/api/refresh to run the pipeline manually (curl -X POST ...)", PORT)
        httpd.serve_forever()


if __name__ == "__main__":
    main()