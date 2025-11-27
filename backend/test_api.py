"""
Lightweight integration tests for the FastAPI application.
"""

import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.main import UPLOAD_DIR, app, jobs_store  # noqa: E402  pylint: disable=wrong-import-position

client = TestClient(app)


def setup_function(_: object) -> None:
    jobs_store.clear()
    for file in UPLOAD_DIR.glob("*"):
        if file.is_file():
            file.unlink()


def test_upload_analyze_results_flow(tmp_path: Path) -> None:
    response = client.post(
        "/upload",
        files={"file": ("report.txt", b"Acme Corp reported record growth.", "text/plain")},
    )
    assert response.status_code == 200
    body = response.json()
    job_id = body["job_id"]

    analyze_response = client.post("/analyze", json={"job_id": job_id})
    assert analyze_response.status_code == 200

    final_body = None
    for _ in range(20):
        result_response = client.get(f"/results/{job_id}")
        final_body = result_response.json()
        if final_body["status"] in {"completed", "partial", "failed"}:
            break
        time.sleep(0.2)

    assert final_body is not None
    assert final_body["job_id"] == job_id
    assert final_body["results"]["summary"] is not None

