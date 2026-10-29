"""Real local HTTP servers: offline routing, explicit no-live response, Streamlit startup."""

import os
import secrets
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx


def port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for(url: str, process: subprocess.Popen) -> None:
    for _ in range(100):
        if process.poll() is not None:
            raise RuntimeError("Server exited before becoming ready")
        try:
            if httpx.get(url, timeout=0.5).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    raise RuntimeError("Server startup timed out")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="routelab-http-") as directory:
        api_port, ui_port = port(), port()
        token = secrets.token_urlsafe(32)
        env = {
            **os.environ,
            "ROUTELAB_API_TOKEN": token,
            "ROUTELAB_LIVE_ENABLED": "false",
            "ROUTELAB_ARTIFACT_DIR": str(Path("artifacts/demo").resolve()),
        }
        python = sys.executable
        if os.name == "nt":
            # Launch the interpreter directly: Windows venv redirectors spawn an extra child.
            python = sys._base_executable
            env["PYTHONPATH"] = os.pathsep.join(
                [
                    str(Path("src").resolve()),
                    str(Path(sys.prefix) / "Lib" / "site-packages"),
                ]
            )
        processes = []
        with (Path(directory) / "servers.log").open("w") as log:
            try:
                api = subprocess.Popen(
                    [
                        python,
                        "-m",
                        "uvicorn",
                        "routelab.api.app:app",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(api_port),
                    ],
                    env=env,
                    stdout=log,
                    stderr=log,
                )
                processes.append(api)
                base = f"http://127.0.0.1:{api_port}"
                wait_for(base + "/health", api)
                with httpx.Client(
                    base_url=base, headers={"Authorization": f"Bearer {token}"}
                ) as client:
                    response = client.post(
                        "/api/v1/route", json={"prompt": "Explain TCP and UDP", "strategy": "ml"}
                    )
                    response.raise_for_status()
                    assert response.json()["status"] == "selected"
                    assert (
                        client.post("/api/v1/chat", json={"prompt": "Explain TCP"}).status_code
                        == 503
                    )
                    assert (
                        client.post(
                            "/api/v1/route", json={"prompt": "Explain TCP", "strategy": "oracle"}
                        ).status_code
                        == 422
                    )
                ui = subprocess.Popen(
                    [
                        python,
                        "-m",
                        "streamlit",
                        "run",
                        "apps/dashboard.py",
                        "--server.headless",
                        "true",
                        "--server.address",
                        "127.0.0.1",
                        "--server.port",
                        str(ui_port),
                        "--browser.gatherUsageStats",
                        "false",
                    ],
                    env={**env, "ROUTELAB_API_URL": base},
                    stdout=log,
                    stderr=log,
                )
                processes.append(ui)
                wait_for(f"http://127.0.0.1:{ui_port}/_stcore/health", ui)
                print(
                    "PASS: real HTTP routing, disabled-live/oracle rejection and Streamlit startup"
                )
            finally:
                for process in reversed(processes):
                    if process.poll() is None:
                        process.terminate()
                    process.wait(timeout=15)


if __name__ == "__main__":
    main()
