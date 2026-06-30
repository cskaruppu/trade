"""Claude Code (CLI) as an AI backend — uses your Max subscription, no API key.

If the ``claude`` command-line tool is installed and signed in (which a Claude
Max subscription covers), EdgeForge can run its text AI features by shelling out
to ``claude -p`` instead of calling the paid Anthropic API. Same model, no
per-call API charge — it draws on your existing subscription quota.

Only *text* generation goes through here. Vision (reading a chart image) still
needs the API. The subprocess call is isolated so it can be mocked in tests.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Optional


def claude_cli_available() -> bool:
    """True if the ``claude`` CLI is on PATH (and so usable as a backend)."""
    return shutil.which("claude") is not None


def run_claude_cli(prompt: str, *, system: Optional[str] = None,
                   model: Optional[str] = None, timeout: int = 180) -> str:
    """Run ``claude -p`` headless and return the response text.

    Authenticates with whatever the local Claude Code is signed in as (e.g. a
    Max subscription). Raises ``RuntimeError`` on failure with the CLI's stderr.
    """
    exe = shutil.which("claude")
    if not exe:
        raise RuntimeError(
            "The 'claude' CLI was not found on PATH. Install Claude Code and "
            "sign in (your Max subscription covers it), or set an API key.")

    cmd = [exe, "-p", prompt, "--output-format", "text"]
    if system:
        cmd += ["--append-system-prompt", system]
    if model:
        cmd += ["--model", model]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - env-dependent
        raise RuntimeError(f"claude CLI timed out after {timeout}s") from exc
    except OSError as exc:  # pragma: no cover - env-dependent
        raise RuntimeError(f"could not launch claude CLI: {exc}") from exc

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"claude CLI failed (exit {proc.returncode}): {err}")
    return (proc.stdout or "").strip()
