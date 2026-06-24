"""Alert delivery: console, local file log, and optional Telegram.

The live scanner calls :func:`dispatch` with an alert. By default alerts print
to the console and append to ``alerts.log``. If you configure a Telegram bot
(token + chat id in config.yaml under ``alerts.telegram`` or via the
``TELEGRAM_BOT_TOKEN`` / ``TELEGRAM_CHAT_ID`` env vars) they're also pushed to
your phone.

Message *formatting* is pure and unit-tested; only the network send needs
credentials.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Optional

ALERT_LOG = "alerts.log"


@dataclass
class AlertConfig:
    console: bool = True
    log_file: Optional[str] = ALERT_LOG
    telegram_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None

    @classmethod
    def from_config(cls, cfg: dict) -> "AlertConfig":
        a = (cfg or {}).get("alerts", {}) or {}
        tg = a.get("telegram", {}) or {}
        return cls(
            console=a.get("console", True),
            log_file=a.get("log_file", ALERT_LOG),
            telegram_token=tg.get("token") or os.environ.get("TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=(str(tg.get("chat_id")) if tg.get("chat_id")
                              else os.environ.get("TELEGRAM_CHAT_ID")),
        )

    @property
    def telegram_enabled(self) -> bool:
        return bool(self.telegram_token and self.telegram_chat_id)


def format_alert(title: str, lines: list[str]) -> str:
    """Build a plain-text alert body from a title and detail lines."""
    body = "\n".join(f"• {ln}" for ln in lines if ln)
    return f"{title}\n{body}" if body else title


def send_telegram(token: str, chat_id: str, text: str, *, timeout: int = 10) -> bool:
    """Send ``text`` to a Telegram chat. Returns True on HTTP 200."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps({"chat_id": chat_id, "text": text}).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return resp.status == 200
    except Exception:  # noqa: BLE001 - never let a failed alert crash a scan
        return False


def dispatch(text: str, config: AlertConfig) -> dict:
    """Deliver ``text`` through every enabled channel. Returns per-channel status."""
    status = {}
    if config.console:
        print(text)
        status["console"] = True
    if config.log_file:
        try:
            with open(config.log_file, "a", encoding="utf-8") as fh:
                fh.write(text + "\n---\n")
            status["log"] = True
        except OSError:
            status["log"] = False
    if config.telegram_enabled:
        status["telegram"] = send_telegram(
            config.telegram_token, config.telegram_chat_id, text)
    return status
