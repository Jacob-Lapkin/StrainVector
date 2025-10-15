from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class Logger:
    def __init__(self, file: Optional[Path] = None):
        self.file = file
        if self.file is not None:
            self.file.parent.mkdir(parents=True, exist_ok=True)
            # Touch file
            self.file.open("a").close()

    def _ts(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _write(self, level: str, msg: str) -> None:
        line = f"[{self._ts()}] {level}: {msg}"
        print(line)
        if self.file is not None:
            with self.file.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def info(self, msg: str) -> None:
        self._write("INFO", msg)

    def warn(self, msg: str) -> None:
        self._write("WARN", msg)

    def error(self, msg: str) -> None:
        self._write("ERROR", msg)


def logger_for_refdb(out_dir: Path) -> Logger:
    return Logger(out_dir / "logs" / "index.log")


def logger_for_profile(out_json: Path) -> Logger:
    return Logger(out_json.with_suffix(".log"))

