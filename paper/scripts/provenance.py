"""Lightweight provenance stamping for experiment outputs.

Usage:
    from provenance import stamp
    stamp("figures/ratio_divergence.pdf", __file__)

Creates a .meta sidecar file with script, git SHA, timestamp, and Python version.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def stamp(output_path: str, script_path: str) -> None:
    """Write a .meta sidecar for an output file."""
    meta = {
        "script": os.path.basename(script_path),
        "git_sha": _git_sha(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
    }
    meta_path = Path(output_path).with_suffix(Path(output_path).suffix + ".meta")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
