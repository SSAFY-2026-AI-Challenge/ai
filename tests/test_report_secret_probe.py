from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "report_secret_probe",
    ROOT / "deploy" / "report_secret_probe.py",
)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_sanitize_keeps_readable_flags_only() -> None:
    report = module.sanitize(
        {
            "ok": True,
            "project_id": "demo",
            "secrets": {
                "DB_PASSWORD": {
                    "readable": True,
                    "chars": 6,
                    "value": "s3cret",
                }
            },
        }
    )
    assert report["ok"] is True
    assert report["secrets"]["DB_PASSWORD"] == {"readable": True, "chars": 6}
    dumped = str(report)
    assert "s3cret" not in dumped
