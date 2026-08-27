#!/usr/bin/env python3
"""Post Secret Manager probe JSON to a GitHub issue. Never posts secret values."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

ALLOWED_SECRET_KEYS = ("readable", "chars", "error")


def sanitize(report: object) -> dict[str, object]:
    if not isinstance(report, dict):
        return {"ok": False, "error": "probe JSON is not an object"}
    secrets_out: dict[str, object] = {}
    secrets_in = report.get("secrets") or {}
    if isinstance(secrets_in, dict):
        for name, info in secrets_in.items():
            if not isinstance(info, dict):
                secrets_out[str(name)] = {"readable": False, "error": "invalid"}
                continue
            secrets_out[str(name)] = {
                key: info[key] for key in ALLOWED_SECRET_KEYS if key in info
            }
    return {
        "ok": bool(report.get("ok")),
        "project_id": report.get("project_id"),
        "error": report.get("error"),
        "secrets": secrets_out,
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "sha": os.environ.get("GITHUB_SHA"),
        "job_status": os.environ.get("JOB_STATUS"),
    }


def main() -> None:
    token = os.environ["GH_TOKEN"]
    repo = os.environ["GITHUB_REPOSITORY"]
    issue = os.environ["ISSUE_NUMBER"]
    path = Path(os.environ.get("PROBE_FILE", "")).expanduser()
    if path.is_file():
        report = json.loads(path.read_text(encoding="utf-8"))
        safe = sanitize(report)
        dumped = json.dumps(safe, ensure_ascii=False, indent=2)
        body = f"## Secret Manager probe\n\n```json\n{dumped}\n```\n"
    else:
        status = os.environ.get("JOB_STATUS", "unknown")
        body = f"probe 결과 파일이 없습니다. job_status={status}"
    payload = json.dumps({"body": body}).encode("utf-8")
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/issues/{issue}/comments",
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "credit-ai-secret-probe",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            print(f"issue comment {resp.status}")
    except urllib.error.HTTPError as exc:
        print(exc.read().decode("utf-8", errors="replace"))
        raise


if __name__ == "__main__":
    main()
