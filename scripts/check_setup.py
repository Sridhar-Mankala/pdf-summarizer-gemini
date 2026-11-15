"""Phase 0 gate: prove the Gemini key, git identity and GitHub login all work.

Runs with the standard library only, so it works before `pip install`.

Usage:
    python scripts/check_setup.py
    python scripts/check_setup.py --skip-gemini     # no network / no key yet
    python scripts/check_setup.py --skip-github     # key only

Exit code 0 only when every check passes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_MODEL = "gemini-2.5-flash"

_ANSI = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _ANSI else text


OK = _c("32", "[ OK ]")
FAIL = _c("31", "[FAIL]")
WARN = _c("33", "[WARN]")
INFO = _c("36", "[ .. ]")

results: list[tuple[str, bool]] = []  # (label, passed)


def report(status: str, label: str, detail: str = "", passed: bool | None = None) -> None:
    line = f"{status} {label}"
    if detail:
        line += f" - {detail}"
    print(line)
    if passed is not None:
        results.append((label, passed))


def load_env(path: Path) -> dict[str, str]:
    """Minimal .env parser so this script needs no third-party packages."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :]
        value = value.split(" #", 1)[0].strip().strip("'\"")
        values[key] = value

    # Real environment wins over .env, matching dotenv behaviour.
    for key in ("GEMINI_API_KEY", "GEMINI_MODEL"):
        if os.environ.get(key):
            values[key] = os.environ[key]
    return values


def mask(secret: str) -> str:
    if len(secret) <= 8:
        return "*" * len(secret)
    return f"{secret[:4]}...{secret[-4:]}"


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)


def check_python() -> None:
    version = sys.version.split()[0]
    ok = sys.version_info >= (3, 10)
    report(OK if ok else FAIL, "Python >= 3.10", f"found {version}", ok)


def check_venv() -> None:
    in_venv = sys.prefix != sys.base_prefix
    report(OK if in_venv else WARN, "Virtual environment", sys.prefix, None if in_venv else False)


def check_env_file(env: dict[str, str]) -> str | None:
    if not ENV_FILE.exists():
        report(FAIL, ".env file", f"missing at {ENV_FILE}", False)
        return None

    size = ENV_FILE.stat().st_size
    if size == 0:
        report(FAIL, ".env file", "exists but is empty", False)
        return None
    report(OK, ".env file", f"{size} bytes", True)

    key = env.get("GEMINI_API_KEY", "").strip()
    if not key:
        report(FAIL, "GEMINI_API_KEY", "not set in .env", False)
        return None
    if key.lower().startswith(("your_", "sk-", "<", "changeme")):
        report(FAIL, "GEMINI_API_KEY", f"still a placeholder ({mask(key)})", False)
        return None
    report(OK, "GEMINI_API_KEY", f"present ({mask(key)}, {len(key)} chars)", True)

    if not re.fullmatch(r"(AIza[A-Za-z0-9_\-]{30,}|AQ\.[A-Za-z0-9_\-]{20,})", key):
        report(WARN, "Key format", "unusual for a Google AI Studio key - verifying live", None)
    return key


def check_gemini(key: str, env: dict[str, str]) -> None:
    model = env.get("GEMINI_MODEL") or DEFAULT_MODEL
    url = GEMINI_ENDPOINT.format(model=model)
    payload = json.dumps(
        {
            "contents": [{"role": "user", "parts": [{"text": "Reply with the single word: OK"}]}],
            "generationConfig": {"maxOutputTokens": 16, "temperature": 0},
        }
    ).encode()
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
        method="POST",
    )

    import time

    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="ignore")
        try:
            message = json.loads(detail)["error"]["message"]
        except Exception:  # noqa: BLE001 - any malformed body
            message = detail[:200] or str(exc)
        hint = ""
        if exc.code in (400, 403):
            hint = " (key invalid, restricted, or Generative Language API not enabled)"
        elif exc.code == 429:
            hint = " (quota/rate limit - key is valid but throttled)"
        elif exc.code == 404:
            hint = f" (model '{model}' not available to this key)"
        report(FAIL, f"Gemini live call ({model})", f"HTTP {exc.code}: {message}{hint}", False)
        return
    except urllib.error.URLError as exc:
        report(FAIL, f"Gemini live call ({model})", f"network error: {exc.reason}", False)
        return

    elapsed = (time.perf_counter() - started) * 1000
    try:
        text = body["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError):
        report(FAIL, f"Gemini live call ({model})", f"unexpected response: {body}", False)
        return
    report(OK, f"Gemini live call ({model})", f'replied "{text[:40]}" in {elapsed:.0f} ms', True)


def check_git() -> None:
    if not shutil.which("git"):
        report(FAIL, "git", "not installed or not on PATH", False)
        return
    version = run(["git", "--version"]).stdout.strip()
    report(OK, "git", version, True)

    name = run(["git", "config", "--global", "user.name"]).stdout.strip()
    email = run(["git", "config", "--global", "user.email"]).stdout.strip()
    if not name or not email:
        report(FAIL, "git identity", "set user.name and user.email", False)
        return
    report(OK, "git identity", f"{name} <{email}>", True)


def check_github() -> None:
    if not shutil.which("gh"):
        report(FAIL, "GitHub CLI", "not installed or not on PATH", False)
        return
    version = run(["gh", "--version"]).stdout.splitlines()[0].strip()
    report(OK, "GitHub CLI", version, True)

    status = run(["gh", "auth", "status"])
    output = f"{status.stdout}\n{status.stderr}"
    if status.returncode != 0:
        report(FAIL, "GitHub auth", "not logged in - run: gh auth login -h github.com -w", False)
        return

    account = run(["gh", "api", "user", "--jq", ".login"]).stdout.strip()
    if not account:
        report(FAIL, "GitHub auth", "token works but the account could not be resolved", False)
        return
    report(OK, "GitHub auth", f"logged in as {account}", True)

    scopes_match = re.search(r"Token scopes:\s*(.+)", output)
    scopes = [s.strip().strip("'") for s in scopes_match.group(1).split(",")] if scopes_match else []
    if not scopes:
        report(WARN, "Token scopes", "could not read scopes; expecting 'repo'", None)
    elif "repo" in scopes:
        report(OK, "Token scopes", "repo (can create and push repositories)", True)
    else:
        report(
            FAIL,
            "Token scopes",
            f"missing 'repo' (have: {', '.join(scopes)}) - run: gh auth refresh -s repo",
            False,
        )


def check_remote() -> None:
    if not shutil.which("git"):
        return
    remote = run(["git", "remote", "get-url", "origin"])
    if remote.returncode == 0 and remote.stdout.strip():
        report(OK, "git remote 'origin'", remote.stdout.strip(), None)
    else:
        report(INFO, "git remote 'origin'", "not set yet - created in Phase 0 step 7", None)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Gemini key + GitHub setup")
    parser.add_argument("--skip-gemini", action="store_true", help="skip the live API call")
    parser.add_argument("--skip-github", action="store_true", help="skip gh checks")
    args = parser.parse_args()

    print("\n=== PDF Summarizer - Phase 0 setup check ===\n")

    env = load_env(ENV_FILE)
    check_python()
    check_venv()
    key = check_env_file(env)
    if key and not args.skip_gemini:
        check_gemini(key, env)
    elif not key:
        report(FAIL, "Gemini live call", "skipped - no usable key", False)

    check_git()
    if not args.skip_github:
        check_github()
        check_remote()

    passed = sum(1 for _, ok in results if ok)
    failed = [label for label, ok in results if not ok]
    print(f"\n=== {passed}/{len(results)} checks passed ===")
    if failed:
        print("Blocking: " + ", ".join(dict.fromkeys(failed)))
        return 1
    print("All green - Phase 0 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
