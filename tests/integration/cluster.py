"""SSH / .env helpers for the PCSS Eagle integration tests.

Kept local to the integration suite so the tests are self-contained and don't
depend on anything at the repo root. Parses the same ./.env that connect.sh
uses and materialises the private key into a temp file for OpenSSH.
"""
from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path

# SSH options shared by every connection. We disable host-key prompts so the
# tests never block on an interactive "yes/no", and keep known_hosts out of the
# user's real file (the cluster login node is a shared, well-known host).
SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "LogLevel=ERROR",
    "-o", "IdentitiesOnly=yes",
]


def load_env(path: str | os.PathLike = ".env") -> dict[str, str]:
    """Parse a dotenv file into a dict.

    Supports single-line ``KEY="value"`` / ``KEY=value`` pairs and multi-line
    double-quoted values (used for SSH_KEY). Comment lines starting with '#'
    are ignored; inline '# ...' comments after an unquoted value are stripped.
    """
    text = Path(path).read_text(encoding="utf-8")
    env: dict[str, str] = {}

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1

        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            continue

        key, _, rest = line.partition("=")
        key = key.strip()
        if not key:
            continue
        rest = rest.lstrip()

        if rest.startswith('"'):
            # Double-quoted value, possibly spanning multiple lines until the
            # closing quote.
            body = rest[1:]
            if '"' in body:
                value = body[: body.index('"')]
            else:
                parts = [body]
                while i < len(lines):
                    nxt = lines[i]
                    i += 1
                    if '"' in nxt:
                        parts.append(nxt[: nxt.index('"')])
                        break
                    parts.append(nxt)
                value = "\n".join(parts)
        elif rest.startswith("'"):
            body = rest[1:]
            value = body[: body.index("'")] if "'" in body else body
        else:
            # Unquoted: strip inline comment and surrounding whitespace.
            value = rest.split("#", 1)[0].strip()

        env[key] = value

    return env


def write_keyfile(key: str) -> str:
    """Write a private key to a fresh temp file with 0600 perms; return path.

    OpenSSH rejects keys that are world-readable or lack a trailing newline,
    so we normalise the terminating newline and re-assert the mode.
    """
    # mkstemp already creates the file with 0600; re-assert it defensively for
    # platforms/umasks where that can't be relied on (os.chmod is portable;
    # os.fchmod is Unix-only).
    fd, path = tempfile.mkstemp(prefix="pcss_key_", suffix=".pem")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(key.rstrip("\n") + "\n")
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)  # 0600
    except Exception:
        try:
            os.unlink(path)
        finally:
            raise
    return path
