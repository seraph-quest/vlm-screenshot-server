from __future__ import annotations

import os
import sys
from urllib.parse import urlparse


def main() -> int:
    base_url = os.getenv("VLM_BASE_URL", "").strip()
    if not base_url:
        return 0
    parsed = urlparse(base_url)
    if parsed.hostname not in {"127.0.0.1", "localhost"}:
        return 0
    if os.path.exists("/.dockerenv") and os.getenv("VLM_ALLOW_DOCKER_LOCALHOST", "").lower() not in {"1", "true"}:
        print(
            "VLM_BASE_URL points at container-local localhost. "
            "For bridge-mode Docker use host.docker.internal, or set "
            "VLM_ALLOW_DOCKER_LOCALHOST=true only when the wrapper runs with "
            "Linux host networking.",
            file=sys.stderr,
        )
        return 78
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
