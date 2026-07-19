import os
from typing import Final

RTSP_SCHEME: Final[str] = "rtsp"
RTSP_PORT: Final[int] = 554
DEFAULT_STREAM: Final[str] = "stream1"


def build_rtsp_url(user: str, password: str, host: str, stream: str = DEFAULT_STREAM) -> str:
    """Build an RTSP URL for the Tapo camera."""
    return f"{RTSP_SCHEME}://{user}:{password}@{host}:{RTSP_PORT}/{stream}"


def require_env(name: str) -> str:
    """Return an environment variable or raise a clear error if it is missing."""
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value
