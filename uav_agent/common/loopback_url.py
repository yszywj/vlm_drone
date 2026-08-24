"""Strict local HTTP endpoint validation for image-bearing model services."""

from __future__ import annotations

from urllib.parse import urlsplit


def validate_loopback_http_url(value: object, field_name: str = "url") -> str:
    """Return one canonical loopback URL or fail closed.

    Parsing the authority is essential: string-prefix checks accept values
    such as ``http://127.0.0.1:8011@evil.example`` whose actual hostname is
    remote.  Model service URLs intentionally support no credentials, path,
    query, fragment, implicit port, HTTPS downgrade ambiguity, or IPv6 alias.
    """

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty URL")
    raw = value.strip()
    try:
        parsed = urlsplit(raw)
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a valid loopback HTTP URL") from exc
    if parsed.scheme.casefold() != "http":
        raise ValueError(f"{field_name} must use loopback HTTP")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{field_name} must not contain user information")
    hostname = None if parsed.hostname is None else parsed.hostname.casefold()
    if hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError(f"{field_name} must use loopback HTTP")
    if port is None or not 1 <= port <= 65_535:
        raise ValueError(f"{field_name} must contain an explicit valid port")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError(
            f"{field_name} must not contain a path, query, or fragment"
        )
    return f"http://{hostname}:{port}"


__all__ = ["validate_loopback_http_url"]
