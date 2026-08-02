"""Single-source SDK version string.

Extracted from ``relata/__init__.py`` so internal modules (e.g. ``_http.py``,
which builds the ``User-Agent`` header) can import ``__version__`` without a
circular import through the package ``__init__`` (#2757). This must be kept
in lockstep with ``Cargo.toml``'s ``[workspace.package].version`` — the SDK
canonical version, enforced by ``scripts/check_versions.py`` — and with
``pyproject.toml``'s ``version`` field.
"""

from __future__ import annotations

__version__ = "2.0.0"
