"""IPython/Jupyter cell magic for RelataDB (#967 CX).

Register with::

    %load_ext relata.ipython

Then use in a cell::

    %%relata --purpose analytics
    SELECT * FROM Person LIMIT 10

Or line magic::

    %relata SELECT * FROM Person LIMIT 5

Results are returned as a pandas DataFrame when pandas is installed,
otherwise as a list of dicts. The server URL and token are read from
environment variables ``RELATA_URL`` (default ``http://localhost:9090``)
and ``RELATA_TOKEN``.

Configuration::

    %relata_config --url http://my-server:9090 --token my-secret
"""

from __future__ import annotations

import os
from typing import Any

try:
    from IPython.core.magic import Magics, magics_class, line_magic, cell_magic, line_cell_magic
    from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False


def _get_client(url: str | None, token: str | None):
    """Lazily import and construct a RelataClient."""
    from relata import RelataClient
    base = url or os.environ.get("RELATA_URL", "http://localhost:9090")
    tok = token or os.environ.get("RELATA_TOKEN", "")
    return RelataClient(base, bearer_token=tok or None)


def _to_dataframe(rows: list[dict[str, Any]]):
    """Convert rows to a pandas DataFrame if available, else return list."""
    try:
        import pandas as pd
        return pd.DataFrame(rows)
    except ImportError:
        return rows


if HAS_IPYTHON:

    @magics_class
    class RelataMagics(Magics):
        """``%%relata`` cell magic — execute SQL against a Relata server."""

        def __init__(self, shell):
            super().__init__(shell)
            self._url: str | None = None
            self._token: str | None = None
            self._purpose: str = "analytics"

        @line_magic
        def relata_config(self, line: str):
            """Configure the Relata connection.

            Usage::

                %relata_config --url http://localhost:9090 --token secret --purpose analytics
            """
            args = line.split()
            i = 0
            while i < len(args):
                if args[i] == "--url" and i + 1 < len(args):
                    self._url = args[i + 1]; i += 2
                elif args[i] == "--token" and i + 1 < len(args):
                    self._token = args[i + 1]; i += 2
                elif args[i] == "--purpose" and i + 1 < len(args):
                    self._purpose = args[i + 1]; i += 2
                else:
                    i += 1
            print(f"Relata config: url={self._url or '(env)'}, purpose={self._purpose}")

        @line_cell_magic
        def relata(self, line: str, cell: str | None = None):
            """Execute SQL against Relata and return results as a DataFrame.

            Line magic::

                %relata SELECT * FROM Person LIMIT 5

            Cell magic::

                %%relata --purpose analytics
                SELECT name, email FROM Person
                WHERE name LIKE 'A%'
                LIMIT 10

            Options:
                --purpose <p>   Override the default purpose
                --url <url>     Override the server URL
                --token <tok>   Override the bearer token
            """
            # Parse line arguments.
            purpose = self._purpose
            url = self._url
            token = self._token
            parts = line.split()
            i = 0
            while i < len(parts):
                if parts[i] == "--purpose" and i + 1 < len(parts):
                    purpose = parts[i + 1]; i += 2
                elif parts[i] == "--url" and i + 1 < len(parts):
                    url = parts[i + 1]; i += 2
                elif parts[i] == "--token" and i + 1 < len(parts):
                    token = parts[i + 1]; i += 2
                else:
                    i += 1

            # The SQL is the cell body (for %%relata) or the line (for %relata).
            sql = (cell or line).strip()
            if not sql:
                print("Usage: %%relata [--purpose <p>]\\n<SQL>")
                return None

            client = _get_client(url, token)
            try:
                result = client.query(sql, purpose=purpose)
                rows = [dict(r) for r in result]
                if not rows:
                    print("(0 rows)")
                    return []
                df = _to_dataframe(rows)
                print(f"({len(rows)} rows)")
                return df
            except Exception as e:
                print(f"Error: {e}")
                return None


    def load_ipython_extension(ipython):
        """Register the magic when the user runs %load_ext relata.ipython."""
        ipython.register_magics(RelataMagics)
        print("✓ %relata magic loaded. Try: %%relata\\nSELECT * FROM Person LIMIT 5")
        print("  Configure: %relata_config --url <url> --token <token>")

else:
    def load_ipython_extension(ipython):
        """Stub — IPython not installed."""
        print("IPython is not installed. Install with: pip install ipython jupyter")
