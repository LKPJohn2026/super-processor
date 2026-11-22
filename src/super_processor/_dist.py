"""The name this package is published under on PyPI.

The import package and the command are ``super_processor`` and
``super-processor``, but the distribution on PyPI is ``ancient-processor``.
Every install hint is built from :data:`DIST_NAME`, so none can point users
at a different (possibly someone else's) package.
"""

from __future__ import annotations

DIST_NAME = "ancient-processor"
GPU_INSTALL_HINT = f'python -m pip install "{DIST_NAME}[gpu]"'
