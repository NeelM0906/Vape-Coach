"""vape-coach: a local-first webcam vape tracker that tapers you off with nags, roasts and jumping jacks.

Entry point for the Hermes plugin loader. All logic lives in ``vapecoach/``; this file only wires
it to ``ctx`` (see ``vapecoach/hermes_glue.py``).
"""

if __package__:  # loaded by Hermes as a package
    from .vapecoach.hermes_glue import register  # noqa: F401
else:  # imported as a bare module (e.g. pytest rootdir collection)
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
    from vapecoach.hermes_glue import register  # noqa: F401
