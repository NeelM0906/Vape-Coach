"""vape-coach core: surface-independent logic (storage, taper, interventions, gating, detection).

Nothing in this package imports Hermes at module level, so the core runs (and is tested)
without a Hermes install; ``hermes_glue`` is the only module that touches ``ctx``.
"""

__all__ = ["PLUGIN_NAME"]

PLUGIN_NAME = "vape-coach"
