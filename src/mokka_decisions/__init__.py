"""mokka-decisions: state + question + option descriptions -> calibrated decision.

Public contract lives in :mod:`mokka_decisions.contracts`; everything else is
an implementation detail. Optional backends (torch, laya, gliner, sklearn) are
imported lazily so a missing optional dependency never breaks the contract.
"""

__version__ = "0.1.0"
