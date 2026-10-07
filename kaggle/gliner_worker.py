"""GLiNER2.5 worker: runs INSIDE an isolated venv (its own transformers<5).

Protocol (argv): cases.jsonl rows_out.jsonl
Reads decision cases, writes BackendRow JSONL. Kept dependency-free of the
main kernel env: the parent installs this venv with the mokka wheel +
gliner2[local] + transformers<5, then invokes this file with that venv's
python. The main environment (trainer, future VLMs) stays untouched.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def main() -> int:
    cases_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    sys.path.insert(0, str(Path(__file__).parent))

    from mokka_decisions.backends.gliner import GlinerBackend
    from mokka_decisions.contracts import DecisionCase
    from mokka_decisions.evaluate import run_backend

    cases = [DecisionCase.from_dict(json.loads(ln)) for ln in cases_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    t0 = time.time()
    backend = GlinerBackend()
    rows = run_backend(backend, cases)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    errors = sum(1 for r in rows if r.error)
    print(f"[gliner-worker] {len(cases)} cases in {time.time()-t0:.0f}s, errors={errors}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
