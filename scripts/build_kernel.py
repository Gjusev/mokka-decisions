"""Assemble the single-file Kaggle kernel.

Kaggle script kernels only materialise ``code_file`` as /kaggle/src/script.py;
sibling modules in the push folder are NOT importable. This builder
concatenates:

* ``kaggle/run.py``      (dispatcher + textual stages; entry call moved last)
* ``kaggle/mm_pilot.py`` (multimodal stages; defs only)

and embeds ``kaggle/gliner_worker.py`` as a JSON string that the baselines
stage writes to /tmp and runs inside the isolated venv. The mm ``STAGES`` dict
is merged inside ``main()``; the entry ``main()`` call is appended at the very
end so every module-level definition exists before execution.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = ROOT / "kaggle"
OUT = KAGGLE / "kernel_flat"

ENTRY = 'if __name__ == "__main__":\n    main()\n'


def strip_module_boilerplate(src: str) -> str:
    src = re.sub(r'^"""(?:.|\n)*?"""\n', "", src, count=1)
    src = src.replace("from __future__ import annotations\n", "")
    return src


def main() -> None:
    run_src = (KAGGLE / "run.py").read_text(encoding="utf-8")
    mm_src = strip_module_boilerplate((KAGGLE / "mm_pilot.py").read_text(encoding="utf-8"))
    worker_src = (KAGGLE / "gliner_worker.py").read_text(encoding="utf-8")

    # 1. drop the mm import (STAGES arrives inline at module level)
    run_src = run_src.replace(
        "    from mm_pilot import STAGES as MM_STAGES\n\n    stages.update(MM_STAGES)\n",
        "    stages.update(STAGES)  # inlined mm_pilot stages (defined below)\n",
    )
    # 2. worker source: written to /tmp at runtime instead of a sibling file
    run_src = run_src.replace(
        '        worker = Path(__file__).parent / "gliner_worker.py"',
        '        worker = Path("/tmp/gliner_worker.py")\n'
        '        worker.write_text(json.loads(GLINER_WORKER_SRC), encoding="utf-8")',
    )
    # 3. move the entry call to the absolute end
    assert ENTRY in run_src, "entry point not found in run.py"
    run_src = run_src.replace(ENTRY, "")

    flat = (
        run_src.rstrip("\n")
        + "\n\n# ==== inlined: multimodal pilot stages (kaggle/mm_pilot.py) ====\n"
        + mm_src.rstrip("\n")
        + "\n\n# ==== embedded: gliner worker source (runs inside the isolated venv) ====\n"
        + f"GLINER_WORKER_SRC = {json.dumps(worker_src)}\n\n"
        + ENTRY
    )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "run.py").write_text(flat, encoding="utf-8")
    meta = json.loads((KAGGLE / "kernel-metadata.json").read_text(encoding="utf-8"))
    (OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    import ast

    ast.parse(flat)
    print(f"flat kernel: {OUT / 'run.py'} ({len(flat.splitlines())} lines, parses OK)")


if __name__ == "__main__":
    main()
