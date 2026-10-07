"""Source dataset loading into a unified utterance table.

Downloads from the *official* sources (the dataset scripts' own URLs), not from
HF dataset-script infrastructure (unsupported in `datasets>=3`):

* BANKING77: CSVs from the PolyAI-LDN task-specific-datasets repository;
* MASSIVE 1.0: the amazon-massive-dataset tarball from Amazon S3 (per-locale
  JSONL inside), only the configured locales are extracted;
* CLINC150/OOS 'plus' config: HF-hosted parquet conversion.

Everything is cached under ``data/raw/`` (gitignored). Each loader normalises
to one row per utterance *before* any candidate-set expansion.

Unified row shape::

    {"id", "group_id", "source", "language", "domain", "text",
     "label", "official_split", "label_origin"}
"""

from __future__ import annotations

import csv
import io
import json
import tarfile
import urllib.request
from pathlib import Path
from typing import Iterator

SOURCE_META: dict[str, dict[str, str]] = {
    "banking77": {
        "repo": "PolyAI/banking77",
        "license": "CC BY 4.0",
        "url": "https://github.com/PolyAI-LDN/task-specific-datasets (banking_data)",
        "domain": "banking",
        "languages": "en",
    },
    "massive": {
        "repo": "AmazonScience/massive",
        "license": "CC BY 4.0",
        "url": "https://github.com/alexa/massive (dataset 1.0)",
        "domain": "assistants",
        "languages": "en,de,es",
    },
    "clinc": {
        "repo": "clinc/clinc_oos",
        "license": "CC BY 3.0",
        "url": "https://huggingface.co/datasets/clinc/clinc_oos (parquet, plus)",
        "domain": "assistants",
        "languages": "en",
    },
}

MASSIVE_LOCALES = ("en-US", "de-DE", "es-ES")
RAW_DIR = Path("data/raw")

BANKING77_TRAIN_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"
)
BANKING77_TEST_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"
)
MASSIVE_URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz"
BANKING77_LABELS_URL = "https://huggingface.co/datasets/PolyAI/banking77/raw/main/banking77.py"
CLINC_PARQUET = (
    "https://huggingface.co/api/datasets/clinc/clinc_oos/parquet/{config}/{split}/0.parquet"
)


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
    return dest


def banking77_label_names() -> list[str]:
    """Canonical 77 label names, extracted from the official dataset script."""
    script = _download(BANKING77_LABELS_URL, RAW_DIR / "banking77_script.py")
    src = script.read_text(encoding="utf-8")
    match = None
    import re

    for match in re.finditer(r"names=\[(.*?)\]", src, re.S):
        pass  # take the last names=[...] block (the label ClassLabel)
    if match is None:
        raise RuntimeError("could not find label names in banking77.py")
    return re.findall(r'"([^"]+)"', match.group(1))


def load_banking77() -> Iterator[dict]:
    """10,003 train / 3,080 test, 77 intents, English (official CSVs).

    The CSV 'category' column already carries the canonical label string;
    we cross-check it against the 77 names from the official HF script.
    """
    canonical = set(banking77_label_names())
    for official, url in (("train", BANKING77_TRAIN_URL), ("test", BANKING77_TEST_URL)):
        path = _download(url, RAW_DIR / f"banking77_{official}.csv")
        reader = csv.DictReader(io.StringIO(path.read_text(encoding="utf-8")))
        for i, row in enumerate(reader):
            label = row["category"]
            if label not in canonical:
                raise RuntimeError(
                    f"banking77 label {label!r} not in the official 77-name list"
                )
            yield {
                "id": f"bank-{official}-{i:05d}",
                "group_id": f"bank-{official}-{i:05d}",
                "source": "banking77",
                "language": "en",
                "domain": "banking",
                "text": row["text"].strip(),
                "label": label,
                "official_split": official,
                "label_origin": "public-dataset",
            }


def _massive_jsonl(locale: str) -> Path:
    tar_path = _download(MASSIVE_URL, RAW_DIR / "amazon-massive-dataset-1.0.tar.gz")
    out = RAW_DIR / f"massive-{locale}.jsonl"
    if not out.exists():
        member = f"1.0/data/{locale}.jsonl"
        with tarfile.open(tar_path, "r:gz") as tar:
            extracted = tar.extractfile(member)
            if extracted is None:
                raise KeyError(f"{member} not in tarball")
            out.write_bytes(extracted.read())
    return out


def load_massive(locales: tuple[str, ...] = MASSIVE_LOCALES) -> Iterator[dict]:
    """MASSIVE 1.0 intents; ``id`` is aligned across locales -> shared group_id."""
    lang_of = {"en-US": "en", "de-DE": "de", "es-ES": "es"}
    official_of = {"train": "train", "dev": "validation", "test": "test"}
    for locale in locales:
        path = _massive_jsonl(locale)
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                partition = row.get("partition") or row.get("split")
                if partition not in official_of:
                    continue
                yield {
                    "id": f"massive-{locale}-{row['id']}",
                    "group_id": f"massive-{row['id']}",
                    "source": "massive",
                    "language": lang_of[locale],
                    "domain": "assistants",
                    "text": row["utt"].strip(),
                    "label": row["intent"],
                    "official_split": official_of[partition],
                    "label_origin": "public-dataset",
                }


def load_clinc() -> Iterator[dict]:
    """CLINC150+OOS, 'plus' config (oos-augmented splits), via HF parquet.

    Intent 150 (the 151st class) is the out-of-scope label; kept as 'oos'.
    """
    import pyarrow.parquet as pq

    names = None
    for official in ("train", "validation", "test"):
        url = CLINC_PARQUET.format(config="plus", split=official)
        path = _download(url, RAW_DIR / f"clinc_plus_{official}.parquet")
        table = pq.read_table(path)
        if names is None:
            # metadata carries the ClassLabel list (huggingface metadata)
            meta = table.schema.metadata or {}
            hf_info = meta.get(b"huggingface", b"{}")
            info = json.loads(hf_info.decode())
            try:
                names = info["info"]["features"]["intent"]["names"]
            except (KeyError, TypeError):
                names = None
        rows = table.to_pylist()
        for i, row in enumerate(rows):
            label_idx = row["intent"]
            if names is not None:
                label = names[label_idx] if isinstance(label_idx, int) else str(label_idx)
            else:
                label = str(label_idx)
            yield {
                "id": f"clinc-{official}-{i:05d}",
                "group_id": f"clinc-{official}-{i:05d}",
                "source": "clinc",
                "language": "en",
                "domain": "assistants",
                "text": row["text"].strip(),
                "label": label,
                "official_split": official,
                "label_origin": "public-dataset",
            }


def label_is_oos(label: str) -> bool:
    return label.strip().lower() in {"oos", "out_of_scope", "150"}


LOADERS = {"banking77": load_banking77, "massive": load_massive, "clinc": load_clinc}


def load_source(source: str) -> Iterator[dict]:
    if source not in LOADERS:
        raise KeyError(f"unknown source {source!r}; known: {sorted(LOADERS)}")
    return LOADERS[source]()
