"""Pin public bilingual sources and extract bounded raw samples via HTTP Range."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import random
import re
import time

import pyarrow as pa
import pyarrow.parquet as pq
import requests

ROOT = Path(__file__).resolve().parent / "data/pretrain_sources"
SPECS = [("fineweb-2", "cmn_Hani", "zh"), ("fineweb-edu", "sample-10BT", "en")]


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RangeFile(io.RawIOBase):
    """Seekable remote file with strict range and byte-budget validation."""

    def __init__(self, url, size, max_bytes=512 * 1024 * 1024):
        self.url, self.size, self.pos = url, size, 0
        self.max_bytes, self.downloaded, self.requests = max_bytes, 0, 0
        self.session = requests.Session()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        pos = offset + (self.pos if whence == 1 else self.size if whence == 2 else 0)
        if whence not in (0, 1, 2) or not 0 <= pos <= self.size:
            raise ValueError("Invalid source seek")
        self.pos = pos
        return pos

    def read(self, size=-1):
        count = self.size - self.pos if size < 0 else min(size, self.size - self.pos)
        if count == 0:
            return b""
        if self.downloaded + count > self.max_bytes:
            raise RuntimeError("Sample exceeded its bounded network budget")
        start, end = self.pos, self.pos + count - 1
        for attempt in range(3):
            try:
                with self.session.get(self.url, headers={"Range": f"bytes={start}-{end}"},
                                      timeout=(15, 60), stream=True) as response:
                    response.raise_for_status()
                    content_range = response.headers.get("Content-Range", "")
                    expected = f"bytes {start}-{end}/{self.size}"
                    if response.status_code != 206 or content_range != expected:
                        raise RuntimeError("Source did not honor exact byte range")
                    data = response.raw.read(count + 1)
                    if len(data) != count:
                        raise RuntimeError("Truncated or excessive range response")
                break
            except requests.RequestException:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
        self.pos += count
        self.downloaded += count
        self.requests += 1
        return data

    def close(self):
        self.session.close()
        super().close()


def make_lock():
    sources = []
    for name, config, language in SPECS:
        meta = json.loads((ROOT / f"{name}.metadata.json").read_text())
        files = json.loads((ROOT / f"{name}.pilot_files.json").read_text())
        if meta.get("gated"):
            raise RuntimeError("Source unexpectedly gated")
        prefix = "data/cmn_Hani/train/" if name == "fineweb-2" else "sample/10BT/"
        selected = []
        for item in files:
            if item["type"] != "file" or not item["path"].startswith(prefix):
                raise RuntimeError("Source manifest contains an unexpected path")
            oid = item["lfs"]["oid"]
            if not re.fullmatch(r"[0-9a-f]{64}", oid):
                raise RuntimeError("Missing full-source hash")
            selected.append({"path": item["path"], "bytes": item["size"],
                             "source_sha256": oid})
        sources.append({"repo": f"HuggingFaceFW/{name}", "revision": meta["sha"],
                        "config": config, "source_split": "train", "language": language,
                        "license": "ODC-BY", "verified_listing_prefix_only": True,
                        "candidate_files": selected})
    lock = {"version": "r24-pretrain-source-lock-v1", "sources": sources,
            "raw_document_format": "r24-pretrain-document-v1",
            "synthetic_guard_record_schema_changed": False}
    path = ROOT / "sources_lock.json"
    if path.exists() and json.loads(path.read_text()) != lock:
        raise RuntimeError("Refusing to silently replace frozen source lock")
    path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n")
    return lock


def extract(source, rows):
    file = source["candidate_files"][0]
    url = (f"https://huggingface.co/datasets/{source['repo']}/resolve/"
           f"{source['revision']}/{file['path']}")
    out = ROOT / "raw_samples"
    out.mkdir(exist_ok=True)
    target = out / f"{source['language']}-{rows}.parquet"
    manifest_path = target.with_suffix(".manifest.json")
    if target.exists() or manifest_path.exists():
        if target.exists() and manifest_path.exists():
            old = json.loads(manifest_path.read_text())
            if old["source"] == source and file_sha(target) == old["output_sha256"]:
                return old
        raise RuntimeError("Existing sample is incomplete or differs from lock")
    part = target.with_suffix(".parquet.part")
    picked, tables = [], []
    with RangeFile(url, file["bytes"]) as remote:
        parquet = pq.ParquetFile(remote, pre_buffer=False)
        groups = list(range(parquet.num_row_groups))
        random.Random(20260923).shuffle(groups)
        count = 0
        for group in groups:
            table = parquet.read_row_group(group, columns=["text", "id", "url"],
                                           use_threads=False)
            take = min(table.num_rows, rows - count)
            table = table.slice(0, take)
            table = table.append_column("source_row_group", pa.array([group] * take))
            table = table.append_column("row_in_group", pa.array(range(take)))
            tables.append(table)
            picked.append({"index": group, "rows": take})
            count += take
            if count == rows:
                break
        if count != rows:
            raise RuntimeError("Source has too few rows for requested sample")
        traffic = {"range_bytes": remote.downloaded, "range_requests": remote.requests}
    table = pa.concat_tables(tables)
    table = table.replace_schema_metadata({b"record_version": b"r24-pretrain-document-v1",
                                           b"source": json.dumps(source).encode()})
    pq.write_table(table, part, compression="zstd", row_group_size=1000)
    if pq.read_metadata(part).num_rows != rows:
        raise RuntimeError("Written Parquet row count mismatch")
    content = table.column("text").to_pylist()
    fingerprints = [hashlib.sha256((text or "").encode()).hexdigest() for text in content]
    summary = {"source": source, "source_file": file, "rows": rows,
               "selected_row_groups": picked, "output": str(target),
               "output_sha256": file_sha(part), "whole_source_sha256_verified": False,
               "integrity_scope": "pinned revision; exact HTTP ranges; parsed Parquet; local output hash",
               "empty_texts": sum(not text or not text.strip() for text in content),
               "duplicate_texts_exact": len(fingerprints) - len(set(fingerprints)),
               "characters": sum(len(text or "") for text in content),
               "token_count": None, "training_ready": False,
               "remaining": ["quality filtering", "deduplication and evaluation decontamination",
                             "document-family train/validation split", "frozen tokenizer and packing"],
               **traffic}
    part.rename(target)
    manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-sample", action="store_true")
    parser.add_argument("--rows-per-source", type=int, default=10000)
    args = parser.parse_args()
    if not 1 <= args.rows_per_source <= 100000:
        parser.error("Use 1–100000 rows per source for bounded sample extraction")
    lock = make_lock()
    if args.download_sample:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(extract, source, args.rows_per_source)
                       for source in lock["sources"]]
            for future in futures:
                summary = future.result()
                print(json.dumps({key: summary[key] for key in
                                  ("output", "rows", "characters", "range_bytes",
                                   "empty_texts", "duplicate_texts_exact", "training_ready")}), flush=True)
    else:
        print(json.dumps({"lock": str(ROOT / "sources_lock.json"), "sources": len(lock["sources"])}))


if __name__ == "__main__":
    main()
