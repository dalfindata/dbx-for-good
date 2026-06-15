#!/usr/bin/env python3
"""Download the installed Virtue Foundation Marketplace tables as CSV files."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


PROFILE = "dbc-4bc93aff-6fa4"
WAREHOUSE_ID = "06bc70049e5864b7"
CATALOG = "databricks_virtue_foundation_dataset_dais_2026"
SCHEMA = "virtue_foundation_dataset"
OUT_DIR = Path("data/raw/virtue_foundation_dataset")
TABLES = [
    "facilities",
    "india_post_pincode_directory",
    "nfhs_5_district_health_indicators",
]


def run_databricks(args: list[str]) -> dict:
    cmd = ["databricks", *args, "--profile", PROFILE, "-o", "json"]
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return json.loads(result.stdout)


def post_statement(table: str) -> dict:
    body = {
        "warehouse_id": WAREHOUSE_ID,
        "statement": f"SELECT * FROM {CATALOG}.{SCHEMA}.{table}",
        "format": "CSV",
        "disposition": "EXTERNAL_LINKS",
        "wait_timeout": "50s",
        "on_wait_timeout": "CONTINUE",
    }
    return run_databricks(
        ["api", "post", "/api/2.0/sql/statements", "--json", json.dumps(body)]
    )


def get_statement(statement_id: str) -> dict:
    return run_databricks(["api", "get", f"/api/2.0/sql/statements/{statement_id}"])


def get_chunk(statement_id: str, chunk_index: int) -> dict:
    return run_databricks(
        [
            "api",
            "get",
            f"/api/2.0/sql/statements/{statement_id}/result/chunks/{chunk_index}",
        ]
    )


def wait_for_statement(response: dict) -> dict:
    statement_id = response["statement_id"]
    while response.get("status", {}).get("state") in {"PENDING", "RUNNING"}:
        time.sleep(5)
        response = get_statement(statement_id)
    state = response.get("status", {}).get("state")
    if state != "SUCCEEDED":
        raise RuntimeError(f"Statement {statement_id} ended with state {state}: {response}")
    return response


def download(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url) as response, destination.open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)


def append_file(source: Path, destination: Path) -> None:
    with source.open("rb") as src, destination.open("ab") as dst:
        dst.write(src.read())


def export_table(table: str) -> dict:
    print(f"Exporting {table}...", flush=True)
    response = wait_for_statement(post_statement(table))
    statement_id = response["statement_id"]
    manifest = response["manifest"]
    chunks = manifest["chunks"]

    table_dir = OUT_DIR / table
    table_dir.mkdir(parents=True, exist_ok=True)
    combined_csv = table_dir / f"{table}.csv"
    metadata_path = table_dir / "manifest.json"

    if combined_csv.exists():
        combined_csv.unlink()

    first_links = {
        link["chunk_index"]: link
        for link in response.get("result", {}).get("external_links", [])
    }
    downloaded_chunks = []
    for chunk in chunks:
        index = chunk["chunk_index"]
        link = first_links.get(index)
        if link is None:
            chunk_response = get_chunk(statement_id, index)
            links = chunk_response.get("external_links") or chunk_response.get("result", {}).get(
                "external_links", []
            )
            if not links:
                raise RuntimeError(f"No external link returned for {table} chunk {index}")
            link = links[0]

        chunk_path = table_dir / f"{table}.part-{index:05d}.csv"
        download(link["external_link"], chunk_path)
        append_file(chunk_path, combined_csv)
        downloaded_chunks.append(
            {
                "chunk_index": index,
                "row_count": chunk["row_count"],
                "byte_count": chunk["byte_count"],
                "file": str(chunk_path),
            }
        )
        print(f"  chunk {index}: {chunk['row_count']} rows", flush=True)

    metadata = {
        "table": f"{CATALOG}.{SCHEMA}.{table}",
        "statement_id": statement_id,
        "total_row_count": manifest["total_row_count"],
        "total_byte_count": manifest["total_byte_count"],
        "total_chunk_count": manifest["total_chunk_count"],
        "combined_csv": str(combined_csv),
        "chunks": downloaded_chunks,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"  wrote {combined_csv}", flush=True)
    return metadata


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = [export_table(table) for table in TABLES]
    (OUT_DIR / "download_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote manifest: {OUT_DIR / 'download_manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
