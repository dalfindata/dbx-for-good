#!/usr/bin/env python3
"""Download the installed Virtue Foundation Marketplace tables as CSV files."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import TypeAlias, cast


JSONValue: TypeAlias = (
    None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
)
JSONObject: TypeAlias = dict[str, JSONValue]


PROFILE = "dbc-4bc93aff-6fa4"
WAREHOUSE_ID = "06bc70049e5864b7"
CATALOG = "databricks_virtue_foundation_dataset_dais_2026"
SCHEMA = "virtue_foundation_dataset"
OUT_DIR = Path("data/raw/virtue_foundation_dataset")
SQL_WAIT_TIMEOUT = "50s"
STATEMENT_POLL_INTERVAL_SECONDS = 5
DOWNLOAD_CHUNK_SIZE_BYTES = 1024 * 1024
FIRST_EXTERNAL_LINK_INDEX = 0
CSV_PART_INDEX_WIDTH = 5
JSON_INDENT_SPACES = 2
TABLES = [
    "facilities",
    "india_post_pincode_directory",
    "nfhs_5_district_health_indicators",
]


def run_databricks(args: list[str]) -> JSONObject:
    cmd = ["databricks", *args, "--profile", PROFILE, "-o", "json"]
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return cast(JSONObject, json.loads(result.stdout))


def post_statement(table: str) -> JSONObject:
    body = {
        "warehouse_id": WAREHOUSE_ID,
        "statement": f"SELECT * FROM {CATALOG}.{SCHEMA}.{table}",
        "format": "CSV",
        "disposition": "EXTERNAL_LINKS",
        "wait_timeout": SQL_WAIT_TIMEOUT,
        "on_wait_timeout": "CONTINUE",
    }
    return run_databricks(
        ["api", "post", "/api/2.0/sql/statements", "--json", json.dumps(body)]
    )


def get_statement(statement_id: str) -> JSONObject:
    return run_databricks(["api", "get", f"/api/2.0/sql/statements/{statement_id}"])


def get_chunk(statement_id: str, chunk_index: int) -> JSONObject:
    return run_databricks(
        [
            "api",
            "get",
            f"/api/2.0/sql/statements/{statement_id}/result/chunks/{chunk_index}",
        ]
    )


def _as_mapping(value: object) -> JSONObject:
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object, got {type(value).__name__}")
    return cast(JSONObject, value)


def _as_list(value: object) -> list[JSONValue]:
    if not isinstance(value, list):
        raise TypeError(f"Expected JSON list, got {type(value).__name__}")
    return cast(list[JSONValue], value)


def _as_str(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"Expected JSON string, got {type(value).__name__}")
    return value


def _as_int(value: object) -> int:
    if not isinstance(value, int):
        raise TypeError(f"Expected JSON integer, got {type(value).__name__}")
    return value


def wait_for_statement(response: JSONObject) -> JSONObject:
    statement_id = _as_str(response["statement_id"])
    while _as_mapping(response.get("status", {})).get("state") in {"PENDING", "RUNNING"}:
        time.sleep(STATEMENT_POLL_INTERVAL_SECONDS)
        response = get_statement(statement_id)
    state = _as_mapping(response.get("status", {})).get("state")
    if state != "SUCCEEDED":
        raise RuntimeError(f"Statement {statement_id} ended with state {state}: {response}")
    return response


def download(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url) as response, destination.open("wb") as output:
        while True:
            chunk = response.read(DOWNLOAD_CHUNK_SIZE_BYTES)
            if not chunk:
                break
            output.write(chunk)


def append_file(source: Path, destination: Path) -> None:
    with source.open("rb") as src, destination.open("ab") as dst:
        dst.write(src.read())


def export_table(table: str) -> JSONObject:
    print(f"Exporting {table}...", flush=True)
    response = wait_for_statement(post_statement(table))
    statement_id = _as_str(response["statement_id"])
    manifest = _as_mapping(response["manifest"])
    chunks = _as_list(manifest["chunks"])

    table_dir = OUT_DIR / table
    table_dir.mkdir(parents=True, exist_ok=True)
    combined_csv = table_dir / f"{table}.csv"
    metadata_path = table_dir / "manifest.json"

    if combined_csv.exists():
        combined_csv.unlink()

    result = _as_mapping(response.get("result", {}))
    first_links: dict[int, JSONObject] = {}
    for raw_link in _as_list(result.get("external_links", [])):
        initial_link = _as_mapping(raw_link)
        first_links[_as_int(initial_link["chunk_index"])] = initial_link

    downloaded_chunks: list[JSONValue] = []
    for chunk in chunks:
        chunk_info = _as_mapping(chunk)
        index = _as_int(chunk_info["chunk_index"])
        link: JSONObject | None = first_links.get(index)
        if link is None:
            chunk_response = get_chunk(statement_id, index)
            chunk_result = _as_mapping(chunk_response.get("result", {}))
            links = _as_list(
                chunk_response.get("external_links", chunk_result.get("external_links", []))
            )
            if not links:
                raise RuntimeError(f"No external link returned for {table} chunk {index}")
            link = _as_mapping(links[FIRST_EXTERNAL_LINK_INDEX])

        chunk_path = table_dir / f"{table}.part-{index:0{CSV_PART_INDEX_WIDTH}d}.csv"
        download(_as_str(link["external_link"]), chunk_path)
        append_file(chunk_path, combined_csv)
        downloaded_chunks.append(
            {
                "chunk_index": index,
                "row_count": chunk_info["row_count"],
                "byte_count": chunk_info["byte_count"],
                "file": str(chunk_path),
            }
        )
        print(f"  chunk {index}: {chunk_info['row_count']} rows", flush=True)

    metadata: JSONObject = {
        "table": f"{CATALOG}.{SCHEMA}.{table}",
        "statement_id": statement_id,
        "total_row_count": manifest["total_row_count"],
        "total_byte_count": manifest["total_byte_count"],
        "total_chunk_count": manifest["total_chunk_count"],
        "combined_csv": str(combined_csv),
        "chunks": downloaded_chunks,
    }
    metadata_path.write_text(json.dumps(metadata, indent=JSON_INDENT_SPACES) + "\n")
    print(f"  wrote {combined_csv}", flush=True)
    return metadata


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = [export_table(table) for table in TABLES]
    (OUT_DIR / "download_manifest.json").write_text(
        json.dumps(summary, indent=JSON_INDENT_SPACES) + "\n"
    )
    print(f"Wrote manifest: {OUT_DIR / 'download_manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
