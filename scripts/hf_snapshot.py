#!/usr/bin/env python3
"""Download one immutable HF snapshot and verify its model weights."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys


ALLOWED_SUFFIXES = (
    ".safetensors", ".json", ".model", ".tiktoken", ".vocab", ".txt",
)
DEFAULT_ENDPOINT = "https://hf-mirror.com"


def endpoint():
    return (os.environ.get("HF_ENDPOINT") or DEFAULT_ENDPOINT).rstrip("/")


def wanted(path):
    parts = PurePosixPath(path).parts
    return len(parts) == 1 and not parts[0].startswith(".") and parts[0].endswith(ALLOWED_SUFFIXES)


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_tree(directory, entries):
    """Verify expected size for every file and HF LFS digest for weights."""
    record = []
    weights = 0
    for entry in sorted(entries, key=lambda item: item.path):
        relative = PurePosixPath(entry.path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid repository file path")
        path = directory.joinpath(*relative.parts)
        if path.is_symlink() or not path.is_file():
            raise ValueError("downloaded file missing or symbolic link")
        size = path.stat().st_size
        if entry.size is None or size != entry.size or size <= 0:
            raise ValueError("downloaded file size mismatch")
        checksum = file_digest(path)
        if entry.path.endswith(".safetensors"):
            weights += 1
            remote_sha = (entry.lfs or {}).get("sha256")
            if not remote_sha or checksum.lower() != remote_sha.lower():
                raise ValueError("safetensors LFS SHA-256 mismatch or unavailable")
        record.append({"path": entry.path, "size_bytes": size, "sha256": checksum})
    if weights == 0:
        raise ValueError("repository has no safetensors weights")
    return record


def run(repo_id, revision, directory, report_path):
    # Delay importing Hub until the conda environment is activated.
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.hf_api import RepoFile

    token = os.environ.get("HF_TOKEN") or None
    hub_endpoint = endpoint()
    api = HfApi(endpoint=hub_endpoint, token=token)
    info = api.model_info(repo_id=repo_id, revision=revision)
    if info.sha.lower() != revision.lower():
        raise ValueError("repository commit differs from requested revision")

    entries = [
        item for item in api.list_repo_tree(
            repo_id=repo_id, repo_type="model", revision=revision,
            recursive=True, expand=True, token=token,
        ) if isinstance(item, RepoFile) and wanted(item.path)
    ]
    if not entries:
        raise ValueError("repository has no supported conversion files")
    paths = [entry.path for entry in entries]
    snapshot_download(
        repo_id=repo_id, repo_type="model", revision=revision,
        local_dir=str(directory), allow_patterns=paths, token=token,
        max_workers=4, endpoint=hub_endpoint,
    )
    record = verify_tree(directory, entries)
    report = {"repository": repo_id, "revision": revision, "files": record}
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    print("Verified", len(record), "downloaded files at", directory)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    try:
        run(args.repo, args.revision, args.directory, args.report)
    except ValueError as exc:
        print("Hugging Face verification failed:", str(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        # Never print request headers, environment variables or token values.
        print("Hugging Face download or verification failed:", type(exc).__name__, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
