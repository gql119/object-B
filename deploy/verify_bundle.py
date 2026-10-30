#!/usr/bin/env python3
"""Verify the offline deployment package without importing project code."""
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "package_manifest.json"
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".jpg", ".jpeg", ".png", ".bmp", ".tar"}
REQUIRED = {
    "AT_SOURCE_87456.zip",
    "deploy/preflight.sh",
    "deploy/run_at4_clean.sh",
    "deploy/run_at4_poison.sh",
    "deploy/run_at8_clean.sh",
    "deploy/run_at8_poison.sh",
    "native_sources/ULTRALYTICS_LICENSE",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("schema") != "s4.at.cloud-package-manifest.v1":
        raise SystemExit("unexpected package manifest schema")
    if manifest.get("formal_ready") is not False:
        raise SystemExit("package must remain explicitly not formal-ready")
    if manifest.get("source_commit") != "87456fa7a57206551b0e3be0253cc4866f0cdcf1":
        raise SystemExit("unexpected scientific source commit")
    if any(manifest.get(key) is not False for key in ("contains_data", "contains_weights", "contains_credentials")):
        raise SystemExit("package metadata must declare no data, weights, or credentials")

    expected = manifest.get("files")
    if not isinstance(expected, dict):
        raise SystemExit("manifest files must be an object")
    actual = {}
    for path in ROOT.rglob("*"):
        if path.is_symlink():
            raise SystemExit(f"symlink is not allowed in package: {path.relative_to(ROOT)}")
        if path.is_file() and path != MANIFEST:
            rel = path.relative_to(ROOT).as_posix()
            if path.suffix.lower() in FORBIDDEN_SUFFIXES:
                raise SystemExit(f"data/weight/archive payload is not allowed: {rel}")
            actual[rel] = path
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        unlisted = sorted(set(actual) - set(expected))
        raise SystemExit(f"manifest membership mismatch; missing={missing[:5]} unlisted={unlisted[:5]}")
    for rel, path in sorted(actual.items()):
        record = expected[rel]
        if path.stat().st_size != record.get("bytes") or sha256(path) != record.get("sha256"):
            raise SystemExit(f"size/SHA256 mismatch: {rel}")

    snapshot = ROOT / "source_snapshot"
    for rel, wanted in manifest.get("source_closure", {}).items():
        path = snapshot / PurePosixPath(rel)
        if not path.is_file() or sha256(path) != wanted:
            raise SystemExit(f"scientific source-closure mismatch: {rel}")
    for rel, record in manifest.get("native_sources", {}).items():
        path = ROOT / "native_sources/ultralytics" / PurePosixPath(rel)
        if not path.is_file() or path.stat().st_size != record.get("bytes") or sha256(path) != record.get("sha256"):
            raise SystemExit(f"pinned native-source mismatch: {rel}")

    if not REQUIRED.issubset(actual):
        raise SystemExit(f"required package files missing: {sorted(REQUIRED - set(actual))}")
    archive = ROOT / "AT_SOURCE_87456.zip"
    with zipfile.ZipFile(archive) as zf:
        archive_entries = zf.infolist()
        for member in archive_entries:
            member_path = PurePosixPath(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise SystemExit(f"unsafe source archive path: {member.filename}")
            if not member.is_dir() and member_path.suffix.lower() in FORBIDDEN_SUFFIXES:
                raise SystemExit(f"data/weight payload in source archive: {member.filename}")
            if not member.is_dir():
                snapshot_file = snapshot / member_path
                if not snapshot_file.is_file() or snapshot_file.read_bytes() != zf.read(member):
                    raise SystemExit(f"source zip/snapshot content mismatch: {member.filename}")
    print(json.dumps({"state": "PACKAGE_HASHES_PASS", "files_checked": len(actual),
                      "archive_entries_checked": len(archive_entries),
                      "formal_ready": False}, indent=2))


if __name__ == "__main__":
    main()
