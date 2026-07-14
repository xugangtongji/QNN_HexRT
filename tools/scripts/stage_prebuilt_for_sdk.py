#!/usr/bin/env python3
"""Build an immutable, receipt-addressed QHexRT payload for RunAnywhere."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[2]
ABI = "arm64-v8a"


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: pathlib.Path) -> dict[str, object]:
    return {"sha256": sha256(path), "size_bytes": path.stat().st_size}


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def source_git_identity() -> tuple[str, bool]:
    sentinel = "0" * 40
    head = subprocess.run(
        ["git", "-C", os.fspath(ROOT), "rev-parse", "--verify", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    git_sha = head.stdout.strip()
    if head.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}", git_sha):
        return sentinel, True

    status = subprocess.run(
        ["git", "-C", os.fspath(ROOT), "status", "--porcelain", "--untracked-files=all"],
        capture_output=True,
        text=True,
        check=False,
    )
    if status.returncode != 0:
        return git_sha, True
    return git_sha, bool(status.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdk-root", required=True, type=pathlib.Path)
    parser.add_argument("--build-dir", default=ROOT / "build", type=pathlib.Path)
    parser.add_argument("--qnn-sdk-root", default=os.environ.get("QNN_SDK_ROOT"), type=pathlib.Path)
    parser.add_argument("--ndk-root", default=os.environ.get("ANDROID_NDK_HOME"), type=pathlib.Path)
    args = parser.parse_args()
    if not args.qnn_sdk_root or not args.ndk_root:
        raise SystemExit("QNN_SDK_ROOT and ANDROID_NDK_HOME are required")

    files = {
        "include/qhexrt/qhexrt_c.h": ROOT / "include/qhexrt/qhexrt_c.h",
        f"lib/{ABI}/libqhexrt_core.a": args.build_dir / "libqhexrt_core.a",
        f"lib/{ABI}/libqhexrt_host.a": args.build_dir / "libqhexrt_host.a",
    }
    for path in files.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"missing build artifact: {path}")

    toolchain = args.ndk_root / "toolchains/llvm/prebuilt/linux-x86_64/bin"
    compiler = toolchain / "clang++"
    llvm_ar = toolchain / "llvm-ar"
    readelf = toolchain / "llvm-readelf"
    ndk_metadata = args.ndk_root / "source.properties"
    qnn_metadata = args.qnn_sdk_root / "sdk.yaml"
    cache = args.build_dir / "CMakeCache.txt"
    system_state = next((args.build_dir / "CMakeFiles").glob("*/CMakeSystem.cmake"))
    compiler_state = next((args.build_dir / "CMakeFiles").glob("*/CMakeCXXCompiler.cmake"))
    version_output = command(str(compiler), "--version")
    source_hash = hashlib.sha256()
    for path in sorted((ROOT / "src").glob("*")) + sorted((ROOT / "include").rglob("*")):
        if path.is_file():
            source_hash.update(path.relative_to(ROOT).as_posix().encode())
            source_hash.update(bytes.fromhex(sha256(path)))

    artifacts = {relative: identity(path) for relative, path in files.items()}
    build = {
        "android_abi": ABI,
        "archive_evidence": {
            "core": {"member_count": len(command(str(llvm_ar), "t", str(files[f'lib/{ABI}/libqhexrt_core.a'])).splitlines())},
            "host": {"member_count": len(command(str(llvm_ar), "t", str(files[f'lib/{ABI}/libqhexrt_host.a'])).splitlines())},
            "llvm_ar_sha256": sha256(llvm_ar),
            "llvm_readelf_sha256": sha256(readelf),
        },
        "build_type": "Release",
        "cmake_cache_sha256": sha256(cache),
        "cmake_system": {
            "crosscompiling": True,
            "name": "Android",
            "processor": "aarch64",
            "state_file_sha256": sha256(system_state),
        },
        "compiler": {
            "id": "Clang",
            "sha256": sha256(compiler),
            "state_file_sha256": sha256(compiler_state),
            "version": version_output.splitlines()[0],
            "version_output_sha256": hashlib.sha256(version_output.encode()).hexdigest(),
        },
        "ndk": {"metadata_file": "source.properties", "metadata_sha256": sha256(ndk_metadata)},
        "qnn_sdk": {"metadata_file": "sdk.yaml", "metadata_sha256": sha256(qnn_metadata)},
    }
    git_sha, git_dirty = source_git_identity()
    source = {
        "git_dirty": git_dirty,
        "git_sha": git_sha,
        "state_sha256": source_hash.hexdigest(),
    }
    receipt = {
        "artifacts": artifacts,
        "build": build,
        "c_abi": {"major": 1, "minor": 1},
        "qhexrt_version": "0.1.0",
        "schema": "qhexrt-build-receipt/v1",
        "source": source,
    }

    prebuilt = args.sdk_root / "engines/qhexrt/prebuilt"
    versions = prebuilt / "versions"
    versions.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=prebuilt, prefix=".stage-") as temp_name:
        temp = pathlib.Path(temp_name)
        for relative, source_path in files.items():
            destination = temp / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination)
        receipt_path = temp / "qhexrt-build-receipt.json"
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        receipt_hash = sha256(receipt_path)
        manifest = {
            "android_abi": ABI,
            "build": build,
            "build_receipt_sha256": receipt_hash,
            "c_abi": receipt["c_abi"],
            "files": artifacts,
            "qhexrt_version": receipt["qhexrt_version"],
            "schema": "qhexrt-prebuilt/v2",
            "source": source,
        }
        (temp / "qhexrt-prebuilt.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        selected = versions / receipt_hash
        if not selected.exists():
            os.rename(temp, selected)

    current_tmp = prebuilt / ".current"
    current_tmp.unlink(missing_ok=True)
    current_tmp.symlink_to(f"versions/{receipt_hash}")
    os.replace(current_tmp, prebuilt / "current")
    print(selected)


if __name__ == "__main__":
    main()
