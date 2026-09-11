"""Rebuild a validated submission with portable ZIP metadata and identical payloads.

This checks the submitted archive; it does not certify a remote Docker builder.
No Dockerfile is added: the participant contract supplies the runtime image.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import stat
import sys
import zipfile
from pathlib import Path, PurePosixPath

from tools.submission_audit import expanded_bytes


def rebuild(source: Path, destination: Path) -> dict[str, object]:
    if destination.exists():
        raise ValueError("use a new output path to preserve the previous artifact")
    payloads: dict[str, bytes] = {}
    imports: set[str] = set()
    with zipfile.ZipFile(source) as archive:
        if archive.testzip() is not None:
            raise ValueError("archive checksum failure")
        case_names: set[str] = set()
        for entry in archive.infolist():
            name = entry.filename
            path = PurePosixPath(name)
            if (not name or path.is_absolute() or path.as_posix() != name
                    or any(part in (".", "..", "") for part in name.split("/"))
                    or "\\" in name or ":" in name
                    or any(ord(char) < 32 or ord(char) > 126 for char in name)
                    or any(part.endswith((".", " ")) for part in path.parts)):
                raise ValueError(f"nonportable archive path: {name!r}")
            if name.casefold() in case_names or stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError(f"duplicate path or symbolic link: {name}")
            case_names.add(name.casefold())
            if entry.flag_bits & 1:
                raise ValueError(f"encrypted member: {name}")
            if len(path.parts) == 1:
                if path.suffix != ".py":
                    raise ValueError(f"unexpected root file: {name}")
            elif (len(path.parts) != 2 or path.parts[0] not in ("book", "weights", "syzygy")
                  or path.suffix not in (".npz", ".rtbw", ".rtbz", ".bin", ".gz")):
                raise ValueError(f"unexpected asset: {name}")
            data = archive.read(entry)
            if path.suffix == ".py":
                tree = ast.parse(data.decode("utf-8-sig"), filename=name)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imports.update(item.name.split(".")[0] for item in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        imports.add(node.module.split(".")[0])
            payloads[name] = data
    if "agent.py" not in payloads:
        raise ValueError("agent.py must be at the archive root")
    modules = {Path(name).stem for name in payloads if "/" not in name}
    preinstalled = {"torch", "numpy", "chess", "onnxruntime", "numba"}
    if modules & (sys.stdlib_module_names | preinstalled):
        raise ValueError("submission shadows a standard or preinstalled module")
    unsupported = imports - modules - sys.stdlib_module_names - preinstalled
    if unsupported:
        raise ValueError(f"unsupported imports: {sorted(unsupported)}")
    unzipped = sum(map(len, payloads.values()))
    recursive = sum(expanded_bytes(data) for data in payloads.values())
    if max(unzipped, recursive) > 50_000_000:
        raise ValueError("submission exceeds the 50 MB expanded budget")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "x", zipfile.ZIP_DEFLATED, allowZip64=False) as archive:
        for name, data in sorted(payloads.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 11, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data, compresslevel=6)
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != set(payloads):
            raise ValueError("rebuilt archive failed validation")
        for name, data in payloads.items():
            if archive.read(name) != data:
                raise ValueError(f"payload changed while repacking: {name}")
    return {
        "source_zip_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "zip_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "zip_bytes": destination.stat().st_size, "unzipped_bytes": unzipped,
        "fully_recursive_bytes": recursive, "margin_bytes": 50_000_000 - recursive,
        "file_count": len(payloads), "all_payloads_byte_identical": True,
        "metadata": "ZIP 2.0 DEFLATE; ASCII POSIX paths; regular files 0644; no extras/comments",
        "unsupported_imports": sorted(unsupported),
        "remote_docker_build_validated": False,
        "files": {name: hashlib.sha256(data).hexdigest()
                  for name, data in sorted(payloads.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    report = rebuild(args.source_zip, args.out)
    args.manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps({key: value for key, value in report.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
