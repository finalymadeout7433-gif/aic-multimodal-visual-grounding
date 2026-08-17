from __future__ import annotations

import pathlib
from types import SimpleNamespace

from tools.cloud.deploy_rgbtir_phase19_d2_full import _download_file, _promote_upload


class _ExistingDestinationSFTP:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def posix_rename(self, source: str, destination: str) -> None:
        self.calls.append((source, destination))


def test_promote_upload_atomically_replaces_existing_destination() -> None:
    sftp = _ExistingDestinationSFTP()

    _promote_upload(sftp, "/remote/bundle.uploading", "/remote/bundle.zip")

    assert sftp.calls == [
        ("/remote/bundle.uploading", "/remote/bundle.zip"),
    ]


class _DownloadSFTP:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> "_DownloadSFTP":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def stat(self, _path: str) -> SimpleNamespace:
        return SimpleNamespace(st_size=len(self.payload))

    def get(self, _remote: str, local: str, callback: object) -> None:
        pathlib.Path(local).write_bytes(self.payload)
        callback(len(self.payload), len(self.payload))  # type: ignore[operator]


class _DownloadClient:
    def __init__(self, payload: bytes) -> None:
        self.sftp = _DownloadSFTP(payload)

    def open_sftp(self) -> _DownloadSFTP:
        return self.sftp


def test_download_file_is_atomic_and_reports_hash(
    tmp_path: pathlib.Path,
    capsys: object,
) -> None:
    target = tmp_path / "nested" / "artifact.zip"

    _download_file(
        _DownloadClient(b"verified-payload"),  # type: ignore[arg-type]
        remote_path="/remote/artifact.zip",
        local_path=target,
    )

    assert target.read_bytes() == b"verified-payload"
    assert not list(target.parent.glob("*.downloading"))
    output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "LOCAL_SIZE=16" in output
    assert "LOCAL_SHA256=" in output
