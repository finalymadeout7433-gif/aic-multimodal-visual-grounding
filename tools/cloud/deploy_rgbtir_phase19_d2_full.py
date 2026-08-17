from __future__ import annotations

import argparse
import base64
import hashlib
import os
import pathlib
import shlex
import sys
import tempfile

import paramiko


def _password_from_stdin() -> str:
    password = sys.stdin.readline().rstrip("\r\n")
    if not password:
        raise RuntimeError("SSH password was not provided on stdin")
    return password


def _connect(args: argparse.Namespace) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=args.host,
        port=args.port,
        username=args.user,
        password=_password_from_stdin(),
        timeout=30,
        banner_timeout=30,
        auth_timeout=30,
    )
    return client


def _run(client: paramiko.SSHClient, command: str) -> tuple[int, str, str]:
    encoded = base64.b64encode(command.encode("utf-8")).decode("ascii")
    wrapper = f"echo {shlex.quote(encoded)} | base64 -d | bash"
    _, stdout, stderr = client.exec_command(wrapper, get_pty=False)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    return code, out, err


def _print_result(code: int, out: str, err: str) -> None:
    if out:
        sys.stdout.write(out)
    if err:
        sys.stderr.write(err)
    if code != 0:
        raise RuntimeError(f"remote command failed with exit code {code}")


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _promote_upload(sftp: object, temporary: str, destination: str) -> None:
    """Atomically replace a stale destination with the verified upload."""
    posix_rename = getattr(sftp, "posix_rename", None)
    if not callable(posix_rename):
        raise RuntimeError("SFTP server/client does not support atomic replacement")
    posix_rename(temporary, destination)


def _upload_prepare(
    client: paramiko.SSHClient,
    *,
    local_bundle: pathlib.Path,
    remote_bundle: str,
    expected_sha256: str,
) -> None:
    local_bundle = local_bundle.resolve()
    if not local_bundle.is_file():
        raise FileNotFoundError(local_bundle)
    actual = _sha256(local_bundle)
    if actual != expected_sha256.upper():
        raise RuntimeError(f"local bundle SHA-256 mismatch: {actual}")
    total = local_bundle.stat().st_size
    next_report = 10

    def callback(transferred: int, remote_total: int) -> None:
        nonlocal next_report
        percent = int(transferred * 100 / max(1, remote_total))
        if percent >= next_report:
            print(f"SFTP_PROGRESS percent={percent} bytes={transferred}/{remote_total}")
            next_report += 10

    temporary = remote_bundle + ".uploading"
    with client.open_sftp() as sftp:
        try:
            existing_size = sftp.stat(temporary).st_size
        except OSError:
            existing_size = -1
        if existing_size == total:
            print(f"SFTP_REUSE_COMPLETE_TEMP bytes={existing_size}")
        else:
            sftp.put(str(local_bundle), temporary, callback=callback, confirm=True)
        remote_size = sftp.stat(temporary).st_size
        if remote_size != total:
            raise RuntimeError(
                f"remote upload size mismatch: {remote_size} != {total}"
            )
        _promote_upload(sftp, temporary, remote_bundle)

    quoted_bundle = shlex.quote(remote_bundle)
    expected = shlex.quote(expected_sha256.upper())
    command = f"""
set -euo pipefail
actual="$(sha256sum {quoted_bundle} | awk '{{print toupper($1)}}')"
test "$actual" = {expected}
python3 - {quoted_bundle} <<'PY'
import sys, zipfile
path = sys.argv[1]
with zipfile.ZipFile(path) as handle:
    bad = handle.testzip()
    if bad is not None:
        raise RuntimeError(f"bad ZIP entry: {{bad}}")
print("REMOTE_ZIP_TEST_GO")
PY
target=/home/featurize/aic_rgbtir_phase19_d2_full_bundle
if [[ -e "$target" ]]; then
  backup="${{target}}.predeploy.$(date -u +%Y%m%dT%H%M%SZ)"
  mv "$target" "$backup"
  echo "PREVIOUS_BUNDLE_MOVED=$backup"
fi
unzip -q {quoted_bundle} -d /home/featurize
test -f "$target/BUNDLE_MANIFEST.json"
test -f "$target/tools/cloud/run_rgbtir_phase19_d2_full_gated.sh"
echo "REMOTE_BUNDLE_READY=$target"
echo "REMOTE_BUNDLE_SHA256=$actual"
"""
    _print_result(*_run(client, command))


def _download_file(
    client: paramiko.SSHClient,
    *,
    remote_path: str,
    local_path: pathlib.Path,
) -> None:
    """Download one remote file atomically and report its verified local hash."""
    local_path = local_path.resolve()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with client.open_sftp() as sftp:
        remote_size = sftp.stat(remote_path).st_size
        next_report = 10

        def callback(transferred: int, total: int) -> None:
            nonlocal next_report
            percent = int(transferred * 100 / max(1, total))
            if percent >= next_report:
                print(f"SFTP_PROGRESS percent={percent} bytes={transferred}/{total}")
                next_report += 10

        fd, temporary_name = tempfile.mkstemp(
            prefix=local_path.name + ".",
            suffix=".downloading",
            dir=local_path.parent,
        )
        os.close(fd)
        pathlib.Path(temporary_name).unlink()
        try:
            sftp.get(remote_path, temporary_name, callback=callback)
            temporary = pathlib.Path(temporary_name)
            if temporary.stat().st_size != remote_size:
                raise RuntimeError(
                    f"download size mismatch: {temporary.stat().st_size} != {remote_size}"
                )
            temporary.replace(local_path)
        finally:
            pathlib.Path(temporary_name).unlink(missing_ok=True)
    print(f"LOCAL_FILE={local_path}")
    print(f"LOCAL_SIZE={local_path.stat().st_size}")
    print(f"LOCAL_SHA256={_sha256(local_path)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--user", required=True)
    subparsers = parser.add_subparsers(dest="action", required=True)
    upload = subparsers.add_parser("upload-prepare")
    upload.add_argument("--local-bundle", type=pathlib.Path, required=True)
    upload.add_argument("--remote-bundle", required=True)
    upload.add_argument("--expected-sha256", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--command-base64", required=True)
    download = subparsers.add_parser("download-file")
    download.add_argument("--remote-path", required=True)
    download.add_argument("--local-path", type=pathlib.Path, required=True)
    args = parser.parse_args()

    client = _connect(args)
    try:
        if args.action == "upload-prepare":
            _upload_prepare(
                client,
                local_bundle=args.local_bundle,
                remote_bundle=args.remote_bundle,
                expected_sha256=args.expected_sha256,
            )
            return 0
        if args.action == "run":
            command = base64.b64decode(args.command_base64).decode("utf-8")
            _print_result(*_run(client, command))
            return 0
        if args.action == "download-file":
            _download_file(
                client,
                remote_path=args.remote_path,
                local_path=args.local_path,
            )
            return 0
        raise AssertionError(args.action)
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
