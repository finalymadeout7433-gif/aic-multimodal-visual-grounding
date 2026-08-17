from __future__ import annotations

import argparse
import base64
import os
import pathlib
import sys

import paramiko


def connect() -> paramiko.SSHClient:
    required = ["AIC_SSH_HOST", "AIC_SSH_PORT", "AIC_SSH_USER", "AIC_SSH_PASSWORD"]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"missing SSH environment variables: {', '.join(missing)}")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=os.environ["AIC_SSH_HOST"],
        port=int(os.environ["AIC_SSH_PORT"]),
        username=os.environ["AIC_SSH_USER"],
        password=os.environ["AIC_SSH_PASSWORD"],
        timeout=20,
        banner_timeout=20,
        auth_timeout=20,
    )
    return client


def run(client: paramiko.SSHClient, command: str) -> int:
    _, stdout, stderr = client.exec_command(command, get_pty=False)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    if out:
        sys.stdout.buffer.write(out.encode("utf-8", errors="replace"))
    if err:
        sys.stderr.buffer.write(err.encode("utf-8", errors="replace"))
    return code


def upload(client: paramiko.SSHClient, local: pathlib.Path, remote: str) -> None:
    with client.open_sftp() as sftp:
        sftp.put(str(local), remote, confirm=True)
        stat = sftp.stat(remote)
    if stat.st_size != local.stat().st_size:
        raise RuntimeError(
            f"upload size mismatch: local={local.stat().st_size}, remote={stat.st_size}"
        )
    print(f"UPLOAD_OK bytes={stat.st_size} remote={remote}")


def download(client: paramiko.SSHClient, remote: str, local: pathlib.Path) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    temporary = local.with_name(local.name + ".partial")
    with client.open_sftp() as sftp:
        remote_stat = sftp.stat(remote)
        sftp.get(remote, str(temporary))
    if temporary.stat().st_size != remote_stat.st_size:
        raise RuntimeError(
            f"download size mismatch: remote={remote_stat.st_size}, "
            f"local={temporary.stat().st_size}"
        )
    temporary.replace(local)
    print(f"DOWNLOAD_OK bytes={local.stat().st_size} local={local}")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("command")
    run_parser.add_argument("--base64", action="store_true")
    upload_parser = sub.add_parser("upload")
    upload_parser.add_argument("local", type=pathlib.Path)
    upload_parser.add_argument("remote")
    download_parser = sub.add_parser("download")
    download_parser.add_argument("remote")
    download_parser.add_argument("local", type=pathlib.Path)
    args = parser.parse_args()

    client = connect()
    try:
        if args.action == "run":
            command = args.command
            if args.base64:
                command = base64.b64decode(command).decode("utf-8")
            return run(client, command)
        if args.action == "upload":
            upload(client, args.local.resolve(), args.remote)
            return 0
        if args.action == "download":
            download(client, args.remote, args.local.resolve())
            return 0
        raise AssertionError(args.action)
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
