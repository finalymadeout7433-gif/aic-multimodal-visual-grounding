#!/usr/bin/env bash
set -euo pipefail

ARCHIVE_ROOT="${1:?usage: extract_rgbt_groundbench.sh ARCHIVE_ROOT [DEST_ROOT]}"
DEST_ROOT="${2:-/home/featurize/data/RGBT_GroundBench/extracted}"

declare -A EXPECTED=(
  [ann_flir.tar]="7CAEBB8F3E947E6D5440EF0FA5C5CF13FCB492B896EF9D3DF2FC595B9A68E24A"
  [ann_m3fd.tar]="C69BFB83B5BBAA68312044BBCF6E0B5A3B43E468F795E767D5A134EC03A53FD3"
  [ann_mfad.tar]="BEEE1CADF991E9823CAF5043BCE8AA33016054E2E6334952B846E8B7CB5654EA"
  [data_flir.tar]="0B7B81A59ED796BFB2D1BB4258F45EA4B90BE816C9D33C4C1A1F194CFADB470E"
  [data_m3fd.tar]="BDBA0DB59A3FB1BA02E67B5B6CEE19320BE50B11D787115E15F7ED42E1339F7C"
  [data_mfad.tar]="B84155830051A66B601253FDCD734894F1753DF4836C31D2F10AAE54111F73E6"
)

for name in ann_flir.tar ann_m3fd.tar ann_mfad.tar data_flir.tar data_m3fd.tar data_mfad.tar; do
  path="${ARCHIVE_ROOT}/${name}"
  [[ -f "${path}" ]] || { echo "missing ${path}" >&2; exit 2; }
  actual="$(sha256sum "${path}" | awk '{print toupper($1)}')"
  [[ "${actual}" == "${EXPECTED[$name]}" ]] || {
    echo "SHA256 mismatch for ${name}: ${actual}" >&2
    exit 3
  }
done

mkdir -p "${DEST_ROOT}"
for name in ann_flir.tar ann_m3fd.tar ann_mfad.tar data_flir.tar data_m3fd.tar data_mfad.tar; do
  tar -xf "${ARCHIVE_ROOT}/${name}" -C "${DEST_ROOT}"
done

for required in image_data rgbtvg_flir rgbtvg_m3fd rgbtvg_mfad; do
  [[ -d "${DEST_ROOT}/${required}" ]] || {
    echo "missing extracted directory ${DEST_ROOT}/${required}" >&2
    exit 4
  }
done

echo "RGBT_GROUNDBENCH_READY=${DEST_ROOT}"
