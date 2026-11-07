from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PlyGaussian:
    """Binary 3DGS PLY with float32 properties (EGS / standard layout)."""

    header: bytes
    vertex_count: int
    property_names: list[str]
    row_dtype: np.dtype
    raw_rows: np.ndarray  # shape (N,) structured

    @property
    def n(self) -> int:
        return int(self.vertex_count)

    def xyz(self) -> np.ndarray:
        d = self.raw_rows
        return np.stack([d["x"], d["y"], d["z"]], axis=1).astype(np.float64)

    def opacity_sigmoid(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.raw_rows["opacity"].astype(np.float64)))

    def scales_exp(self) -> np.ndarray:
        s = self.raw_rows
        return np.exp(
            np.stack([s["scale_0"], s["scale_1"], s["scale_2"]], axis=1).astype(np.float64)
        )


def read_binary_gaussian_ply(path: Path | str) -> PlyGaussian:
    path = Path(path)
    with open(path, "rb") as f:
        header_lines: list[bytes] = []
        while True:
            line = f.readline()
            header_lines.append(line)
            if line.strip() == b"end_header":
                break
        header = b"".join(header_lines)
        fmt = "ascii"
        vertex_count = 0
        props: list[tuple[str, str]] = []
        for line in header_lines:
            if line.startswith(b"format "):
                # e.g. "format binary_little_endian 1.0"
                fmt = line.decode("ascii").strip().split(" ", 1)[1].strip()
            elif line.startswith(b"element vertex "):
                vertex_count = int(line.split()[-1])
            elif line.startswith(b"property "):
                parts = line.split()
                props.append((parts[-1].decode("ascii"), parts[1].decode("ascii")))
        if fmt != "binary_little_endian 1.0":
            raise ValueError(f"Unsupported PLY format: {fmt}")
        dtype = np.dtype([(name, "<f4") for name, _t in props])
        raw = np.fromfile(f, dtype=dtype, count=vertex_count)
        if raw.shape[0] != vertex_count:
            raise EOFError("Truncated PLY body")
    names = [p[0] for p in props]
    return PlyGaussian(header=header, vertex_count=vertex_count, property_names=names, row_dtype=dtype, raw_rows=raw)


def write_binary_gaussian_ply(path: Path | str, ply: PlyGaussian, indices: np.ndarray | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = ply.raw_rows if indices is None else ply.raw_rows[indices.astype(np.int64)]
    new_count = rows.shape[0]
    header_str = ply.header.decode("utf-8")
    lines = header_str.splitlines()
    out_lines: list[str] = []
    for line in lines:
        if line.startswith("element vertex "):
            out_lines.append(f"element vertex {new_count}")
        else:
            out_lines.append(line)
    new_header = ("\n".join(out_lines) + "\n").encode("utf-8")
    with open(path, "wb") as f:
        f.write(new_header)
        rows.tofile(f)


def byte_identical_roundtrip(src: Path, tmp: Path) -> bool:
    ply = read_binary_gaussian_ply(src)
    write_binary_gaussian_ply(tmp, ply)
    return Path(src).read_bytes() == Path(tmp).read_bytes()
