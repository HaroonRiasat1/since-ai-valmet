from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from splat_objects.config import repo_root
from splat_objects.ply_io import byte_identical_roundtrip


class TestPlyRoundtrip(unittest.TestCase):
    def test_roundtrip(self) -> None:
        src = (repo_root().parent.parent / "Equipment Room.ply").resolve()
        self.assertTrue(src.is_file(), msg=str(src))
        with tempfile.TemporaryDirectory() as td:
            dst = Path(td) / "out.ply"
            self.assertTrue(byte_identical_roundtrip(src, dst))


if __name__ == "__main__":
    unittest.main()
