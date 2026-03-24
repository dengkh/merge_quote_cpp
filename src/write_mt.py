"""
Drop-in replacement for pq.ParquetWriter with multi-threaded compression.

Uses a C++ bridge (libwrite_mt.so) via the Arrow C Data Interface for
zero-copy data passing from Python to the C++ parquet writer.

Usage (identical to pq.ParquetWriter):

    from write_mt import ParquetWriterMT

    with ParquetWriterMT(path, schema, compression="zstd") as writer:
        writer.write_table(table)
"""

import ctypes
import os
from pathlib import Path

import pyarrow as pa

# ── Load shared library ──────────────────────────────────────────────

_LIB_SEARCH = [
    Path(__file__).resolve().parent.parent / "build" / "libwrite_mt.so",
    Path(__file__).resolve().parent.parent / "libwrite_mt.so",
    Path(__file__).resolve().parent / "libwrite_mt.so",
]
_lib = None
for _p in _LIB_SEARCH:
    if _p.exists():
        _lib = ctypes.CDLL(str(_p))
        break
if _lib is None:
    raise ImportError(
        f"libwrite_mt.so not found in: {[str(p) for p in _LIB_SEARCH]}"
    )

_lib.writer_open.argtypes = [ctypes.c_uint64, ctypes.c_char_p,
                              ctypes.c_char_p, ctypes.c_int]
_lib.writer_open.restype = ctypes.c_void_p

_lib.writer_write_batch.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
_lib.writer_write_batch.restype = ctypes.c_int

_lib.writer_close.argtypes = [ctypes.c_void_p]
_lib.writer_close.restype = ctypes.c_int

# Arrow C Data Interface struct sizes (64-bit)
_SCHEMA_SIZE = 72   # struct ArrowSchema: 9 pointers
_ARRAY_SIZE = 80    # struct ArrowArray:  10 pointers


class ParquetWriterMT:
    """Parquet writer with multi-threaded column compression.

    API-compatible with ``pyarrow.parquet.ParquetWriter`` for the subset
    used by parquet_sort_modified.py.
    """

    def __init__(
        self,
        where: str,
        schema: pa.Schema,
        compression: str = "zstd",
        compression_level: int = -1,
    ):
        self._schema = schema
        self._ptr = None

        schema_buf = (ctypes.c_byte * _SCHEMA_SIZE)()
        schema._export_to_c(ctypes.addressof(schema_buf))

        self._ptr = _lib.writer_open(
            ctypes.addressof(schema_buf),
            str(where).encode("utf-8"),
            compression.encode("utf-8"),
            compression_level,
        )
        if not self._ptr:
            raise RuntimeError("Failed to open multi-threaded parquet writer")

    def write_table(self, table: pa.Table) -> None:
        """Write an Arrow Table (may be chunked) as one or more row groups."""
        if self._ptr is None:
            raise RuntimeError("Writer is closed")
        for batch in table.to_batches():
            array_buf = (ctypes.c_byte * _ARRAY_SIZE)()
            batch._export_to_c(ctypes.addressof(array_buf))
            ret = _lib.writer_write_batch(self._ptr, ctypes.addressof(array_buf))
            if ret != 0:
                raise RuntimeError("Failed to write batch")

    def close(self) -> None:
        if self._ptr:
            ret = _lib.writer_close(self._ptr)
            self._ptr = None
            if ret != 0:
                raise RuntimeError("Failed to close writer")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        if self._ptr:
            _lib.writer_close(self._ptr)
            self._ptr = None
