import mmap
import os
from pathlib import Path


def discard_file_pages(stream, offset=0, length=0):
    if hasattr(os, "posix_fadvise"):
        os.posix_fadvise(stream.fileno(), offset, length, os.POSIX_FADV_DONTNEED)


def release_private_mapping(array):
    array.flush()
    if hasattr(array._mmap, "madvise"):
        array._mmap.madvise(mmap.MADV_DONTNEED)
    if hasattr(os, "posix_fadvise"):
        with Path(array.filename).open("rb") as stream:
            discard_file_pages(stream)
