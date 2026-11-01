"""Filename bridge for nnU-Net 2.8.1's broken dictionary-loading branch.

The pinned upstream accepts Union[dict,str] in its signature but initializes its
local checkpoint only for str. Always pass a filename to that implementation.
"""
import os
from pathlib import Path
import tempfile


def load_parent_checkpoint(parent_loader, value, torch):
    if isinstance(value, (str, os.PathLike)):
        filename = os.fsdecode(os.fspath(value))
        parent_loader(filename)
        return torch.load(filename, map_location='cpu', weights_only=False)
    if not isinstance(value, dict):
        raise TypeError('检查点必须是文件路径或检查点字典')
    # Retain the advertised mapping API without invoking the defective upstream
    # branch. Temporary bytes are removed on success and on parent exceptions.
    with tempfile.TemporaryDirectory(prefix='vista_checkpoint_load_') as temp:
        filename = Path(temp) / 'checkpoint.pth'
        torch.save(value, filename)
        parent_loader(str(filename))
    return value
