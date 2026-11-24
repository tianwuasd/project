import gzip
import os

import nibabel as nib


def local_image(image, source, name, workspace):
    if source.suffix.lower() != '.gz' or workspace.config['technical']['roi_xyz'] is not None:
        return image
    target = workspace.path/(name+'.nii')
    with gzip.open(source, 'rb') as incoming, target.open('xb') as outgoing:
        workspace.streams.append(outgoing)
        try:
            count = 0
            while True:
                with workspace.allocation(8*1024**2, 'decompress'):
                    block = incoming.read(4*1024**2)
                    if not block:
                        break
                    outgoing.write(block)
                    count += 1
                    workspace.report('decompress', count)
                    del block
            outgoing.flush()
            os.fsync(outgoing.fileno())
        finally:
            workspace.streams.remove(outgoing)
    return nib.load(target, mmap='r', keep_file_open=False)
