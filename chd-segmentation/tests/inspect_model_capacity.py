import json
import argparse
from pathlib import Path
from zipfile import ZipFile


def storage_bytes(path):
    with ZipFile(path) as archive:
        sizes = [item.file_size for item in archive.infolist() if '/data/' in item.filename]
    return dict(path=str(path), file_bytes=path.stat().st_size,
                tensor_storage_bytes=sum(sizes), tensor_storages=len(sizes))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stu', type=Path, required=True)
    parser.add_argument('--med', type=Path, required=True)
    arguments = parser.parse_args()
    paths = [arguments.stu, arguments.med]
    report = [storage_bytes(path) for path in paths]
    import torch
    from chd_segmentation.models.q_route1.localizer import Localizer
    from chd_segmentation.models.q_route1.network import LocalNetwork
    with torch.device('meta'):
        for name, model in (('q_localizer', Localizer()), ('q_local', LocalNetwork(method='M07'))):
            report.append(dict(model=name, tensor_storage_bytes=sum(
                value.numel()*value.element_size() for value in model.state_dict().values())))
    print(json.dumps(report, ensure_ascii=False))
