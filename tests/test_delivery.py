import json
import os
from pathlib import Path
import shutil
import unittest
from zipfile import ZipFile, ZIP_DEFLATED

from chd_segmentation.delivery.archive import build_package, read_package
from chd_segmentation.delivery.configs import model_configs


PROJECT = Path(__file__).resolve().parents[1]


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.parent = Path(os.environ['CHD_DELIVERY_TEST_ROOT']).resolve(strict=True)
        self.work = self.parent / self._testMethodName
        self.work.mkdir()

    def tearDown(self):
        if not self.work.is_relative_to(self.parent) or self.work.is_symlink():
            raise ValueError('测试目录归属异常')
        shutil.rmtree(self.work)

    def entries(self):
        return [dict(source=PROJECT/'README.md', target='inputs/readme.md'),
                dict(source=PROJECT/'README.md', target='inputs/q1282/readme.md')]

    def test_package_reuses_identical_content_and_preserves_repeat_mtime(self):
        path = self.work/'private.zip'
        first = build_package(self.entries(), path, 'private')
        stamp = path.stat().st_mtime_ns
        second = build_package(self.entries(), path, 'private')
        self.assertEqual(first['identity'], second['identity'])
        self.assertEqual(stamp, path.stat().st_mtime_ns)
        with read_package(path) as package:
            self.assertEqual(len(package.manifest['files']), 2)
            self.assertEqual(len(package.archive.namelist()), 2)
            for row in package.manifest['files']:
                self.assertEqual(package.archive.read(row['object']), (PROJECT/'README.md').read_bytes())

    def test_package_rejects_traversal_and_duplicate_targets_before_writing(self):
        path = self.work/'invalid.zip'
        for entries in ([dict(source=PROJECT/'README.md', target='../escape')],
                        [self.entries()[0], self.entries()[0]]):
            with self.assertRaises(ValueError):
                build_package(entries, path, 'private')
            self.assertFalse(path.exists())

    def test_package_detects_changed_payload_before_installation(self):
        good, damaged = self.work/'good.zip', self.work/'damaged.zip'
        build_package(self.entries(), good, 'private')
        with ZipFile(good) as source, ZipFile(damaged, 'w', compression=ZIP_DEFLATED) as target:
            for name in source.namelist():
                content = source.read(name)
                target.writestr(name, content+b'changed' if name.startswith('objects/') else content)
        with self.assertRaises(ValueError):
            with read_package(damaged):
                self.fail('损坏数据被接受')

    def test_generated_configs_keep_scientific_budgets_and_shared_resource_root(self):
        assets = dict(stunet=PROJECT/'data/stu.model', medsam2=PROJECT/'data/med.pt',
                      official=PROJECT/'vendor/MedSAM2', dependencies=self.work/'linux-env')
        configs = model_configs(self.work, PROJECT, assets)
        self.assertEqual(len(configs), 5)
        for stage, updates in [('base', 40000), ('localizer', 5000), ('local', 20000)]:
            cfg = configs[f'configs/CHD1282-Q/20261115/{stage}.yaml']
            self.assertEqual(cfg['updates'], updates)
            self.assertEqual(cfg['accumulation'], 4)
            self.assertEqual(cfg['work_root'], str(self.work/'course/chd-segmentation'))
            self.assertEqual(cfg['groups'], str(self.work/'course/chd-segmentation/inputs/q1282/q_groups.json'))
            self.assertIn('CHD1282-Q', cfg['run_dir'])
        local = configs['configs/CHD1282-Q/20261115/local.yaml']
        self.assertEqual(local['parents']['base'], str(self.work/'course/chd-segmentation/runs/CHD1282-Q/20261115-offline/base/best.pt'))
        for dataset in ['CHD1282', 'ImageCHD']:
            cfg = configs[f'configs/{dataset}-MedSAM2/20261115/model.yaml']
            self.assertEqual((cfg['frames'], cfg['resolution'], cfg['epochs']), (8, 512, 75))
            self.assertFalse(cfg['freeze_image_encoder'])
            self.assertEqual(cfg['official_root'], str(assets['official']))
            self.assertEqual(cfg['work_root'], local['work_root'])


if __name__ == '__main__':
    unittest.main()
