"""确保增强只影响图像，并与固定版本参考数学实现一致。"""
import sys
from pathlib import Path
import unittest
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "05_src"))
from appearance import BiasField, MonotoneBezier


class AppearanceTests(unittest.TestCase):
    def test_label_unchanged(self):
        image = torch.randn(1, 12, 16, 20)
        label = torch.randint(0, 8, (1, 12, 16, 20))
        before = label.clone()
        for transform in (BiasField(), MonotoneBezier()):
            result = transform(image=image.clone(), segmentation=label)
            self.assertTrue(torch.equal(before, result["segmentation"]))
            self.assertTrue(torch.isfinite(result["image"]).all())

    def test_monotonic_range_and_constant(self):
        transform = MonotoneBezier()
        img = torch.linspace(-3, 4, 1000).reshape(1, 10, 10, 10)
        for seed in range(5):
            np.random.seed(seed)
            out = transform(image=img)["image"].flatten()
            self.assertTrue((torch.diff(out) >= -1e-6).all())
            self.assertAlmostEqual(float(out[0]), -3)
            self.assertAlmostEqual(float(out[-1]), 4)
        constant = torch.ones_like(img) * 5
        self.assertTrue(torch.equal(transform(image=constant)["image"], constant))

    def test_reference_equivalence(self):
        # 参考文件只用于本地核对，不作为训练代码依赖。
        if not all((ROOT / "02_materials/code_reference" / name).is_file() for name in ["bias_field_transform.py", "bezier_dualnorm_transform.py"]):
            self.skipTest("GitHub 不分发第三方参考脚本；本地有脚本时再运行数值对照测试")
        sys.path.insert(0, str(ROOT / "02_materials/code_reference"))
        from bias_field_transform import BiasFieldTransform3D
        from bezier_dualnorm_transform import DualNormBezierIntensityTransform3D
        img = torch.linspace(-2, 3, 120).reshape(1, 4, 5, 6)
        for own, ref in [(BiasField(), BiasFieldTransform3D()), (MonotoneBezier(), DualNormBezierIntensityTransform3D(mode="similar", lut_size=1000))]:
            for seed in [0, 17, 42]:
                np.random.seed(seed)
                a = own(image=img.clone())["image"]
                np.random.seed(seed)
                b = ref(image=img.clone())["image"]
                torch.testing.assert_close(a, b, rtol=0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
