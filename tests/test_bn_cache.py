import os
import tempfile
import unittest
import torch
import torch.nn as nn

from models.ops import ResnetBatchNorm2d
from models.train import (
    save_bn_calibration,
    load_bn_calibration,
    recalibrate_bn,
)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = ResnetBatchNorm2d(16)
        self.fc = nn.Linear(16, 10)

    def set_width(self, width):
        self.bn1.width_mult = width

    def set_bit_width(self, bit_width):
        self.bn1.bit_width = bit_width

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = x.mean([2, 3])
        return self.fc(x)


class BatchNormCacheTests(unittest.TestCase):
    def test_save_and_load_roundtrip(self):
        model1 = DummyModel()
        key1 = (1.0, 32)
        key2 = (0.5, 32)

        mean1 = torch.tensor([1.0, 2.0, 3.0, 4.0])
        var1 = torch.tensor([0.1, 0.2, 0.3, 0.4])
        model1.bn1.calibrated_running_mean[key1] = mean1
        model1.bn1.calibrated_running_var[key1] = var1

        mean2 = torch.tensor([5.0, 6.0])
        var2 = torch.tensor([0.5, 0.6])
        model1.bn1.calibrated_running_mean[key2] = mean2
        model1.bn1.calibrated_running_var[key2] = var2

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = os.path.join(tmpdir, "test_bn_cache.pt")
            saved = save_bn_calibration(model1, cache_path=cache_file)
            self.assertEqual(saved, {key1, key2})
            self.assertTrue(os.path.isfile(cache_file))

            model2 = DummyModel()
            self.assertEqual(len(model2.bn1.calibrated_running_mean), 0)

            loaded = load_bn_calibration(model2, cache_path=cache_file)
            self.assertEqual(loaded, {key1, key2})
            self.assertTrue(torch.equal(model2.bn1.calibrated_running_mean[key1], mean1))
            self.assertTrue(torch.equal(model2.bn1.calibrated_running_var[key1], var1))
            self.assertTrue(torch.equal(model2.bn1.calibrated_running_mean[key2], mean2))
            self.assertTrue(torch.equal(model2.bn1.calibrated_running_var[key2], var2))

    def test_incremental_merge_preserves_previous_configs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = os.path.join(tmpdir, "test_bn_cache.pt")

            # First run: calibrate (1.0, 32)
            model1 = DummyModel()
            key1 = (1.0, 32)
            model1.bn1.calibrated_running_mean[key1] = torch.tensor([1.0])
            model1.bn1.calibrated_running_var[key1] = torch.tensor([0.1])
            save_bn_calibration(model1, cache_path=cache_file)

            # Second run: calibrate (0.75, 32) on a fresh model
            model2 = DummyModel()
            key2 = (0.75, 32)
            model2.bn1.calibrated_running_mean[key2] = torch.tensor([2.0])
            model2.bn1.calibrated_running_var[key2] = torch.tensor([0.2])
            save_bn_calibration(model2, cache_path=cache_file)

            # Third model: load both!
            model3 = DummyModel()
            loaded = load_bn_calibration(model3, cache_path=cache_file)
            self.assertEqual(loaded, {key1, key2})
            self.assertIn(key1, model3.bn1.calibrated_running_mean)
            self.assertIn(key2, model3.bn1.calibrated_running_mean)

    def test_missing_cache_file_returns_empty_set_safely(self):
        model = DummyModel()
        loaded = load_bn_calibration(model, cache_path="nonexistent_cache.pt")
        self.assertEqual(loaded, set())


if __name__ == "__main__":
    unittest.main()

