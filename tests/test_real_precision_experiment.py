import unittest

import torch
from torch import nn

from scripts.experiment_real_precision import quantize_static_int8


class RealPrecisionExperimentTests(unittest.TestCase):
    def test_cpu_int8_conversion_uses_native_quantized_conv_and_linear(self):
        model = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, 10),
        ).eval()
        calibration_batch = (torch.randn(2, 3, 32, 32), torch.zeros(2, dtype=torch.long))

        quantized = quantize_static_int8(model, [calibration_batch], calibration_batches=1)
        call_modules = [
            quantized.get_submodule(node.target)
            for node in quantized.graph.nodes
            if node.op == "call_module"
        ]

        self.assertTrue(any(isinstance(module, torch.ao.nn.quantized.Conv2d) for module in call_modules))
        self.assertTrue(any(isinstance(module, torch.ao.nn.quantized.Linear) for module in call_modules))
        self.assertFalse(any(isinstance(module, nn.Conv2d) for module in call_modules))
        self.assertFalse(any(isinstance(module, nn.Linear) for module in call_modules))


if __name__ == "__main__":
    unittest.main()