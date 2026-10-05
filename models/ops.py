import torch
import torch.nn as nn
from .config import FLAGS

def make_divisible(v, divisor=8, min_value=None):
    """Round a channel count to the nearest multiple of `divisor`
    (never rounding down by more than 10%)."""
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def fake_quantize_weight(w, bits, per_channel=True):
    """Symmetric fake quantization for weights, straight-through gradient estimator.
    bits >= 32 is treated as full precision (no-op)."""
    if bits >= 32:
        return w

    qmax = 2 ** (bits - 1) - 1  # symmetric range, e.g. bits=8 -> [-127, 127]

    if per_channel:
        # scale computed per output channel (dim 0) -- matches how we slice
        # channels per width_mult, so a sliced sub-network gets its own scale
        dims = tuple(range(1, w.dim()))
        max_val = w.detach().abs().amax(dim=dims, keepdim=True).clamp(min=1e-8)
    else:
        max_val = w.detach().abs().max().clamp(min=1e-8)

    scale = max_val / qmax
    w_q = torch.clamp(torch.round(w / scale), -qmax - 1, qmax) * scale

    # straight-through estimator: forward uses w_q, backward gradient
    # passes through as if this were the identity function
    return w + (w_q - w).detach()


def fake_quantize_act(x, bits):
    """Unsigned fake quantization for post-ReLU activations (always >= 0).
    Per-tensor, dynamic range (recomputed every forward from the current
    batch), straight-through gradient estimator. bits >= 32 is a no-op."""
    if bits >= 32:
        return x

    qmax = 2 ** bits - 1  # unsigned range, e.g. bits=8 -> [0, 255]
    max_val = x.detach().max().clamp(min=1e-8)
    scale = max_val / qmax
    x_q = torch.clamp(torch.round(x / scale), 0, qmax) * scale
    return x + (x_q - x).detach()


class ResnetConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, is_stem=False):
        super(ResnetConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.max_in_channels = in_channels
        self.max_out_channels = out_channels
        self.groups_ = groups
        self.is_stem = is_stem
        self.width_mult = max(FLAGS.width_mult_range)
        self.bit_width = max(FLAGS.bit_width_list)

        self.quant_cache_enabled = False
        self._quant_cache = {}

    def clear_quant_cache(self):
        self._quant_cache = {}

    def _quantized_weight(self, weight):
        if self.bit_width >= 32:
            return weight
        if self.quant_cache_enabled:
            key = (self.width_mult, self.bit_width)
            if key in self._quant_cache:
                return self._quant_cache[key]
            wq = fake_quantize_weight(weight, self.bit_width)
            self._quant_cache[key] = wq
            return wq
        return fake_quantize_weight(weight, self.bit_width)

    def forward(self, input):
        in_ratio = 1.0 if self.is_stem else self.width_mult
        self.in_channels = make_divisible(self.max_in_channels * in_ratio, FLAGS.width_divisor)
        self.out_channels = make_divisible(self.max_out_channels * self.width_mult, FLAGS.width_divisor)
        self.groups = self.in_channels if self.groups_ != 1 else 1

        weight = self.weight[:self.out_channels, :self.in_channels, :, : ]
        weight = self._quantized_weight(weight)

        bias = self.bias[:self.out_channels] if self.bias is not None else None
        return nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.groups)


class ResnetLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(ResnetLinear, self).__init__(in_features, out_features, bias=bias)
        self.max_in_features = in_features
        self.max_out_features = out_features
        self.width_mult = max(FLAGS.width_mult_range)
        self.bit_width = max(FLAGS.bit_width_list)

        self.quant_cache_enabled = False
        self._quant_cache = {}

    def clear_quant_cache(self):
        self._quant_cache = {}

    def _quantized_weight(self, weight):
        if self.bit_width >= 32:
            return weight
        if self.quant_cache_enabled:
            key = (self.width_mult, self.bit_width)
            if key in self._quant_cache:
                return self._quant_cache[key]
            wq = fake_quantize_weight(weight, self.bit_width)
            self._quant_cache[key] = wq
            return wq
        return fake_quantize_weight(weight, self.bit_width)

    def forward(self, input):
        self.in_features = make_divisible(self.max_in_features * self.width_mult, FLAGS.width_divisor)
        self.out_features = self.max_out_features

        weight = self.weight[:self.out_features, :self.in_features]
        weight = self._quantized_weight(weight)

        bias = self.bias if self.bias is not None else None
        return nn.functional.linear(input, weight, bias)


class ResnetBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features):
        super(ResnetBatchNorm2d, self).__init__(num_features, affine=True)
        self.max_features = num_features
        self.width_mult = max(FLAGS.width_mult_range)
        self.bit_width = max(FLAGS.bit_width_list)
        self.calibrated_running_mean = {}
        self.calibrated_running_var = {}

    def forward(self, input):
        c = make_divisible(self.max_features * self.width_mult, FLAGS.width_divisor)
        weight = self.weight[:c]
        bias = self.bias[:c]

        key = (self.width_mult, self.bit_width)
        if (not self.training) and key in self.calibrated_running_mean:
            mean = self.calibrated_running_mean[key]
            var = self.calibrated_running_var[key]
            return nn.functional.batch_norm(
                input, mean, var, weight, bias, training=False, eps=self.eps)

        return nn.functional.batch_norm(
            input, None, None, weight, bias,
            training=True, momentum=0.0, eps=self.eps)


class ActQuant(nn.Module):
    def __init__(self):
        super().__init__()
        self.bit_width = max(FLAGS.bit_width_list)

    def forward(self, x):
        if self.bit_width >= 32:
            return x
        return fake_quantize_act(x, self.bit_width)


def clear_all_quant_caches(model):
    core = model.module if hasattr(model, 'module') else model
    for m in core.modules():
        if hasattr(m, 'clear_quant_cache'):
            m.clear_quant_cache()


def set_quant_cache_enabled(model, enabled):
    core = model.module if hasattr(model, 'module') else model
    for m in core.modules():
        if hasattr(m, 'quant_cache_enabled'):
            m.quant_cache_enabled = enabled
    if enabled:
        clear_all_quant_caches(model)
