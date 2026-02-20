import torch
import pytest

# import your classes here:
# from your_module import Conv, Bottleneck, C2f

def _zero_module_params(m: torch.nn.Module):
    for p in m.parameters():
        torch.nn.init.zeros_(p)

def test_conv_bias_is_false():
    m = Conv(3, 16)
    assert m.conv.bias is None, "Conv2d bias should be None because bias=False"

def test_conv_activation_toggle():
    m1 = Conv(3, 16, activation=True)
    m2 = Conv(3, 16, activation=False)
    assert m1.act.__class__.__name__ == "SiLU"
    assert m2.act.__class__.__name__ == "Identity"

def test_conv_shape_stride1_padding1_kernel3():
    m = Conv(3, 16, kernel_size=3, stride=1, padding=1)
    x = torch.randn(2, 3, 64, 64)
    y = m(x)
    assert y.shape == (2, 16, 64, 64)

def test_conv_shape_stride2_padding1_kernel3():
    m = Conv(3, 16, kernel_size=3, stride=2, padding=1)
    x = torch.randn(2, 3, 64, 64)
    y = m(x)
    # For even sizes, stride=2 halves
    assert y.shape == (2, 16, 32, 32)

def test_conv_groups_validity():
    # groups must divide both in_channels and out_channels
    m = Conv(8, 16, groups=4)  # ok: 8%4==0, 16%4==0
    x = torch.randn(1, 8, 32, 32)
    y = m(x)
    assert y.shape == (1, 16, 32, 32)

def test_bottleneck_shortcut_true_identity_when_zero_weights():
    m = Bottleneck(16, 16, shortcut=True)
    m.eval()
    _zero_module_params(m)  # conv outputs become 0, so output should be x + 0 = x

    x = torch.randn(2, 16, 32, 32)
    with torch.no_grad():
        y = m(x)
    assert torch.allclose(y, x, atol=1e-6), "With zero weights and shortcut=True, Bottleneck should return input"

def test_bottleneck_shortcut_false_zero_when_zero_weights():
    m = Bottleneck(16, 16, shortcut=False)
    m.eval()
    _zero_module_params(m)

    x = torch.randn(2, 16, 32, 32)
    with torch.no_grad():
        y = m(x)
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-6), "With zero weights and shortcut=False, output should be zeros"

def test_c2f_output_shape():
    m = C2f(in_channels=32, out_channels=64, num_bottlenecks=2, shortcut=True)
    x = torch.randn(2, 32, 40, 40)
    y = m(x)
    assert y.shape == (2, 64, 40, 40)

def test_c2f_bottlenecks_are_distinct_modules():
    m = C2f(in_channels=32, out_channels=64, num_bottlenecks=3, shortcut=True)
    ids = [id(b) for b in m.m]
    assert len(set(ids)) == len(ids), (
        "Bottlenecks are shared! You probably used [block] * n. "
        "Use a list comprehension to create distinct modules."
    )

def test_c2f_backward_pass():
    m = C2f(in_channels=32, out_channels=64, num_bottlenecks=2, shortcut=True)
    x = torch.randn(2, 32, 20, 20, requires_grad=True)
    y = m(x).mean()
    y.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

def test_bn_train_vs_eval_difference():
    m = Conv(3, 8)
    x = torch.randn(4, 3, 32, 32)

    m.train()
    y_train = m(x)

    m.eval()
    y_eval = m(x)

    # Not guaranteed to differ a lot, but usually not identical
    assert not torch.allclose(y_train, y_eval), "BN typically makes train/eval outputs differ"
