import torch
from torch.testing import assert_allclose

from dsntnn import dsnt, linear_expectation


SIMPLE_INPUT = torch.Tensor([[[
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.1, 0.0],
    [0.0, 0.0, 0.1, 0.6, 0.1],
    [0.0, 0.0, 0.0, 0.1, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0],
]]])

SIMPLE_OUTPUT = torch.Tensor([[[0.4, 0.0]]])

SIMPLE_TARGET = torch.Tensor([[[0.5, 0.5]]])

# Expected dloss/dinput when using MSE with target (0.5, 0.5)
SIMPLE_GRAD_INPUT = torch.Tensor([[[
    [0.4800, 0.4400, 0.4000, 0.3600, 0.3200],
    [0.2800, 0.2400, 0.2000, 0.1600, 0.1200],
    [0.0800, 0.0400, 0.0000, -0.0400, -0.0800],
    [-0.1200, -0.1600, -0.2000, -0.2400, -0.2800],
    [-0.3200, -0.3600, -0.4000, -0.4400, -0.4800],
]]])


def test_dsnt_forward():
    expected = SIMPLE_OUTPUT
    actual = dsnt(SIMPLE_INPUT)
    assert_allclose(actual, expected)


def test_dsnt_backward():
    mse = torch.nn.MSELoss()

    in_var = SIMPLE_INPUT.detach().requires_grad_(True)
    output = dsnt(in_var)

    loss = mse(output, SIMPLE_TARGET)
    loss.backward()

    assert_allclose(in_var.grad, SIMPLE_GRAD_INPUT)


def test_dsnt_cuda():
    mse = torch.nn.MSELoss()

    in_var = SIMPLE_INPUT.detach().cuda().requires_grad_(True)

    expected_output = SIMPLE_OUTPUT.cuda()
    output = dsnt(in_var)
    assert_allclose(output, expected_output)

    target_var = SIMPLE_TARGET.cuda()
    loss = mse(output, target_var)
    loss.backward()

    expected_grad = SIMPLE_GRAD_INPUT.cuda()
    assert_allclose(in_var.grad, expected_grad)


def test_dsnt_3d():
    inp = torch.Tensor([[
        [[
            [0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00],
        ], [
            [0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00],
            [1.00, 0.00, 0.00],
        ], [
            [0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00],
        ]]
    ]])

    expected = torch.Tensor([[[-2/3, 2/3, 0]]])
    assert_allclose(dsnt(inp), expected)


def test_dsnt_linear_expectation():
    probs = torch.Tensor([[[
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.2, 0.3, 0.0],
        [0.0, 0.0, 0.3, 0.2, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ]]])
    values = [torch.arange(d, dtype=probs.dtype, device=probs.device) for d in probs.size()[2:]]

    expected = torch.Tensor([[[1.5, 2.5]]])
    actual = linear_expectation(probs, values)
    assert_allclose(actual, expected)
