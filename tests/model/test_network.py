"""Tests for DualHeadNetwork."""

from __future__ import annotations

import torch

from shogi_ai.model.config import ANIMAL_SHOGI_CONFIG, FULL_SHOGI_CONFIG, NetworkConfig
from shogi_ai.model.network import DualHeadNetwork, ResBlock


class TestResBlock:
    def test_output_shape_unchanged(self) -> None:
        block = ResBlock(channels=32)
        x = torch.randn(2, 32, 4, 3)
        out = block(x)
        assert out.shape == (2, 32, 4, 3)

    def test_skip_connection_gradient_flow(self) -> None:
        block = ResBlock(channels=16)
        x = torch.randn(1, 16, 4, 3, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


class TestDualHeadNetworkAnimalShogi:
    def test_output_shapes(self) -> None:
        net = DualHeadNetwork(ANIMAL_SHOGI_CONFIG)
        x = torch.randn(4, 14, 4, 3)
        policy, value = net(x)
        assert policy.shape == (4, 180)
        assert value.shape == (4, 1)

    def test_single_sample(self) -> None:
        net = DualHeadNetwork(ANIMAL_SHOGI_CONFIG)
        x = torch.randn(1, 14, 4, 3)
        policy, value = net(x)
        assert policy.shape == (1, 180)
        assert value.shape == (1, 1)

    def test_value_in_range(self) -> None:
        net = DualHeadNetwork(ANIMAL_SHOGI_CONFIG)
        x = torch.randn(8, 14, 4, 3)
        _, value = net(x)
        assert (value >= -1.0).all()
        assert (value <= 1.0).all()

    def test_forward_deterministic_in_eval(self) -> None:
        net = DualHeadNetwork(ANIMAL_SHOGI_CONFIG)
        net.eval()
        x = torch.randn(2, 14, 4, 3)
        with torch.no_grad():
            p1, v1 = net(x)
            p2, v2 = net(x)
        assert torch.equal(p1, p2)
        assert torch.equal(v1, v2)

    def test_gradient_flows_to_input(self) -> None:
        net = DualHeadNetwork(ANIMAL_SHOGI_CONFIG)
        x = torch.randn(1, 14, 4, 3, requires_grad=True)
        policy, value = net(x)
        loss = policy.sum() + value.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


class TestDualHeadNetworkFullShogi:
    def test_output_shapes(self) -> None:
        net = DualHeadNetwork(FULL_SHOGI_CONFIG)
        x = torch.randn(2, 43, 9, 9)
        policy, value = net(x)
        assert policy.shape == (2, 13689)  # ACTION_SPACE in full_shogi/moves.py
        assert value.shape == (2, 1)


class TestNetworkConfig:
    def test_animal_shogi_defaults(self) -> None:
        cfg = ANIMAL_SHOGI_CONFIG
        assert cfg.board_h == 4
        assert cfg.board_w == 3
        assert cfg.in_channels == 14
        assert cfg.action_size == 180

    def test_custom_config(self) -> None:
        cfg = NetworkConfig(
            board_h=5,
            board_w=5,
            in_channels=10,
            action_size=100,
            num_res_blocks=2,
            num_channels=32,
        )
        net = DualHeadNetwork(cfg)
        x = torch.randn(1, 10, 5, 5)
        policy, value = net(x)
        assert policy.shape == (1, 100)
        assert value.shape == (1, 1)
