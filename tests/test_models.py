import pytest

torch = pytest.importorskip("torch")

from src.models import (  # noqa: E402
    MultiscaleResidualFusionCNN,
    SleepStageCNN,
    SleepStageCNNGRU,
)


def test_sleep_stage_cnn_returns_three_class_logits():
    model = SleepStageCNN(in_channels=4, num_classes=3)
    x = torch.randn(5, 4, 64)

    logits = model(x)

    assert logits.shape == (5, 3)
    assert logits.dtype == torch.float32


def test_sleep_stage_cnn_rejects_wrong_input_rank():
    model = SleepStageCNN(in_channels=4, num_classes=3)

    with pytest.raises(ValueError, match="expects input shaped"):
        model(torch.randn(4, 64))


def test_sleep_stage_cnn_gru_returns_many_to_one_logits():
    model = SleepStageCNNGRU(
        in_channels=4,
        num_classes=3,
        filters=(8, 16),
        gru_hidden_size=12,
        target_position="center",
    )
    x = torch.randn(5, 7, 4, 64)

    logits = model(x)

    assert logits.shape == (5, 3)
    assert logits.dtype == torch.float32


def test_sleep_stage_cnn_gru_can_return_many_to_many_logits():
    model = SleepStageCNNGRU(
        in_channels=4,
        num_classes=3,
        filters=(8,),
        gru_hidden_size=10,
        output_mode="many_to_many",
    )
    x = torch.randn(5, 7, 4, 64)

    logits = model(x)

    assert logits.shape == (5, 7, 3)


def test_sleep_stage_cnn_gru_rejects_wrong_input_rank():
    model = SleepStageCNNGRU(in_channels=4, num_classes=3)

    with pytest.raises(ValueError, match="expects input shaped"):
        model(torch.randn(5, 4, 64))


def test_multiscale_residual_fusion_cnn_preserves_temporal_bins_and_backpropagates():
    model = MultiscaleResidualFusionCNN(
        in_channels=8,
        num_engineered_features=72,
        dropout=0.0,
    )
    raw_x = torch.randn(2, 8, 1920, requires_grad=True)
    engineered_x = torch.randn(2, 72, requires_grad=True)
    pooled_shapes = []
    hook = model.raw_pool.register_forward_hook(
        lambda _module, _inputs, output: pooled_shapes.append(tuple(output.shape))
    )

    logits = model(raw_x, engineered_x)
    logits.square().mean().backward()
    hook.remove()

    assert logits.shape == (2, 3)
    assert logits.dtype == torch.float32
    assert pooled_shapes == [(2, 64, 12)]
    assert raw_x.grad is not None
    assert engineered_x.grad is not None
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_multiscale_residual_fusion_cnn_uses_group_norm_not_batch_norm():
    model = MultiscaleResidualFusionCNN(
        in_channels=8,
        num_engineered_features=72,
    )
    group_norm_layers = [
        module for module in model.modules() if isinstance(module, torch.nn.GroupNorm)
    ]
    batch_norm_layers = [
        module
        for module in model.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    ]

    assert len(group_norm_layers) == 14
    assert not batch_norm_layers
    assert all(
        layer.num_channels % layer.num_groups == 0 for layer in group_norm_layers
    )


def test_multiscale_residual_fusion_cnn_default_parameter_count_is_compact():
    model = MultiscaleResidualFusionCNN(
        in_channels=8,
        num_engineered_features=72,
    )

    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    assert trainable_parameters == 252_227
    assert trainable_parameters < 300_000


def test_multiscale_residual_fusion_cnn_rejects_invalid_paired_inputs():
    model = MultiscaleResidualFusionCNN(
        in_channels=8,
        num_engineered_features=72,
    )

    with pytest.raises(ValueError, match="raw input shaped"):
        model(torch.randn(8, 1920), torch.randn(1, 72))
    with pytest.raises(ValueError, match="engineered input shaped"):
        model(torch.randn(1, 8, 1920), torch.randn(72))
    with pytest.raises(ValueError, match="same batch size"):
        model(torch.randn(2, 8, 1920), torch.randn(1, 72))
    with pytest.raises(ValueError, match="unexpected feature count"):
        model(torch.randn(1, 8, 1920), torch.randn(1, 71))
