import pytest

torch = pytest.importorskip("torch")

from src.models import SleepStageCNN, SleepStageCNNGRU  # noqa: E402


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
