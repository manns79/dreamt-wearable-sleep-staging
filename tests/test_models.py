import pytest

torch = pytest.importorskip("torch")

from src.models import SleepStageCNN  # noqa: E402


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
