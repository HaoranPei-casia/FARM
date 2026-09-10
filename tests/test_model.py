import torch

from farm.model import FARMReadout


def test_main_configuration_parameter_count_and_shape():
    model = FARMReadout(input_width=1024, hidden_width=32)
    assert sum(parameter.numel() for parameter in model.parameters()) == 33_985
    assert model(torch.randn(7, 10, 1024)).shape == (7,)


def test_attention_pooling_accepts_leading_batch_dimensions():
    model = FARMReadout(input_width=8, hidden_width=4)
    assert model(torch.randn(2, 5, 6, 8)).shape == (2, 5)
