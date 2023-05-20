import torch
import common.summary
import common.torch
from common.log import log
from .progress import ProgressBar


def reset(model):
    """
    Reset BN statistics.

    :param model: model
    :type model: torch.mm.Module
    """

    if hasattr(model, 'running_var'):
        model.running_var.data.fill_(1)
    if hasattr(model, 'running_mean'):
        model.running_mean.data.fill_(0)
    if hasattr(model, 'num_batches_tracked'):
        model.num_batches_tracked.data.fill_(0)

    for module in model.children():
        reset(module)


def momentum(model, value):
    """
    Set momentum for BN.

    See https://github.com/pytorch/pytorch/blob/fa153184c8f70259337777a1fd1d803c7325f758/aten/src/ATen/native/Normalization.cpp.

    :param model: model
    :type model: torch.nn.Module
    :param value: momentum value
    :type value: value
    """

    if hasattr(model, 'momentum'):
        model.momentum = value

    for module in model.children():
        momentum(module, value)
