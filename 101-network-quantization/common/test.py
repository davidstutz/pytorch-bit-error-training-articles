import torch
import common.torch
import common.numpy
from common.progress import ProgressBar


def test(model, testset, eval=True, loss=True, operators=None, cuda=False):
    """
    Test a model on a clean or adversarial dataset.

    :param model: model
    :type model: torch.nn.Module
    :param testset: test set
    :type testset: torch.utils.data.DataLoader
    :param cuda: use CUDA
    :type cuda: bool
    """

    assert model.training is not eval
    assert len(testset) > 0
    assert (cuda and common.torch.is_cuda(model)) or (not cuda and not common.torch.is_cuda(model))

    progress = ProgressBar()
    probabilities = None

    # should work with and without labels
    for b, data in enumerate(testset):
        targets = None
        if isinstance(data, tuple) or isinstance(data, list):
            inputs = data[0]
            targets = data[1]
        else:
            inputs = data

        assert isinstance(inputs, torch.Tensor)

        inputs = common.torch.as_variable(inputs, cuda)
        targets = common.torch.as_variable(targets, cuda)

        logits = model.forward(inputs, operators=operators)
        probabilities_ = common.torch.softmax(logits, dim=1).detach().cpu().numpy()
        probabilities = common.numpy.concatenate(probabilities, probabilities_)

        if targets is not None and loss:
            targets = common.torch.as_variable(targets, cuda)
            error = common.torch.classification_error(logits, targets)
            loss = common.torch.classification_loss(logits, targets)
            progress('test', b, len(testset), info='error=%g loss=%g' % (error.item(), loss.item()))
        else:
            progress('test', b, len(testset))

    return probabilities
