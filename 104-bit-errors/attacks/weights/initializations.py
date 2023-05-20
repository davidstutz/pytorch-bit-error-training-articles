"""
Initializations for weight attacks.
"""
import common.torch
import torch
import torch.utils.data
import numpy
import common.numpy
from common.log import log


class Initialization:
    """
    Interface for initialization.
    """

    def __call__(self, model, perturbed_model, layers, quantization=None, quantization_contexts=None):
        """
        Initialization.

        :param model: original model
        :type model: torch.nn.Module
        :param perturbed_model: perturbed model
        :type perturbed_model: torch.nn.Module
        :param layers: layers to initialize
        :type layers: [int]
        :param quantization: quantization if required
        :type quantization: Quantization
        :param quantization_contexts: quantization contexts for each layer
        :type quantization_contexts: [dict]
        """

        raise NotImplementedError()


class LayerWiseL2UniformNormInitialization(Initialization):
    """
    Uniform initialization, wrt. norm and direction, in L_2 ball.
    """

    def __init__(self, relative_epsilon, randomness=None):
        """
        Constructor.

        :param epsilon: epsilon to project on
        :type epsilon: float
        """

        self.relative_epsilon = relative_epsilon
        """ (float) Relative epsilon. """

        self.callable = common.torch.uniform_norm
        """ (callable) Sampler. """

        self.ord = 2
        """ (float) Norm. """

        self.randomness_ = randomness
        """ (torch.utils.data.DataLoader) Data loader for random values. """

        self.randomness = None
        """ (torch.utils.data.DataLoader) Data loader for random values. """

        if self.randomness_ is not None:
            assert isinstance(self.randomness_, torch.utils.data.DataLoader)
            self.randomness = iter(self.randomness_)

    def __call__(self, model, perturbed_model, layers, quantization=None, quantization_contexts=None):
        """
        Initialization.

        :param model: original model
        :type model: torch.nn.Module
        :param perturbed_model: perturbed model
        :type perturbed_model: torch.nn.Module
        :param layers: layers to initialize
        :type layers: [int]
        :param quantization: quantization if required
        :type quantization: Quantization
        :param quantization_contexts: quantization contexts for each layer
        :type quantization_contexts: [dict]
        """

        assert len(layers) > 0

        parameters = list(model.parameters())
        perturbed_parameters = list(perturbed_model.parameters())
        assert len(parameters) == len(perturbed_parameters)

        cuda = common.torch.is_cuda(model)
        if self.randomness is not None:
            try:
                seed = next(self.randomness)[0].item()
            except StopIteration as e:
                log('resetting seeds')
                self.randomness = iter(self.randomness_)
                seed = next(self.randomness)[0].item()
        else:
            seed = numpy.random.randint(0, 2**32 - 1)
        common.torch.torch_numpy_seed(seed)

        for i in layers:
            size = list(parameters[i].data.shape)
            epsilon = self.relative_epsilon*torch.norm(parameters[i].data.view(-1), self.ord)
            perturbed_parameters[i].data = parameters[i].data + self.callable(1, numpy.prod(size), epsilon=epsilon, ord=self.ord, cuda=cuda).view(size)


class BitRandomInitialization(Initialization):
    """
    Random bit flips.
    """

    def __init__(self, probability, randomness=None):
        """
        Initializer for bit flips.

        :param probability: probability
        :type probability: float
        """

        self.probability = probability
        """ (float) Probability of flip. """

        self.randomness_ = randomness
        """ (torch.utils.data.DataLoader) Data loader for random values. """

        self.randomness = None
        """ (torch.utils.data.DataLoader) Data loader for random values. """

        if self.randomness_ is not None:
            assert isinstance(self.randomness_, torch.utils.data.DataLoader)
            self.randomness = iter(self.randomness_)

    def __call__(self, model, perturbed_model, layers, quantization=None, quantization_contexts=None):
        """
        Initialization.

        :param model: original model
        :type model: torch.nn.Module
        :param perturbed_model: perturbed model
        :type perturbed_model: torch.nn.Module
        :param layers: layers to initialize
        :type layers: [int]
        :param quantization: quantization if required
        :type quantization: Quantization
        :param quantization_contexts: quantization contexts for each layer
        :type quantization_contexts: [dict]
        """

        assert len(layers) > 0
        assert quantization is not None
        assert quantization_contexts is not None
        assert isinstance(quantization_contexts, list)

        parameters = list(model.parameters())
        perturbed_parameters = list(perturbed_model.parameters())
        assert len(parameters) == len(perturbed_parameters)

        # Main reason for overhead is too avoid too many calls to cuda() and rand()!
        precision = quantization.type_precision
        cuda = common.torch.is_cuda(model)
        n, _, _, _ = common.torch.parameter_sizes(model, layers=None)

        if self.randomness is not None:
            try:
                seed = next(self.randomness)[0].item()
            except StopIteration as e:
                log('resetting seeds')
                self.randomness = iter(self.randomness_)
                seed = next(self.randomness)[0].item()
        else:
            seed = numpy.random.randint(0, 2**32 - 1)
        common.torch.torch_numpy_seed(seed)
        if cuda:
            random = torch.cuda.FloatTensor(n, precision).uniform_(0, 1)
        else:
            random = torch.FloatTensor(n, precision).uniform_(0, 1)
        #log('hash: %s' % hashlib.sha256(random.data.cpu().numpy().tostring()).hexdigest())

        n_i = 0
        for i in layers:
            weights = parameters[i].data
            size_i = list(weights.shape)
            # important: quantization at this point only depends on weights, not on the perturbed weights!
            quantized_weights, _ = quantization.quantize(weights, quantization_contexts[i])
            perturbed_quantized_weights = common.torch.int_random_flip(quantized_weights, self.probability, self.probability,
                                                                       protected_bits=quantization.protected_bits,
                                                                       rand=random[n_i:n_i + numpy.prod(size_i)].view(size_i + [precision]))

            perturbed_dequantized_weights = quantization.dequantize(perturbed_quantized_weights, quantization_contexts[i])
            perturbed_parameters[i].data = perturbed_dequantized_weights
            n_i += numpy.prod(size_i)
