"""
Projections for weight attacks.
"""
import common.torch
import torch
import numpy


class Projection:
    def __call__(self, model, perturbed_model, layers, quantization=None, quantization_contexts=None):
        """
        Projection.

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

    def reset(self):
        """
        Reset state of projection.
        """

        pass


class SequentialProjections(Projection):
    def __init__(self, projections):
        """
        Constructor.

        :param projections: list of projections
        :type projections: [Projection]
        """

        assert isinstance(projections, list)
        assert len(projections) > 0
        for projection in projections:
            assert isinstance(projection, Projection)

        self.projections = projections
        """ ([Projection]) Projections. """

    def __call__(self, model, perturbed_model, layers, quantization=None, quantization_contexts=None):
        """
        Projection.

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

        for projection in self.projections:
            projection(model, perturbed_model, layers, quantization, quantization_contexts)


class BoxProjection(Projection):
    def __init__(self, min_bound=0, max_bound=1):
        """
        Constructor.

        :param min_bound: minimum bound
        :param min_bound: float
        :param max_bound: maximum bound
        :type: max_bound: float
        """

        self.min_bound = min_bound
        """ (float) Minimum bound. """

        self.max_bound = max_bound
        """ (float) Maximum bound. """

    def __call__(self, model, perturbed_model, layers, quantization=None, quantization_contexts=None):
        """
        Projection.

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
        perturbed_parameters = list(perturbed_model.parameters())

        for i in layers:
            if self.max_bound is not None:
                perturbed_parameters[i].data = torch.min(torch.ones_like(perturbed_parameters[i].data) * self.max_bound, perturbed_parameters[i].data)
            if self.min_bound is not None:
                perturbed_parameters[i].data = torch.max(torch.ones_like(perturbed_parameters[i].data) * self.min_bound, perturbed_parameters[i].data)


class LayerWiseBoxProjection(Projection):
    def __init__(self, min_bounds=[], max_bounds=[]):
        """
        Constructor.

        :param min_bounds: minimum bound
        :param min_bounds: [float]
        :param max_bounds: maximum bound
        :type: max_bounds: [float]
        """

        assert len(min_bounds) > 0
        assert len(min_bounds) == len(max_bounds)

        self.min_bounds = min_bounds
        """ (float) Minimum bound. """

        self.max_bounds = max_bounds
        """ (float) Maximum bound. """

    def __call__(self, model, perturbed_model, layers, quantization=None, quantization_contexts=None):
        """
        Projection.

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

        perturbed_parameters = list(perturbed_model.parameters())
        j = 0
        for i in layers:
            perturbed_parameters[i].data = torch.min(torch.ones_like(perturbed_parameters[i].data) * self.max_bounds[j], perturbed_parameters[i].data)
            perturbed_parameters[i].data = torch.max(torch.ones_like(perturbed_parameters[i].data) * self.min_bounds[j], perturbed_parameters[i].data)
            j += 1


class LayerWiseL2Projection(Projection):
    def __init__(self, relative_epsilon):
        """
        Constructor.

        :param relative_epsilon: epsilon to project on
        :type relative_epsilon: float
        """

        self.relative_epsilon = relative_epsilon
        """ (float) Relative epsilon. """

        self.ord = 2
        """ (int) Project order. """

    def __call__(self, model, perturbed_model, layers, quantization=None, quantization_contexts=None):
        """
        Projection.

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

        for i in layers:
            perturbation = perturbed_parameters[i].data - parameters[i].data
            size = list(perturbation.shape)
            epsilon = self.relative_epsilon*torch.norm(parameters[i].data.view(-1), self.ord)
            perturbation = common.torch.project_ball(perturbation.view(1, -1), epsilon=epsilon, ord=self.ord).view(-1)
            perturbation = perturbation.view(size)
            perturbed_parameters[i].data = parameters[i].data + perturbation
