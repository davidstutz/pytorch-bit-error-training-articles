import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import argparse
import attacks.weights
import attacks.weights.norms
import attacks.weights.objectives
import attacks.weights.projections
import attacks.weights.initializations
import numpy
import common.state
import common.test
import common.eval
import common.progress
import torchvision
import torch.utils.data
from matplotlib import pyplot as plt


class Main:
    def __init__(self, args=None):
        """
        Initialize.

        :param args: optional arguments if not to use sys.argv
        :type args: [str]
        """

        self.args = None
        """ Arguments of program. """

        parser = self.get_parser()
        if args is not None:
            self.args = parser.parse_args(args)
        else:
            self.args = parser.parse_args()

        test_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=128, shuffle=False, num_workers=2)

        test_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        self.adversarialset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        self.adversarialset = torch.utils.data.Subset(self.adversarialset, range(0, 1000))
        self.adversarialloader = torch.utils.data.DataLoader(self.adversarialset, batch_size=128, shuffle=False, num_workers=2)

    def get_parser(self):
        """
        Get parser.

        :return: parser
        :rtype: argparse.ArgumentParser
        """

        parser = argparse.ArgumentParser(description='Inject bit errors into model.')
        parser.add_argument('--p', type=float, default=0.01)
        parser.add_argument('--attempts', type=int, default=10)
        parser.add_argument('--precision', type=int, default=8)
        parser.add_argument('--model_file', type=str)
        parser.add_argument('--no-cuda', action='store_false', dest='cuda', default=True, help='do not use cuda')

        return parser

    def main(self):
        """
        Main.
        """

        state = common.state.State.load(self.args.model_file)
        print('read %s' % self.args.model_file)

        model = state.model
        model.eval()
        print(model)

        if self.args.cuda:
            model = model.cuda()

        attack = attacks.weights.RandomAttack()
        attack.epochs = 1
        attack.progress = common.progress.ProgressBar()
        attack.layers = None  # could be a list of layer indices
        attack.get_layers = None  # a function that returns a list of layer indices
        attack.initialization = attacks.weights.initializations.BitRandomInitialization(probability=self.args.p)
        attack.projection = None
        attack.quantization = common.quantization.AdaptiveAlternativeUnsymmetricUnsignedRoundedFixedPointQuantization(self.args.precision)
        attack.quantization_contexts = None
        attack.training = False
        attack.norm = attacks.weights.HammingNorm()

        objective = attacks.weights.objectives.UntargetedF0Objective()
        probabilities = common.test.test(model, self.testloader, cuda=self.args.cuda)
        perturbed_models = common.test.attack_weights(model, self.adversarialloader, attack, objective, attempts=self.args.attempts, cuda=self.args.cuda)

        labels = numpy.array(self.testset.targets)
        eval = common.eval.CleanEvaluation(probabilities, labels)

        adversarial_evals = []
        for perturbed_model in perturbed_models:
            if self.args.cuda:
                perturbed_model = perturbed_model.cuda()
            perturbed_probabilities = common.test.test(perturbed_model, self.testloader, cuda=self.args.cuda)
            adversarial_eval = common.eval.AdversarialWeightsEvaluation(probabilities, perturbed_probabilities, labels)
            adversarial_evals.append(adversarial_eval)
        adversarial_eval = common.eval.EvaluationStatistics(adversarial_evals)

        print('error: %g' % eval.test_error())
        print('robust error: %g' % adversarial_eval('robust_test_error', 'mean')[0])


if __name__ == '__main__':
    program = Main()
    program.main()