import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import argparse
import numpy
import common.state
import common.test
import common.eval
import common.quantization
import torchvision
import torch.utils.data


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

    def get_parser(self):
        """
        Get parser.

        :return: parser
        :rtype: argparse.ArgumentParser
        """

        parser = argparse.ArgumentParser(description='Quantize model.')
        parser.add_argument('--model_file', type=str)
        parser.add_argument('--precision', type=int, default=16)
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

        labels = numpy.array(self.testset.targets)
        probabilities = common.test.test(model, self.testloader, cuda=self.args.cuda)
        eval = common.eval.CleanEvaluation(probabilities, labels)
        print('Test error before quantization: ', eval.test_error())

        quantization = common.quantization.AdaptiveAlternativeUnsymmetricUnsignedRoundedFixedPointQuantization(self.args.precision)
        dequantized_model, contexts = common.quantization.quantize(quantization, model)

        probabilities = common.test.test(dequantized_model, self.testloader, cuda=self.args.cuda)
        eval = common.eval.CleanEvaluation(probabilities, labels)
        print('Test error _after_ quantization: ', eval.test_error())


if __name__ == '__main__':
    program = Main()
    program.main()